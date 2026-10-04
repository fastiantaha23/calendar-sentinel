"""Sandbox calendar tools + the single execution gateway.

Every tool call goes through ToolGateway.call(), which owns:
  * argument validation result handling      * autonomy gate (restricted tools never run)
  * timeouts / retries / output validation   * Arena fault injection (first matching operation)
No tool touches a real calendar, mailbox or account: state lives in an in-memory, per-run Sandbox.
"""
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import ROOT, settings
from app.models import ToolTrace

_TIME_RE = re.compile(r'^([01]\d|2[0-3]):[0-5]\d$')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _check_date(value: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ValueError('date must be a string formatted YYYY-MM-DD')
    date.fromisoformat(value)  # rejects impossible dates such as 2026-02-30
    return value


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.casefold()).strip()


# --------------------------------------------------------------------------- arguments
class ArgModel(BaseModel):
    model_config = ConfigDict(extra='ignore')


class EntryIn(ArgModel):
    title: str = Field(min_length=1, max_length=120)
    date: str
    start_time: Optional[str] = None
    duration_minutes: int = Field(default=60, ge=5, le=1440)
    date_evidence: str = Field(min_length=1, max_length=300)

    @field_validator('date')
    @classmethod
    def v_date(cls, v):
        return _check_date(v)

    @field_validator('start_time')
    @classmethod
    def v_time(cls, v):
        if v is not None and not _TIME_RE.match(v):
            raise ValueError('start_time must be HH:MM (24h) or null')
        return v


class CreateEntriesArgs(ArgModel):
    entries: list[EntryIn] = Field(min_length=1, max_length=5)


class ReminderIn(ArgModel):
    entry_id: str = Field(min_length=1, max_length=20)
    days_before: int = Field(ge=0, le=30)


class AddRemindersArgs(ArgModel):
    reminders: list[ReminderIn] = Field(min_length=1, max_length=5)


class ListEntriesArgs(ArgModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    @field_validator('from_date', 'to_date')
    @classmethod
    def v_date(cls, v):
        return None if v is None else _check_date(v)


class CheckConflictsArgs(ArgModel):
    date: str
    start_time: str
    duration_minutes: int = Field(default=60, ge=5, le=1440)

    @field_validator('date')
    @classmethod
    def v_date(cls, v):
        return _check_date(v)

    @field_validator('start_time')
    @classmethod
    def v_time(cls, v):
        if not _TIME_RE.match(v):
            raise ValueError('start_time must be HH:MM (24h)')
        return v


class AnyArgs(ArgModel):
    """Restricted tools are never executed, so their arguments are not interpreted."""


class ToolOutput(BaseModel):
    """Output contract every tool must satisfy; malformed results are retried, never trusted."""
    model_config = ConfigDict(extra='forbid')
    ok: bool
    summary: str = Field(min_length=1, max_length=400)
    data: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- sandbox
def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


class Sandbox:
    """Isolated in-memory calendar for ONE run (seeded from data/sample_data.json)."""

    def __init__(self, today: date, corpus: list[str], seed: Optional[list[dict]] = None):
        self.today = today
        self.corpus = [normalize(c) for c in corpus]
        self.entries: dict[str, dict] = {}
        self.reminders: list[dict] = []
        self._n = 0
        for item in (seed or []):
            when = today + timedelta(days=int(item.get('offset_days', 1)))
            self._add(item['title'], when.isoformat(), item.get('start_time'), int(item.get('duration_minutes', 60)))

    def _add(self, title, day, start, duration) -> dict:
        self._n += 1
        entry = {'id': f'E{self._n}', 'title': title, 'date': day, 'start_time': start, 'duration_minutes': duration}
        self.entries[entry['id']] = entry
        return entry

    def grounded(self, evidence: str) -> bool:
        needle = normalize(evidence)
        return bool(needle) and any(needle in hay for hay in self.corpus)

    def conflicts(self, day: str, start: Optional[str], duration: int) -> list[dict]:
        if start is None:
            return []
        lo, hi = _minutes(start), _minutes(start) + duration
        out = []
        for e in self.entries.values():
            if e['date'] == day and e['start_time']:
                elo = _minutes(e['start_time'])
                if lo < elo + e['duration_minutes'] and elo < hi:
                    out.append(e)
        return out


# --------------------------------------------------------------------------- tool bodies
def create_entries(sb: Sandbox, args: CreateEntriesArgs) -> dict:
    results, created = [], 0
    for item in args.entries:
        if not sb.grounded(item.date_evidence):
            results.append({'title': item.title, 'status': 'rejected', 'reason': 'date_evidence_not_found_in_user_text: '
                            'quote the exact words that give the date or ask the user'})
            continue
        dup = next((e for e in sb.entries.values() if normalize(e['title']) == normalize(item.title)
                    and e['date'] == item.date and e['start_time'] == item.start_time), None)
        if dup:
            results.append({'title': item.title, 'status': 'duplicate', 'id': dup['id']})
            continue
        clash = sb.conflicts(item.date, item.start_time, item.duration_minutes)
        if clash:
            results.append({'title': item.title, 'status': 'rejected', 'reason': 'time_conflict',
                            'conflicts_with': [f"{c['id']} {c['title']} {c['start_time']} ({c['duration_minutes']}m)" for c in clash]})
            continue
        entry = sb._add(item.title, item.date, item.start_time, item.duration_minutes)
        created += 1
        r = {'title': item.title, 'status': 'created', 'id': entry['id'], 'date': entry['date']}
        if date.fromisoformat(item.date) < sb.today:
            r['warning'] = 'date_is_in_the_past'
        results.append(r)
    all_rejected = all(r['status'] == 'rejected' for r in results)
    return {'ok': True, 'summary': f'{created} created, {len(results) - created} not created',
            'data': {'results': results, 'all_rejected': all_rejected, 'created': created}}


def add_reminders(sb: Sandbox, args: AddRemindersArgs) -> dict:
    results, created = [], 0
    for r in args.reminders:
        if r.entry_id not in sb.entries:
            results.append({'entry_id': r.entry_id, 'status': 'rejected', 'reason': 'unknown_entry_id'})
        elif any(x['entry_id'] == r.entry_id and x['days_before'] == r.days_before for x in sb.reminders):
            results.append({'entry_id': r.entry_id, 'status': 'duplicate'})
        else:
            sb.reminders.append({'entry_id': r.entry_id, 'days_before': r.days_before})
            created += 1
            results.append({'entry_id': r.entry_id, 'status': 'created'})
    return {'ok': True, 'summary': f'{created} reminders created',
            'data': {'results': results, 'all_rejected': created == 0 and all(x['status'] == 'rejected' for x in results), 'created': created}}


def list_entries(sb: Sandbox, args: ListEntriesArgs) -> dict:
    rows = [e for e in sb.entries.values()
            if (not args.from_date or e['date'] >= args.from_date) and (not args.to_date or e['date'] <= args.to_date)]
    rows.sort(key=lambda e: (e['date'], e['start_time'] or ''))
    return {'ok': True, 'summary': f'{len(rows)} entries', 'data': {'entries': rows[:25], 'truncated': len(rows) > 25}}


def check_conflicts(sb: Sandbox, args: CheckConflictsArgs) -> dict:
    clash = sb.conflicts(args.date, args.start_time, args.duration_minutes)
    return {'ok': True, 'summary': f'{len(clash)} conflicts', 'data': {'conflicts': clash}}


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ArgModel]
    fn: Optional[Callable[[Sandbox, Any], dict]]
    restricted: bool = False
    example: str = ''


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec('list_entries', 'Read sandbox calendar entries, optionally between two dates.', ListEntriesArgs, list_entries,
             example='{"from_date": "2026-11-01", "to_date": "2026-11-30"}'),
    ToolSpec('check_conflicts', 'Read-only check whether a timed slot overlaps existing entries.', CheckConflictsArgs, check_conflicts,
             example='{"date": "2026-11-05", "start_time": "10:00", "duration_minutes": 60}'),
    ToolSpec('create_entries', 'Create 1-5 sandbox calendar entries. Rejects entries whose date is not quoted from the user text, '
             'duplicates, and time conflicts. start_time null = all-day.', CreateEntriesArgs, create_entries,
             example='{"entries": [{"title": "Report due", "date": "2026-11-05", "start_time": null, "duration_minutes": 60, "date_evidence": "5 November"}]}'),
    ToolSpec('add_reminders', 'Attach 1-5 reminders (days_before 0-30) to existing entry ids.', AddRemindersArgs, add_reminders,
             example='{"reminders": [{"entry_id": "E3", "days_before": 2}]}'),
    # Consequential actions: registered so the model can REQUEST them, but the gateway never executes them.
    ToolSpec('delete_entry', 'RESTRICTED (needs human approval): delete a calendar entry.', AnyArgs, None, True),
    ToolSpec('send_email', 'RESTRICTED (needs human approval): send an email or message to a person.', AnyArgs, None, True),
    ToolSpec('sync_external_calendar', 'RESTRICTED (needs human approval): write to a real external calendar/account.', AnyArgs, None, True),
]}


def load_seed() -> list[dict]:
    try:
        return json.loads((ROOT / 'data' / 'sample_data.json').read_text(encoding='utf-8')).get('calendar_seed', [])
    except (OSError, ValueError):
        return []


def tool_catalog() -> str:
    lines = []
    for t in TOOLS.values():
        lines.append(f'- {t.name}: {t.description}' + (f'\n    arguments example: {t.example}' if t.example else ''))
    return '\n'.join(lines)


# --------------------------------------------------------------------------- gateway
class FaultController:
    """Injects the requested Arena fault ONCE, at the first matching operation."""

    def __init__(self, fault_type: str):
        self.type = fault_type
        self.consumed = False

    def take(self, *kinds: str) -> Optional[str]:
        if not self.consumed and self.type in kinds:
            self.consumed = True
            return self.type
        return None


@dataclass
class ToolOutcome:
    kind: str                 # 'ok' | 'approval_required' | 'failed'
    observation: str
    data: Optional[dict] = None
    failure: str = ''         # last infrastructure failure type when kind == 'failed'


class ToolGateway:
    def __init__(self, sandbox: Sandbox, faults: FaultController, emit: Callable[..., None]):
        self.sandbox = sandbox
        self.faults = faults
        self.emit = emit
        self.traces: list[ToolTrace] = []
        self.errors: list[dict] = []

    def _trace(self, step, tool, attempt, outcome, started):
        self.traces.append(ToolTrace(step=step, tool=tool, attempt=attempt, outcome=outcome,
                                     latency_ms=round((perf_counter() - started) * 1000, 2)))

    async def call(self, step: int, spec: ToolSpec, args: ArgModel) -> ToolOutcome:
        if spec.restricted:
            self._trace(step, spec.name, 1, 'rejected', perf_counter())
            self.emit(step, 'autonomy_gate', tool=spec.name, decision='approval_required')
            return ToolOutcome('approval_required',
                               f'APPROVAL_REQUIRED: {spec.name} was NOT executed. A human must approve this action.')
        attempts = 1 + settings.max_tool_retries
        last = ''
        for attempt in range(1, attempts + 1):
            started = perf_counter()
            fault = self.faults.take('tool_timeout', 'malformed_tool_output') if attempt == 1 else None
            try:
                if fault:
                    self.emit(step, 'fault_injected', fault=fault, tool=spec.name)
                if fault == 'tool_timeout':
                    raise asyncio.TimeoutError()
                if fault == 'malformed_tool_output':
                    raw: Any = '<<corrupted tool payload>>'
                else:
                    raw = await asyncio.wait_for(asyncio.to_thread(spec.fn, self.sandbox, args), settings.tool_timeout_seconds)
                out = ToolOutput.model_validate(raw)
            except (asyncio.TimeoutError, TimeoutError):
                last = 'timeout'
                self._trace(step, spec.name, attempt, 'timeout', started)
                self.errors.append({'type': 'tool_timeout', 'step': step, 'tool': spec.name, 'attempt': attempt})
            except ValidationError:
                last = 'malformed_output'
                self._trace(step, spec.name, attempt, 'malformed_output', started)
                self.errors.append({'type': 'malformed_tool_output', 'step': step, 'tool': spec.name, 'attempt': attempt})
            except Exception as exc:  # tool bug / dependency error: contained, never crashes the run
                last = 'exception'
                self._trace(step, spec.name, attempt, 'exception', started)
                self.errors.append({'type': 'tool_exception', 'step': step, 'tool': spec.name, 'attempt': attempt,
                                    'detail': type(exc).__name__})
            else:
                rejected = bool(out.data.get('all_rejected'))
                self._trace(step, spec.name, attempt, 'rejected' if rejected else 'success', started)
                self.emit(step, 'tool_result', tool=spec.name, attempt=attempt, ok=not rejected)
                text = json.dumps(out.data, ensure_ascii=False, default=str)
                return ToolOutcome('ok', f'{out.summary}. {text}'[:1200], out.data)
            self.emit(step, 'tool_retry' if attempt < attempts else 'tool_gave_up', tool=spec.name, attempt=attempt, failure=last)
        return ToolOutcome('failed', f'TOOL_FAILED: {spec.name} failed after {attempts} attempts ({last}).', None, last)
