"""Calendar Sentinel: bounded observe -> decide -> act -> observe loop.

The MODEL proposes one typed decision per step. The SYSTEM owns everything else:
  parse -> schema validate -> semantic validate -> (repair <= max_repairs | typed failure) -> gateway -> observation
Step accounting: every model call (including repairs) consumes one of max_steps.
"""
import asyncio
import json
import logging
import re
from datetime import date, datetime, timezone
from time import perf_counter
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app import llm
from app.config import settings
from app.models import AgentDecision, AgentState, ArenaResponse, Metrics, TERMINAL_STATUSES
from app.prompts import build_messages, neutralise, new_boundary
from app.tools import FaultController, Sandbox, TOOLS, ToolGateway, load_seed

log = logging.getLogger('agent')

_daily = {'day': None, 'calls': 0}


def _daily_cap_ok() -> bool:
    today = datetime.now(timezone.utc).date()
    if _daily['day'] != today:
        _daily.update(day=today, calls=0)
    if _daily['calls'] >= settings.daily_model_call_cap:
        return False
    _daily['calls'] += 1
    return True


def today_for_run() -> tuple[date, str]:
    if settings.fixed_today:
        return date.fromisoformat(settings.fixed_today), settings.app_timezone
    try:
        return datetime.now(ZoneInfo(settings.app_timezone)).date(), settings.app_timezone
    except Exception:
        return datetime.now(timezone.utc).date(), 'UTC'


# ----------------------------------------------------------------------------- parsing + validation
_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$', re.I)


def extract_json(raw: str) -> Optional[dict]:
    text = _FENCE.sub('', raw.strip())
    start = text.find('{')
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _errs(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors()][:4]


def validate_decision(raw: str):
    """Returns (decision, validated_tool_args, problems). Exactly one of decision/problems is meaningful."""
    obj = extract_json(raw)
    if obj is None:
        return None, None, ['output is not a single JSON object']
    try:
        decision = AgentDecision.model_validate(obj)           # schema layer
    except ValidationError as exc:
        return None, None, _errs(exc)
    if decision.status != 'continue':
        return decision, None, []
    spec = TOOLS.get((decision.action or '').strip())           # semantic layer
    if spec is None:
        return None, None, [f"action '{decision.action}' is not a registered tool"]
    if spec.restricted:
        return decision, spec.args_model(), []
    try:
        return decision, spec.args_model.model_validate(decision.arguments), []
    except ValidationError as exc:
        return None, None, [f'arguments for {spec.name}: {m}' for m in _errs(exc)]


def corrupt_decision(request_id: str) -> str:
    """Arena fault 'invalid_agent_decision': replace the model output with an invalid contract."""
    variant = sum(map(ord, request_id)) % 3
    return [
        '{"status": "proceed", "action": ["create_entries"], "arguments": "now"}',          # wrong enum + wrong types
        '{"status": "continue", "action": "do_everything", "arguments": {}}',               # unknown action
        'Sure! I will go ahead and handle that for you right away.',                        # not JSON at all
    ][variant]


def _draft(args) -> str:
    items = getattr(args, 'entries', None) or []
    return '\n'.join(f'- {e.title} ({e.date}' + (f' {e.start_time}' if e.start_time else '') + ')' for e in items)


# ----------------------------------------------------------------------------- the run
class AgentRun:
    def __init__(self, request, history, model_id: str):
        cfg = request.arena_config
        self.request = request
        self.history = list(history)
        self.today, self.tz = today_for_run()
        self.state = AgentState(request_id=request.request_id, goal=request.task, model_id=model_id,
                                max_steps=min(cfg.max_steps, settings.max_steps))
        corpus = [request.task] + [c.content for c in request.external_context] + \
                 [str(m.content) for m in self.history if m.type == 'human']
        self.sandbox = Sandbox(self.today, corpus, load_seed())
        self.events: list[dict] = []
        self.faults = FaultController(cfg.fault.type)
        self.gateway = ToolGateway(self.sandbox, self.faults, self.emit)
        self.boundary = new_boundary()
        self.started = perf_counter()
        self.deadline = self.started + max(settings.run_timeout_seconds - 4, 2)
        self.last_args = None
        self.errors_extra: list[dict] = []

    # -- observability -----------------------------------------------------------------
    def emit(self, step: int, event: str, **fields) -> None:
        rec = {'step': step, 'event': event, **fields}
        if len(self.events) < 80:
            self.events.append(rec)
        log.info('run=%s %s', self.request.request_id, json.dumps(rec, default=str)[:500])

    # -- result construction --------------------------------------------------------------
    def _progress(self) -> str:
        s = self.state
        if s.created_entries or s.created_reminders:
            return f'\n\n[Sandbox changes this run: {s.created_entries} entries and {s.created_reminders} reminders created]'
        return ''

    def finish(self, status: str, stop_reason: str, message: str, with_progress: bool = True) -> ArenaResponse:
        s = self.state
        s.status, s.stop_reason = status, stop_reason
        body = message.strip() or 'The run ended.'
        text = (body + (self._progress() if with_progress else ''))
        if len(text) > 2000:
            text = text[:1990].rstrip() + ' [...]'
        self.emit(s.step, 'agent_stop', status=status, reason=stop_reason)
        price_in = s.input_tokens
        return ArenaResponse(
            request_id=self.request.request_id, status=status, final_response=text, steps=s.step,
            stop_reason=stop_reason, tool_calls=self.gateway.traces,
            errors=(self.gateway.errors + self.errors_extra)[:30], events=self.events,
            metrics=Metrics(model_calls=s.model_calls, input_tokens=price_in, output_tokens=s.output_tokens,
                            estimated_cost_usd=llm.estimate_cost(s.model_id, price_in, s.output_tokens)))

    def _tokens_used(self) -> int:
        return (self.state.input_tokens or 0) + (self.state.output_tokens or 0)

    # -- model call (transient failures get ONE retry; it does not consume a decision step) -------
    async def _call_model(self, client, messages) -> str:
        for attempt in (1, 2):
            budget = self.deadline - perf_counter()
            if budget < 1.5:
                raise llm.ModelCallError('timeout', 'run time budget')
            try:
                reply = await client.complete(messages[0].content, messages[1:], settings.max_output_tokens,
                                              min(settings.model_timeout_seconds, budget))
            except llm.ModelCallError as exc:
                self.emit(self.state.step, 'model_error', kind=exc.kind, detail=exc.detail, attempt=attempt)
                if exc.kind == 'fatal' or attempt == 2:
                    raise
                await asyncio.sleep(min(2.0, max(0.0, self.deadline - perf_counter() - 2)))
                continue
            s = self.state
            if reply.input_tokens is not None:
                s.input_tokens = (s.input_tokens or 0) + reply.input_tokens
            if reply.output_tokens is not None:
                s.output_tokens = (s.output_tokens or 0) + reply.output_tokens
            return reply.text
        raise llm.ModelCallError('transient', 'unreachable')

    # -- main loop --------------------------------------------------------------------------------
    async def run(self) -> ArenaResponse:
        s, req = self.state, self.request
        self.emit(0, 'agent_start', model=s.model_id, max_steps=s.max_steps, fault=req.arena_config.fault.type,
                  history_messages=len(self.history), external_items=len(req.external_context))

        size = len(req.task) + sum(len(c.content) for c in req.external_context) + sum(len(str(m.content)) for m in self.history)
        if size > settings.max_input_chars:
            return self.finish('budget_exceeded', 'input_budget_exceeded',
                               f'The request plus attached context is {size} characters, above the {settings.max_input_chars} limit. '
                               'Nothing was processed; please split it into smaller requests.', False)
        try:
            client = llm.get_client(s.model_id)
        except llm.ModelUnavailable:
            return self.finish('failed', 'model_not_configured', 'No usable model is configured on this server.', False)

        while True:
            if s.step >= s.max_steps:
                return self.finish('budget_exceeded', 'step_budget_exhausted',
                                   f'Stopped: the {s.max_steps}-step budget was used up before the task finished.')
            if self._tokens_used() >= settings.max_total_tokens:
                return self.finish('budget_exceeded', 'token_budget_exhausted',
                                   'Stopped: the token budget for this run was reached before the task finished.')
            if self.deadline - perf_counter() < 2.0:
                return self.finish('budget_exceeded', 'time_budget_reached',
                                   'Stopped: the time budget for this run was reached before the task finished.')
            if not _daily_cap_ok():
                return self.finish('budget_exceeded', 'service_call_cap_reached',
                                   'The service-wide daily model-call cap has been reached. Try again later.', False)

            messages = build_messages(
                history=self.history, user_goal=req.task, external=req.external_context, boundary=self.boundary,
                today=self.today, tz=self.tz, step=s.step, max_steps=s.max_steps, repairs=s.repairs_in_row,
                max_repairs=settings.max_repairs, n_entries=s.created_entries, n_reminders=s.created_reminders,
                pending=s.pending_approvals, log=s.log)
            s.step += 1
            s.model_calls += 1
            try:
                raw = await self._call_model(client, messages)
            except llm.ModelCallError as exc:
                self.errors_extra.append({'type': 'model_error', 'step': s.step, 'kind': exc.kind})
                reason = 'time_budget_reached' if exc.kind == 'timeout' else 'model_unavailable'
                status = 'budget_exceeded' if exc.kind == 'timeout' else 'failed'
                hint = ''
                if exc.kind == 'fatal':
                    hint = (f' The provider rejected the request ({exc.detail}): HTTP 404 usually means MODEL_NAME is not an '
                            'available model; 401/403 means the API key is wrong.')
                return self.finish(status, reason, 'The language model did not respond in time or is unavailable.'
                                   + hint + ' No further actions were taken.')

            if self.faults.take('invalid_agent_decision'):
                self.emit(s.step, 'fault_injected', fault='invalid_agent_decision')
                raw = corrupt_decision(req.request_id)

            decision, args, problems = validate_decision(raw)
            if problems:
                s.repairs_in_row += 1
                self.emit(s.step, 'decision_invalid', problems=problems, repair=s.repairs_in_row)
                self.errors_extra.append({'type': 'invalid_decision', 'step': s.step, 'detail': problems[:2],
                                          'recovered': s.repairs_in_row <= settings.max_repairs})
                if s.repairs_in_row > settings.max_repairs:
                    return self.finish('contract_error', 'invalid_decision_unrecoverable',
                                       'The model could not produce a valid decision within the repair limit, so the run was '
                                       'stopped without taking any further action.')
                s.log.append(f'[contract_violation @step {s.step}] your previous output was rejected: '
                             + neutralise('; '.join(problems)) + '. Return a corrected decision as one JSON object.')
                continue

            s.repairs_in_row = 0
            self.emit(s.step, 'decision', status=decision.status, action=decision.action,
                      rationale=decision.rationale[:200], validation='passed')

            if decision.status != 'continue':
                return self._terminal(decision)

            spec = TOOLS[decision.action.strip()]
            s.log.append(f'[decision {s.step}] action={spec.name} arguments='
                         + neutralise(json.dumps(decision.arguments, ensure_ascii=False, default=str)[:500]))
            outcome = await self.gateway.call(s.step, spec, args)
            if outcome.kind == 'approval_required':
                s.pending_approvals.append(spec.name)
            elif outcome.kind == 'failed':
                msg = (f'The sandbox tool "{spec.name}" failed after {1 + settings.max_tool_retries} attempts '
                       f'({outcome.failure}). Nothing further was done.')
                draft = _draft(args)
                if draft:
                    msg += f'\nDraft (NOT saved):\n{draft}'
                return self.finish('tool_error', 'tool_failed_after_retries', msg)
            elif outcome.data:
                if spec.name == 'create_entries':
                    s.created_entries += int(outcome.data.get('created', 0))
                elif spec.name == 'add_reminders':
                    s.created_reminders += int(outcome.data.get('created', 0))
            s.log.append(f'[observation {s.step}] ' + neutralise(outcome.observation[:900]))

    def _terminal(self, d: AgentDecision) -> ArenaResponse:
        s = self.state
        msg = d.user_message or ''
        approval = bool(s.pending_approvals)
        if d.status == 'completed':
            return self.finish('approval_required', 'approval_required', msg) if approval else \
                self.finish('completed', 'goal_completed', msg)
        if d.status == 'needs_clarification':
            return self.finish('needs_clarification', 'clarification_needed', msg)
        if d.status == 'approval_required' or (d.status == 'blocked' and approval):
            return self.finish('approval_required', 'approval_required', msg)
        if d.status == 'blocked':
            return self.finish('blocked', 'goal_blocked', msg)
        if d.status == 'budget_exceeded':
            return self.finish('budget_exceeded', 'step_budget_exhausted', msg)
        return self.finish('failed', 'agent_reported_failure', msg)


async def run_agent(request, history, model):
    return await AgentRun(request, history, model).run()
