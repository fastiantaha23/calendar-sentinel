"""Runs the public cases (and one multi-turn clarification test) against a RUNNING server over HTTP.
This calls the configured model and may cost money / use free quota. Results are saved to
evaluation/results/public_results.json (this is the 'Public test results' path for the submission)."""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--delay', type=float, default=6.0, help='seconds between cases (free-tier rate limits)')
parser.add_argument('--out', default=str(HERE / 'results' / 'public_results.json'))
args = parser.parse_args()
base = args.url.rstrip('/')


def describe(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return f'HTTP {exc.response.status_code}: {exc.response.text[:150]}'
    return f'{type(exc).__name__}: {str(exc)[:120]}'


try:
    httpx.get(base + '/health', timeout=60).raise_for_status()
except httpx.HTTPError as exc:
    raise SystemExit(f'CANNOT REACH {base}/health  ({describe(exc)})\n'
                     'Start the server first (python run.py in ANOTHER terminal) or pass --url https://YOUR-APP.onrender.com')


def as_list(x):
    return x if isinstance(x, list) else [x]


results, passed = [], 0
cases = json.loads((HERE / 'public_cases.json').read_text(encoding='utf-8'))
for case in cases:
    row = {'name': case['name'], 'expected_status': case['expected_status'], 'expected_stop_reason': case['expected_stop_reason']}
    try:
        t0 = time.perf_counter()
        r = httpx.post(base + '/arena/run', json=case['request'], timeout=70)
        r.raise_for_status()
        res = r.json()
        ok = res['status'] in as_list(case['expected_status']) and res['stop_reason'] in as_list(case['expected_stop_reason'])
        row.update(status=res['status'], stop_reason=res['stop_reason'], steps=res['steps'],
                   tools=[t['tool'] + ':' + t['outcome'] for t in res['tool_calls']],
                   latency_s=round(time.perf_counter() - t0, 2), metrics=res.get('metrics'), passed=ok,
                   final_response=res['final_response'][:300])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        ok = False
        row.update(passed=False, error=describe(exc))
    passed += ok
    print(('PASS ' if ok else 'FAIL ') + case['name'] + (f"  -> {row.get('status')}/{row.get('stop_reason')}" if not ok else '') + (f"  ERROR: {row['error']}" if row.get('error') else ''))
    results.append(row)
    time.sleep(args.delay)

# --- multi-turn clarification test: same session, second message is only the missing detail --------
session = 'public-mt-' + uuid.uuid4().hex
models = httpx.get(base + '/models', timeout=20).json()
model = models.get('default') or models['models'][0]
mt = {'name': 'MT1 multi-turn clarification (2 messages, same session)'}
try:
    first = httpx.post(base + '/chat', json={'session_id': session, 'model': model,
                       'task': 'Put my project report deadline on my calendar.'}, timeout=70).json()
    time.sleep(args.delay)
    second = httpx.post(base + '/chat', json={'session_id': session, 'model': model,
                        'task': 'It is due on 2026-12-05.'}, timeout=70).json()
    ok = first['status'] == 'needs_clarification' and second['status'] == 'completed' \
        and any(t['tool'] == 'create_entries' and t['outcome'] == 'success' for t in second['tool_calls'])
    mt.update(turn1=[first['status'], first['final_response'][:200]], turn2=[second['status'], second['final_response'][:200]], passed=ok)
except (httpx.HTTPError, KeyError, ValueError) as exc:
    ok = False
    mt.update(passed=False, error=describe(exc))
finally:
    try:
        httpx.delete(base + '/chat/' + session, timeout=20)
    except httpx.HTTPError:
        pass
passed += ok
print(('PASS ' if ok else 'FAIL ') + mt['name'])
results.append(mt)

total = len(cases) + 1
summary = {'url': base, 'ran_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'model': model, 'passed': passed, 'total': total, 'cases': results}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'{passed}/{total} checks passed. Saved to {args.out}')
raise SystemExit(0 if passed == total else 1)
