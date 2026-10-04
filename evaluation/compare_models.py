"""Model Selection Experiment (assignment section 6).
Runs the same 10 development inputs (evaluation/dev_cases.json) against two or more models through the
server's /chat route (which lets the request choose the model) and writes:
  evaluation/results/model_comparison.json   raw per-case data
  evaluation/results/model_comparison.md     the table
and refreshes the table between the MODEL_TABLE markers in README.md.

Usage (server running with API keys for every model you compare):
  python evaluation/compare_models.py --url http://127.0.0.1:8000 --models gemini:gemini-3.1-flash-lite gemini:gemini-3.5-flash
If --models is omitted, every id returned by GET /models is compared.
Nothing here is fabricated: every number comes from real responses. Cost is 'n/a' unless MODEL_PRICES_JSON is set."""
import argparse
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parent
p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8000')
p.add_argument('--models', nargs='*')
p.add_argument('--delay', type=float, default=7.0, help='seconds between calls (free-tier RPM limits)')
args = p.parse_args()
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

models = args.models or httpx.get(base + '/models', timeout=20).json()['models']
allowed = httpx.get(base + '/models', timeout=30).json()['models']
missing = [m for m in models if m not in allowed]
if missing:
    raise SystemExit(f'These models are NOT enabled on the server: {missing}\nEnabled models: {allowed}\nFix ALLOWED_MODELS (and restart / redeploy) or change --models.')
if len(models) < 2:
    raise SystemExit(f'Need at least two models, got {models}. Set ALLOWED_MODELS and the provider keys.')
cases = json.loads((HERE / 'dev_cases.json').read_text(encoding='utf-8'))
day2 = (datetime.now(timezone.utc).date() + timedelta(days=2)).isoformat()


def as_list(x):
    return x if isinstance(x, list) else [x]


report = {}
for model in models:
    rows = []
    for c in cases:
        body = {'session_id': 'cmp-' + uuid.uuid4().hex, 'model': model, 'task': c['task'].replace('{SEED_DAY_2}', day2),
                'external_context': c.get('external_context', []), 'arena_config': {'max_steps': 6, 'fault': 'none'}}
        row = {'case': c['name']}
        try:
            t0 = time.perf_counter()
            r = httpx.post(base + '/chat', json=body, timeout=70)
            r.raise_for_status()
            res = r.json()
            wall = (time.perf_counter() - t0) * 1000
            calls = max(res['metrics']['model_calls'], 1)
            invalid = sum(1 for e in res['events'] if e['event'] == 'decision_invalid')
            tools = [t['tool'] for t in res['tool_calls']]
            want_tool = c['expected_first_tool']
            m = res['metrics']
            row.update(status=res['status'], stop_reason=res['stop_reason'], steps=res['steps'],
                       success=res['status'] in as_list(c['expected_status']),
                       valid_decisions=calls - invalid, total_decisions=calls,
                       action_ok=(tools[:1] == [want_tool]) if want_tool else (not tools),
                       latency_ms=m['latency_ms'] or wall, input_tokens=m['input_tokens'], output_tokens=m['output_tokens'],
                       cost_usd=m['estimated_cost_usd'])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            row.update(error=describe(exc), success=False, valid_decisions=0, total_decisions=1, action_ok=False)
        try:
            httpx.delete(base + '/chat/' + body['session_id'], timeout=20)
        except httpx.HTTPError:
            pass
        rows.append(row)
        print(f"{model:40s} {c['name']:42s} success={row['success']}" + (f"  ERROR: {row['error']}" if row.get('error') else ''))
        time.sleep(args.delay)
    report[model] = rows


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, spec='{:.0f}', none='n/a'):
    return none if v is None else spec.format(v)


if all(r.get('error') for rows in report.values() for r in rows):
    raise SystemExit('Every request failed - NOT writing a table. Fix the error shown above and run again.')

header = ('| Model | Task success | Structured-output validity | Correct action selection | Avg latency / case (ms) | '
          'Avg input tok | Avg output tok | Avg cost / case (USD) |\n|---|---|---|---|---|---|---|---|')
lines = [header]
n = len(cases)
for model, rows in report.items():
    ok = sum(r['success'] for r in rows)
    valid = sum(r['valid_decisions'] for r in rows)
    total = sum(r['total_decisions'] for r in rows)
    act = sum(r['action_ok'] for r in rows)
    cost = avg([r.get('cost_usd') for r in rows])
    lines.append(f"| `{model}` | {ok}/{n} | {valid}/{total} ({100 * valid / max(total, 1):.0f}%) | {act}/{n} | "
                 f"{fmt(avg([r.get('latency_ms') for r in rows]))} | {fmt(avg([r.get('input_tokens') for r in rows]))} | "
                 f"{fmt(avg([r.get('output_tokens') for r in rows]))} | {fmt(cost, '{:.6f}')} |")
table = '\n'.join(lines)
stamp = f"_Measured {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} on {n} identical inputs (`evaluation/dev_cases.json`); raw data in `evaluation/results/model_comparison.json`._"

out = HERE / 'results'
out.mkdir(exist_ok=True)
(out / 'model_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(out / 'model_comparison.md').write_text(table + '\n\n' + stamp + '\n', encoding='utf-8')

readme = ROOT / 'README.md'
text = readme.read_text(encoding='utf-8')
new = re.sub(r'(<!-- MODEL_TABLE_START -->).*?(<!-- MODEL_TABLE_END -->)', lambda m: f'{m.group(1)}\n{table}\n\n{stamp}\n{m.group(2)}', text, flags=re.S)
readme.write_text(new, encoding='utf-8')
print('\n' + table + '\n\nREADME.md table refreshed. Now write your 3-4 line selection rationale under the table.')
