"""Checks the PUBLIC deployment the way the evaluator will (no login, no local files).
Usage: python scripts/verify_deployment.py https://YOUR-APP.onrender.com"""
import sys
import time

import httpx

if len(sys.argv) != 2 or not sys.argv[1].startswith('https://'):
    raise SystemExit('Usage: python scripts/verify_deployment.py https://YOUR-APP.onrender.com')
base = sys.argv[1].rstrip('/')
bad = 0


def check(name, ok, detail=''):
    global bad
    bad += 0 if ok else 1
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))


print('(free Render may need up to ~60 s to wake up)')
with httpx.Client(timeout=90, follow_redirects=True) as c:
    t0 = time.perf_counter()
    r = c.get(base + '/health')
    check('GET /health 200 + status ok', r.status_code == 200 and r.json().get('status') == 'ok', f'{time.perf_counter() - t0:.1f}s')
    check('health says model_configured', r.json().get('model_configured') is True,
          'set GEMINI_API_KEY / MODEL_* in Render env' if not r.json().get('model_configured') else '')
    check('implementation is not "starter"', r.json().get('implementation') != 'starter')
    r = c.get(base + '/arena/manifest')
    m = r.json()
    check('GET /arena/manifest 200', r.status_code == 200 and m.get('arena_version') == '0.1')
    check('manifest not the starter template', m.get('implementation') != 'starter' and m.get('agent_name') != 'Student Starter')
    r = c.get(base + '/')
    check('GET / serves the chat UI', r.status_code == 200 and '<select id="model"' in r.text)
    check('GET /docs reachable', c.get(base + '/docs').status_code == 200)
    r = c.get(base + '/models')
    check('GET /models lists a real model', r.status_code == 200 and r.json()['models'] != ['unconfigured'], str(r.json().get('models')))
    body = {'task': "Add 'Deployment check' on 2026-12-01 to my calendar.", 'external_context': [],
            'arena_config': {'max_steps': 6, 'fault': 'none'}}
    t0 = time.perf_counter()
    r = c.post(base + '/arena/run', json=body)
    res = r.json() if r.status_code == 200 else {}
    check('POST /arena/run valid request -> 200 + contract fields',
          r.status_code == 200 and all(k in res for k in ('status', 'final_response', 'steps', 'stop_reason', 'tool_calls', 'errors')),
          f'{time.perf_counter() - t0:.1f}s')
    print('      ->', res.get('status'), res.get('stop_reason'), 'steps=', res.get('steps'))
    check('POST /arena/run completed the task', res.get('status') == 'completed')
    check('POST /arena/run blank task -> 422', c.post(base + '/arena/run', json={'task': '  '}).status_code == 422)
print('\nALL CHECKS PASSED' if not bad else f'\n{bad} CHECK(S) FAILED - fix before submitting')
raise SystemExit(1 if bad else 0)
