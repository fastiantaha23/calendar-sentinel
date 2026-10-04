"""Calls Gemini DIRECTLY with your key, shows Google's real error text, finds a model that works.
  python scripts/test_model.py            -> test candidates, print results
  python scripts/test_model.py --apply    -> also write the working models into .env"""
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings  # noqa: E402

if not settings.gemini_api_key:
    raise SystemExit('GEMINI_API_KEY is empty in .env')
CANDIDATES = ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite', 'gemini-3.5-flash', 'gemini-3.8-flash',
              'gemini-flash-lite-latest', 'gemini-flash-latest', 'gemini-3-flash-preview']
base = settings.gemini_base_url.rstrip('/')
working = []


def call(model, thinking):
    gen = {'maxOutputTokens': 256, 'temperature': 0, 'responseMimeType': 'application/json'}
    if thinking:
        gen['thinkingConfig'] = {'thinkingBudget': 0}
    body = {'contents': [{'role': 'user', 'parts': [{'text': 'Return exactly this JSON: {"ok": true}'}]}], 'generationConfig': gen}
    r = httpx.post(f'{base}/v1beta/models/{model}:generateContent', json=body,
                   headers={'x-goog-api-key': settings.gemini_api_key}, timeout=40)
    msg = ''
    try:
        msg = r.json().get('error', {}).get('message', '')[:300]
    except ValueError:
        msg = r.text[:300]
    return r.status_code, msg


for m in CANDIDATES:
    try:
        code, msg = call(m, True)
        note = ''
        if code == 400:
            code, msg = call(m, False)
            note = ' (works only WITHOUT thinkingConfig)'
    except httpx.HTTPError as exc:
        print(f'{m:28s} NETWORK ERROR {type(exc).__name__}')
        continue
    print(f'{m:28s} HTTP {code}{note}  {"" if code == 200 else msg}')
    if code == 200:
        working.append(m)
    if len(working) == 2:
        break

if not working:
    print('\nNo candidate worked. Read the HTTP codes/messages above: 429 = quota (wait a minute / new key), 403 = key/API problem.')
    raise SystemExit(1)
print('\nWORKING:', ', '.join(working))
line_a = f'MODEL_NAME={working[0]}'
line_b = 'ALLOWED_MODELS=' + ','.join(f'gemini:{w}' for w in working)
print('Use in .env / Render:\n ', line_a, '\n ', line_b)
if '--apply' in sys.argv:
    env = ROOT / '.env'
    text = env.read_text(encoding='utf-8')
    text = re.sub(r'(?m)^MODEL_NAME=.*$', line_a, text)
    text = re.sub(r'(?m)^ALLOWED_MODELS=.*$', line_b, text)
    env.write_text(text, encoding='utf-8')
    print('\n.env updated. Now STOP the server (Ctrl+C) and start it again.')
