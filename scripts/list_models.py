"""Lists the Gemini models YOUR key can call with generateContent (reads GEMINI_API_KEY from .env).
Run:  python scripts/list_models.py
Pick two names from the output for MODEL_NAME / ALLOWED_MODELS (format provider:name, e.g. gemini:gemini-3-flash-preview)."""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

if not settings.gemini_api_key:
    raise SystemExit('GEMINI_API_KEY is empty. Put it in .env first.')
names, token = [], None
with httpx.Client(timeout=30) as c:
    while True:
        r = c.get(settings.gemini_base_url.rstrip('/') + '/v1beta/models',
                  params={'pageSize': 200, **({'pageToken': token} if token else {})},
                  headers={'x-goog-api-key': settings.gemini_api_key})
        if r.status_code != 200:
            raise SystemExit(f'HTTP {r.status_code}: the key was rejected or the API is not enabled for it.')
        data = r.json()
        names += [m['name'].removeprefix('models/') for m in data.get('models', [])
                  if 'generateContent' in m.get('supportedGenerationMethods', [])]
        token = data.get('nextPageToken')
        if not token:
            break
text = [n for n in names if n.startswith('gemini') and not any(x in n for x in ('image', 'tts', 'embedding', 'live', 'audio', 'robotics', 'computer'))]
print('Text models usable with your key (best candidates first: flash / flash-lite):\n')
for n in sorted(text, key=lambda n: ('flash' not in n, n)):
    print('  ', n)
print('\nCurrently configured MODEL_NAME:', settings.model_name or '(empty)')
print('In .env set e.g.:\n  MODEL_NAME=<one name above>\n  ALLOWED_MODELS=gemini:<name one>,gemini:<name two>')
