import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.arena import execute
from app.config import ROOT, settings
from app.llm import configured_models
from app.models import ArenaRequest, ArenaResponse, ChatRequest

router = APIRouter()


def default_model() -> str:
    available = configured_models()
    return settings.default_model_id if settings.default_model_id in available else available[0]


@router.get('/')
def index():
    return FileResponse(ROOT / 'app/static/index.html')


@router.get('/health')
def health():
    return {'status': 'ok', 'implementation': 'calendar-sentinel',
            'model_configured': default_model() != 'unconfigured'}


@router.get('/arena/manifest')
def manifest():
    return json.loads((ROOT / 'arena_manifest.json').read_text(encoding='utf-8'))


@router.get('/models')
def models():
    """Only configured, allowed model ids. Keys are never exposed."""
    return {'models': configured_models(), 'default': default_model()}


@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest, request: Request):
    async with request.app.state.slots:
        return await execute(payload, model=default_model())


def _rate_limited(request: Request) -> bool:
    fwd = request.headers.get('x-forwarded-for', '')
    ip = fwd.split(',')[0].strip() or (request.client.host if request.client else 'unknown')
    window = request.app.state.hits[ip]
    now = time.monotonic()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.chat_rate_per_minute:
        return True
    window.append(now)
    return False


@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    if payload.model not in models()['models']:
        raise HTTPException(400, 'Model is not enabled')
    if _rate_limited(request):
        raise HTTPException(429, 'Too many requests; please slow down')
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        async with request.app.state.slots:
            result = await execute(payload, memory.get(payload.session_id), payload.model)
        memory.add(payload.session_id, payload.task, result.final_response)
        return result
    finally:
        busy.discard(payload.session_id)


@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
