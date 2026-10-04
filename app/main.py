import asyncio
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import ROOT, settings
from app.memory import Memory

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')


@asynccontextmanager
async def lifespan(app):
    app.state.memory = Memory()                       # in-memory only: lost on restart (single worker)
    app.state.busy = set()                            # one running request per chat session
    app.state.slots = asyncio.Semaphore(settings.max_concurrent_runs)   # admission control for model calls
    app.state.hits = defaultdict(deque)               # per-IP sliding window for the browser /chat route
    yield


app = FastAPI(title='Calendar Sentinel - Agent Arena', version='1.0.0', lifespan=lifespan)
app.include_router(router)
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')
