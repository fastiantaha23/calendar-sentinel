"""Model access. Provider calls use httpx directly so the output-token cap, JSON mode and the
'thinking' budget are explicit. LangChain message objects are the interchange type for history."""
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx
from langchain_core.messages import BaseMessage

from app.config import settings


class ModelUnavailable(Exception):
    """Requested model is not configured (unknown provider, missing key...)."""


class ModelCallError(Exception):
    def __init__(self, kind: str, detail: str = ''):
        super().__init__(f'{kind}: {detail}')
        self.kind = kind            # 'transient' | 'timeout' | 'fatal'
        self.detail = detail


@dataclass
class ModelReply:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ModelClient(Protocol):
    async def complete(self, system: str, messages: list[BaseMessage], max_output_tokens: int,
                       timeout: float) -> ModelReply: ...


def _role(msg: BaseMessage) -> str:
    return 'assistant' if msg.type == 'ai' else 'user'


def _text(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        return ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _alternating(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Merge consecutive same-role messages and make sure the conversation starts with 'user'."""
    out: list[tuple[str, str]] = []
    for m in messages:
        role, text = _role(m), _text(m)
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + '\n\n' + text)
        else:
            out.append((role, text))
    while out and out[0][0] != 'user':
        out.pop(0)
    return out


def _provider_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get('error', {})
        return str(err.get('message') or err)[:200]
    except ValueError:
        return ''


def _classify(resp: httpx.Response) -> None:
    if resp.status_code in (429, 500, 502, 503, 504):
        raise ModelCallError('transient', f'HTTP {resp.status_code}')
    if resp.status_code >= 400:
        raise ModelCallError('fatal', f'HTTP {resp.status_code}: {_provider_message(resp)}'.strip(': '))


class HttpModelClient:
    def __init__(self, provider: str, name: str):
        self.provider, self.name = provider, name
        self.key = settings.api_key_for(provider)
        if provider not in ('gemini', 'anthropic') or not name or not self.key:
            raise ModelUnavailable(f'{provider}:{name} is not configured')

    async def complete(self, system, messages, max_output_tokens, timeout) -> ModelReply:
        turns = _alternating(messages)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                if self.provider == 'gemini':
                    return await self._gemini(http, system, turns, max_output_tokens)
                return await self._anthropic(http, system, turns, max_output_tokens)
        except httpx.TimeoutException as exc:
            raise ModelCallError('timeout', type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise ModelCallError('transient', type(exc).__name__) from None

    async def _gemini(self, http, system, turns, max_tokens) -> ModelReply:
        gen = {'maxOutputTokens': max_tokens, 'temperature': 0, 'responseMimeType': 'application/json'}
        if settings.gemini_thinking_budget.strip() != '':
            gen['thinkingConfig'] = {'thinkingBudget': int(settings.gemini_thinking_budget)}
        body = {'systemInstruction': {'parts': [{'text': system}]},
                'contents': [{'role': 'user' if r == 'user' else 'model', 'parts': [{'text': t}]} for r, t in turns],
                'generationConfig': gen}
        url = f'{settings.gemini_base_url.rstrip("/")}/v1beta/models/{self.name}:generateContent'
        resp = await http.post(url, json=body, headers={'x-goog-api-key': self.key})
        if resp.status_code == 400 and 'thinkingConfig' in gen:      # newer models may not accept thinkingBudget
            gen.pop('thinkingConfig')
            resp = await http.post(url, json=body, headers={'x-goog-api-key': self.key})
        _classify(resp)
        data = resp.json()
        cands = data.get('candidates') or []
        parts = (cands[0].get('content') or {}).get('parts', []) if cands else []
        text = ''.join(p.get('text', '') for p in parts)
        u = data.get('usageMetadata') or {}
        out = (u.get('candidatesTokenCount') or 0) + (u.get('thoughtsTokenCount') or 0)
        return ModelReply(text, u.get('promptTokenCount'), out if u else None)

    async def _anthropic(self, http, system, turns, max_tokens) -> ModelReply:
        body = {'model': self.name, 'max_tokens': max_tokens, 'temperature': 0, 'system': system,
                'messages': [{'role': r, 'content': t} for r, t in turns]}
        resp = await http.post(f'{settings.anthropic_base_url.rstrip("/")}/v1/messages', json=body,
                               headers={'x-api-key': self.key, 'anthropic-version': '2023-06-01'})
        _classify(resp)
        data = resp.json()
        text = ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text')
        u = data.get('usage') or {}
        return ModelReply(text, u.get('input_tokens'), u.get('output_tokens'))


# --- registry / test hook ----------------------------------------------------
_override: Optional[ModelClient] = None


def set_override(client: Optional[ModelClient]) -> None:
    """Tests inject a scripted model here (no network, no cost)."""
    global _override
    _override = client


def get_client(model_id: str) -> ModelClient:
    if _override is not None:
        return _override
    provider, _, name = model_id.partition(':')
    return HttpModelClient(provider, name)


def configured_models() -> list[str]:
    """Allowed model ids whose provider key is present. Keys are never exposed."""
    ids = [m for m in settings.parsed_allowed_models() if settings.api_key_for(m.partition(':')[0])]
    return ids or ['unconfigured']


def estimate_cost(model_id: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    price = settings.prices().get(model_id.partition(':')[2])
    if tokens_in is None or tokens_out is None or not isinstance(price, list) or len(price) != 2:
        return None
    return round((tokens_in * price[0] + tokens_out * price[1]) / 1_000_000, 6)
