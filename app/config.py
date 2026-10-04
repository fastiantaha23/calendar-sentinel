"""Environment-driven configuration. Secrets come only from environment variables."""
import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')

    host: str = '127.0.0.1'
    port: int = 8000

    # --- model selection -------------------------------------------------
    model_provider: str = 'unconfigured'          # 'gemini' | 'anthropic'
    model_name: str = ''                          # e.g. gemini-2.5-flash-lite
    allowed_models: str = ''                      # comma list: "gemini:gemini-2.5-flash,gemini:gemini-2.5-flash-lite"
    gemini_api_key: str = ''
    anthropic_api_key: str = ''
    gemini_base_url: str = 'https://generativelanguage.googleapis.com'
    anthropic_base_url: str = 'https://api.anthropic.com'
    gemini_thinking_budget: str = '0'             # '' = omit; '0' disables thinking on flash models
    # Optional price table (USD per 1M tokens): {"gemini-2.5-flash-lite": [0.10, 0.40]}
    model_prices_json: str = '{}'

    # --- execution limits (the SYSTEM owns these, never the model) --------
    max_steps: int = Field(default=6, ge=1, le=6)             # model decisions incl. repairs
    max_tool_retries: int = Field(default=2, ge=0, le=2)      # infra retries per tool call
    max_repairs: int = Field(default=2, ge=0, le=2)           # consecutive contract repairs
    max_output_tokens: int = Field(default=512, ge=1)         # per model call
    max_total_tokens: int = Field(default=30000, ge=1)        # per run (input+output)
    max_input_chars: int = Field(default=24000, ge=1000)      # task+context+history per run
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    model_timeout_seconds: float = Field(default=14, gt=0)
    tool_timeout_seconds: float = Field(default=3, gt=0)

    # --- public-service protection ----------------------------------------
    chat_rate_per_minute: int = Field(default=20, ge=1)       # UI /chat per IP
    max_concurrent_runs: int = Field(default=4, ge=1)
    daily_model_call_cap: int = Field(default=5000, ge=1)

    app_timezone: str = 'UTC'
    fixed_today: str = ''                                     # YYYY-MM-DD, for reproducible tests only

    # ---------------------------------------------------------------------
    @property
    def default_model_id(self) -> str:
        if self.model_provider == 'unconfigured' or not self.model_name:
            return 'unconfigured'
        return f'{self.model_provider}:{self.model_name}'

    def parsed_allowed_models(self) -> list[str]:
        ids: list[str] = []
        for raw in self.allowed_models.split(','):
            raw = raw.strip()
            if not raw:
                continue
            ids.append(raw if ':' in raw else f'{self.model_provider}:{raw}')
        if self.default_model_id != 'unconfigured' and self.default_model_id not in ids:
            ids.insert(0, self.default_model_id)
        return ids

    def api_key_for(self, provider: str) -> str:
        return {'gemini': self.gemini_api_key, 'anthropic': self.anthropic_api_key}.get(provider, '')

    def prices(self) -> dict:
        try:
            data = json.loads(self.model_prices_json or '{}')
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}


settings = Settings()
