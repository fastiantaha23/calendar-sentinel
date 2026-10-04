# Calendar Sentinel — Agent Arena (Assignment 1)

A bounded, single-agent **Deadline / Calendar Sentinel** (domain 1). It reads a user's request (and optional
pasted documents such as a course outline), decides what to do next, and either writes entries/reminders to an
**isolated sandbox calendar**, asks for missing details, or stops safely. Nothing real is ever modified.

> Student: roll number **i230824** · GitHub: `fastiantaha23` · Course: Agentic AI, Fall 2026

---------------------------------------------------------------------------------------------------

## 1. Problem statement

Students receive deadlines in scattered places (course outlines, chat messages, emails). Copying them into a
calendar is error-prone, and a naive LLM assistant will happily invent dates or obey instructions hidden in the
pasted text. Calendar Sentinel turns deadline text into sandbox calendar entries **only when the facts are actually
present**, and behaves predictably when information is missing, tools fail, the model misbehaves, the input is
hostile, or limits are reached.

## 2. Agent Design Canvas

| # | Element | Definition |
|---|---|---|
| 1 | **Operational goal** | Convert deadlines/events stated by the user (or in a quoted document) into sandbox calendar entries and reminders; list existing entries on request. |
| 2 | **Completion condition** | Every requested item whose title **and** date are grounded in the user's text has a `created`/`duplicate` result in a `create_entries` observation (plus reminders if requested), and the model returns `completed`. Alternative terminal outcomes: clarification question, blocked, approval required, budget exceeded, tool/contract error. |
| 3 | **System boundary** | Inside: the loop in `app/agent.py`, the tool gateway, the per-run in-memory sandbox calendar. Outside (never touched): real calendars, e-mail, accounts, files, network resources other than the model API. |
| 4 | **Observations** | `task` text, bounded chat history (LangChain messages), `external_context` items (untrusted), runtime state (date, step, budget, pending approvals), tool observations. |
| 5 | **Actions / tools** | Allowed: `list_entries`, `check_conflicts`, `create_entries` (batch ≤5), `add_reminders`. Restricted (requested but **never executed**): `delete_entry`, `send_email`, `sync_external_calendar`. |
| 6 | **State** | `AgentState` (goal, step, status, stop reason, repair counter, pending approvals, decision/observation log, token counters) + `Sandbox` (entries, reminders). Created per run; never shared. |
| 7 | **Autonomy boundary** | Automatic: read, check, create sandbox entries/reminders. Not automatic: delete, send, sync to anything real → `approval_required`. Enforced by the tool gateway, not by the prompt. |
| 8 | **Primary risks** | Invented dates (ambiguity); instructions hidden in pasted text (injection); malformed/invalid model output; tool timeout or garbage output; runaway loops/cost; model attempting consequential actions; leaking history between chats. |
| 9 | **Evaluation criteria** | Offline unit tests (41), public Arena-style cases (`evaluation/public_cases.json`), 10-input model comparison, multi-turn clarification test, deployed-endpoint checks. |

## 3. Architecture

```
                 +---------------------------- FastAPI (app/main.py, api.py) ----------------------------+
 Browser UI ---> |  GET /  /health  /models  /arena/manifest  /docs                                       |
 (static/)       |  POST /chat (session memory, rate-limited)      POST /arena/run (stateless, Arena)     |
                 +------------------------------------+-----------------------------------------------+
                                                      v
                                   app/arena.py   timeout wrapper, size bound, never crashes
                                                      v
   +----------------------------------------- app/agent.py : AgentRun -----------------------------------------+
   |  GOAL                                                                                                      |
   |   -> OBSERVE   prompts.build_messages(system | history | runtime_state | user msg | untrusted | log)        |
   |   -> DECIDE    llm.complete()  --raw text-->  parse -> schema validate -> semantic validate                 |
   |                    invalid? -> repair (<=2 in a row, each costs a step)  -> else typed contract_error        |
   |   -> ACT       ToolGateway.call()  [autonomy gate | timeout | retry<=2 | output validation | fault inject] |
   |   -> OBSERVE RESULT  append to execution log, update state  -> loop                                         |
   |  STOP: completed | needs_clarification | blocked | approval_required | tool_error | contract_error |       |
   |        budget_exceeded (steps / tokens / time / input size) | failed                                       |
   +------------------------------------------------------------------------------------------------------------+
```

```
i230824/
├── app/
│   ├── main.py      FastAPI composition, admission control (semaphore), per-IP rate store
│   ├── api.py       routes: /health /models /arena/manifest /arena/run /chat (+DELETE reset)
│   ├── arena.py     common boundary: timeout, response-size bound, error containment
│   ├── agent.py     the bounded loop, decision validation, repair, stopping, fault hooks
│   ├── tools.py     sandbox + 4 allowed tools + 3 restricted tools + ToolGateway
│   ├── prompts.py   SYSTEM prompt (static) + per-step TURN template (dynamic) + untrusted fencing
│   ├── llm.py       Gemini / Anthropic clients (httpx), test override, cost estimate
│   ├── models.py    Arena contracts, AgentDecision, AgentState
│   ├── memory.py    bounded in-memory LangChain message history
│   ├── config.py    environment-driven limits and model config
│   └── static/      minimal UI (chat, model selector, external note, status, tool trace, observations)
├── data/sample_data.json            synthetic seed calendar + sample outline
├── evaluation/                      public_cases.json, run_public_tests.py, dev_cases.json, compare_models.py, results/
├── tests/test_agent.py              41 offline tests with a scripted model
├── scripts/                         build_submission.py (PDF + ZIP), verify_deployment.py
├── arena_manifest.json  render.yaml  Dockerfile  requirements.txt  .env.example  run.py
```

## 4. Control flow and stopping rules

* **Step accounting:** one *step* = one model call. Repair attempts count. Hard cap `MAX_STEPS` (≤ 6), further limited
  by `arena_config.max_steps` (the smaller wins).
* **The system owns stopping.** The model can only *propose* a status; limits are checked before every call:

| Condition | `status` | `stop_reason` |
|---|---|---|
| Goal done per observations | `completed` | `goal_completed` |
| Critical info missing / conflicting | `needs_clarification` | `clarification_needed` |
| Out of scope / unsafe | `blocked` | `goal_blocked` |
| A restricted action was requested | `approval_required` | `approval_required` |
| Tool failed after retries | `tool_error` | `tool_failed_after_retries` |
| Model output invalid after repairs | `contract_error` | `invalid_decision_unrecoverable` |
| Step limit hit / model says work can't fit | `budget_exceeded` | `step_budget_exhausted` |
| Token limit hit | `budget_exceeded` | `token_budget_exhausted` |
| Time limit hit | `budget_exceeded` | `time_budget_reached` |
| Input too large | `budget_exceeded` | `input_budget_exceeded` |
| Model API down / bad key | `failed` | `model_unavailable` / `model_not_configured` |

* A run that completes the safe part of a request but had a restricted part is reported as `approval_required`
  (never silently `completed`).

## 5. Structured output & validation (contract-gated execution)

Schema (`app/models.py`):

```python
class AgentDecision(BaseModel):
    status: Literal["continue","needs_clarification","completed","blocked","approval_required","budget_exceeded","failed"]
    action: Optional[str]; arguments: dict; user_message: Optional[str]; rationale: str
    # validator: continue => action required; any other status => action must be null AND user_message required
```

Pipeline for every model reply: **parse** (extract one JSON object, tolerate code fences) → **schema validation**
(Pydantic: enums, types, cross-field rules) → **semantic validation** (action is a registered tool; arguments satisfy that
tool's Pydantic model: real calendar date, `HH:MM`, duration 5–1440, ≤5 items, …) → execute. Failures trigger up to
`MAX_REPAIRS=2` consecutive repair attempts (the violation list is fed back as an observation; each repair uses a step),
then a **typed** `contract_error`.

Business-rule validation inside the tools: date must be **grounded** (`date_evidence` must be quoted from the user's
messages/documents — this is a mechanism against invented dates, not phrase matching), duplicates are idempotent, timed
overlaps are rejected and reported, reminders must reference existing ids. Tool *outputs* are validated too (`ToolOutput`);
malformed output is retried.

## 6. Prompt & context engineering

Layers are kept separate and labelled (`app/prompts.py`):

| Layer | Where | Notes |
|---|---|---|
| SYSTEM | `SystemMessage`, static policy + auto-generated tool catalog | role, trust hierarchy, output contract, decision rules |
| History | real `HumanMessage`/`AIMessage` objects | last 6 turns / 24k chars (`memory.py`) |
| Runtime state | `<runtime_state>` block (**dynamic template**, LangChain `PromptTemplate`) | today, step n/max, remaining steps, repair counter, sandbox counters, pending approvals |
| User goal | `<current_user_message>` | follow-up answers are merged with the earlier request using history |
| Untrusted content | `<untrusted_external_content boundary="<random>">` | fenced with a per-run random token; `<` `>` neutralised so it cannot forge/close blocks |
| Tool observations | `<execution_log>` | neutralised, length-capped; treated as data |

**Trust rule:** only the system prompt and the user's own request are instructions; documents, notes, tool results and
past assistant messages are data. Independently of the prompt, **the gateway enforces authority**, so even a fooled model
cannot delete, email, or sync anything.

## 7. Message-memory policy

* Stable `session_id` (UI generates a UUID; **New chat** calls `DELETE /chat/{id}` and creates a new id).
* Stored as LangChain messages; bounded to **12 messages (6 turns) and 24,000 characters**; max 100 sessions (oldest evicted).
* **In-memory only, single worker:** history is lost on restart/sleep/redeploy. Each run's sandbox calendar is also per run
  (entries created in one request are not visible in the next), by design for isolation and testability.
* `/arena/run` is stateless: `request_id` is only a correlation id.

## 8. Cost controls

| Limit | Default | Env var |
|---|---|---|
| Model decisions per run (repairs included) | 6 | `MAX_STEPS` |
| Tool retries per call | 2 | `MAX_TOOL_RETRIES` |
| Consecutive contract repairs | 2 | `MAX_REPAIRS` |
| Output tokens per model call | 512 | `MAX_OUTPUT_TOKENS` |
| Total tokens per run | 30,000 | `MAX_TOTAL_TOKENS` |
| Input size (task + context + history) | 24,000 chars | `MAX_INPUT_CHARS` |
| Run / model-call / tool timeouts | 40 s / 14 s / 3 s | `RUN_TIMEOUT_SECONDS` etc. |
| Concurrent model runs | 4 | `MAX_CONCURRENT_RUNS` |
| Browser chat rate | 20 / min / IP | `CHAT_RATE_PER_MINUTE` |
| Service-wide daily model calls | 5,000 | `DAILY_MODEL_CALL_CAP` |

Token usage is summed from provider usage metadata; cost is estimated only when `MODEL_PRICES_JSON` has a price for the
model — otherwise `null` (never a fabricated zero). Gemini "thinking" is disabled (`GEMINI_THINKING_BUDGET=0`) so the output cap is not eaten by hidden reasoning tokens.

## 9. Model selection experiment

Method: the same 10 inputs (`evaluation/dev_cases.json`: explicit dates, missing date, clash, outline, injection,
restricted action, relative date, listing, off-topic) were run against each candidate through `POST /chat` with
`evaluation/compare_models.py`. Metrics: task success (status matches expectation), structured-output validity
(model decisions accepted first time ÷ all decisions), correct action selection (expected first tool, or no tool),
latency, tokens, estimated cost.

<!-- MODEL_TABLE_START -->
| Model | Task success | Structured-output validity | Correct action selection | Avg latency / case (ms) | Avg input tok | Avg output tok | Avg cost / case (USD) |
|---|---|---|---|---|---|---|---|
| _not measured yet_ | run `python evaluation/compare_models.py` | | | | | | |
<!-- MODEL_TABLE_END -->

**Selection rationale:** `TODO(student): after running the script, write 3–4 sentences here: which model you deployed and why,
based on the table (reliability first, then latency, context window, and cost). A larger model gets no credit just for being larger.`

## 10. Autonomy boundary

| Automatic (sandbox only) | Blocked / needs approval |
|---|---|
| read & list sandbox entries; check conflicts; create entries; add reminders; summarise | delete entries; send e-mail/messages; write to a real calendar/account |

Restricted tools are registered so the model can *request* them, but `ToolGateway` never executes them: it records a
`rejected` trace and the run ends as `approval_required`.

## 11. Fault injection (Arena)

`arena_config.fault` is honoured at the model/tool boundary, once, at the first matching operation:
`tool_timeout` and `malformed_tool_output` hit the first real tool execution (retried; writes are idempotent so a retry
never duplicates), `invalid_agent_decision` replaces the first model decision with one of three invalid forms
(wrong enum/types, unknown action, non-JSON) which exercises repair. Every fault, retry and repair appears in `events`/`errors`.

## 12. Observability

Each run returns `events` (agent_start, decision + validation result, fault_injected, tool_retry, tool_result,
autonomy_gate, decision_invalid, agent_stop with reason), `tool_calls` (step, tool, attempt, outcome, latency),
`errors`, and `metrics`. The same events are logged as JSON lines. No secrets are logged.

## 13. Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env          # then put your key in .env (never commit it)
.\.venv\Scripts\python.exe run.py    # http://127.0.0.1:8000/  and /docs
```
macOS/Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cp .env.example .env && .venv/bin/python run.py`

Environment variables: `MODEL_PROVIDER`, `MODEL_NAME`, `ALLOWED_MODELS`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_THINKING_BUDGET`, `MODEL_PRICES_JSON`, `MAX_STEPS`, `MAX_TOOL_RETRIES`, `MAX_REPAIRS`, `MAX_OUTPUT_TOKENS`,
`MAX_TOTAL_TOKENS`, `MAX_INPUT_CHARS`, `RUN_TIMEOUT_SECONDS`, `MODEL_TIMEOUT_SECONDS`, `TOOL_TIMEOUT_SECONDS`,
`CHAT_RATE_PER_MINUTE`, `MAX_CONCURRENT_RUNS`, `DAILY_MODEL_CALL_CAP`, `APP_TIMEZONE`, `HOST`, `PORT` (see `.env.example`).

## 14. Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v               # offline, free (scripted model)
.\.venv\Scripts\python.exe evaluation\run_public_tests.py --url http://127.0.0.1:8000   # real model, saves evaluation/results/public_results.json
.\.venv\Scripts\python.exe evaluation\compare_models.py --url http://127.0.0.1:8000     # model comparison, refreshes the table above
```

The offline suite covers: contract validation, ambiguity → clarification, invented-date rejection, conflict handling,
idempotent retries, multi-turn clarification in one chat, session isolation/reset, memory bound, untrusted-content fencing
and forged-tag neutralisation, restricted tools never executing, all three `fault` types, repeated invalid output →
`contract_error`, out-of-range/wrong-typed arguments, persistent tool failure with draft, step/token/input budgets, model outage,
provider request/response handling, and source scans (no hard-coded injection phrases, no secrets).
Offline tests prove the **mechanisms**; they cannot prove model judgement, which is what the public/dev runs measure.

## 15. Deploy on Render

1. Push this folder to a **private** GitHub repo (`git add . && git commit`; `.env` is git-ignored).
2. Render → New → Web Service → connect the repo → Python runtime; Build `pip install -r requirements.txt`;
   Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`; Health check `/health`; plan **Free**.
   (Or use the `render.yaml` Blueprint.)
3. Environment: set `GEMINI_API_KEY` (secret), `MODEL_PROVIDER=gemini`, `MODEL_NAME`, `ALLOWED_MODELS`, optionally `MODEL_PRICES_JSON`.
4. Verify from outside your network: `python scripts/verify_deployment.py https://YOUR-APP.onrender.com`.

## 16. Limitations (honest list)

* Free Render services sleep after inactivity (first request can take ~30–60 s) and may restart; chat history and nothing else is lost.
* One worker, in-memory memory only; entries are not persisted between requests (per-run sandbox).
* Free Gemini quotas are rate limited (RPM/RPD); heavy test bursts may return `model_unavailable`. The agent retries a transient model error once.
* `date_evidence` grounding is a substring check (case/space-insensitive); it blocks invented dates but can occasionally reject a model that paraphrases the evidence — the repair/observation path then lets it retry or ask.
* "Today" is the server's date in `APP_TIMEZONE` (default UTC).
* Prompt-injection resistance for *content* decisions (e.g. a note that supplies a plausible but unwanted date) relies on the model; resistance for *actions* is enforced structurally.
* A model may still declare `completed` too optimistically; the final response always appends the system's factual count of sandbox changes.
