# dr-providers — Investigation Notes

Sibling repo: `../dr-providers`

## Purpose

Small, typed **OpenRouter HTTP client** — the LLM transport layer only. Stable public API (`v0.1.0`); no benchmark, queue, validation, or persistence logic.

## What it implements

| Component | Role |
|-----------|------|
| `LlmRequest` / `LlmResponse` | Frozen Pydantic chat-completions request/response |
| `OpenRouterProvider` | `httpx` client with `tenacity` retries on transport failures |
| `ReasoningSpec` / `SamplingControls` | Maps reasoning effort or enable/disable into OpenRouter `extra_body` |
| Error types | `ProviderTransportError` vs `ProviderSemanticError` |
| `query_from_prompt` | Script helper: string prompt → `LlmResponse.text` |
| CLI | `query-provider` for ad-hoc queries |

**Request flow**: `LlmRequest` → `prepare()` (API key, endpoint, idempotency key, reasoning body) → POST → `LlmResponse` (text, finish_reason, latency_ms, raw_json).

**Dependencies**: httpx, pydantic, tenacity — no LiteLLM.

## What it does not implement

- Prompt templates or workflow orchestration
- Call logging or pool persistence (dr-llm / dr-bottleneck Mongo)
- DSPy integration
- Code extraction or validation
- Test execution

## Relation to other pieces

| Piece | Relationship |
|-------|----------------|
| **dr-bottleneck** | dr-bottleneck uses LiteLLM directly (`dr_bottleneck.llm.client.call_llm`) plus Mongo logging. dr-providers could replace the HTTP layer; orchestration and logging stay in dr-bottleneck. |
| **nl-code** | HumanEval DSPy eval uses DSPy + LiteLLM. dr-providers fits non-DSPy batch generation scripts, not the current optimizer/eval loop. |
| **code-eval** | Downstream consumer of `response.text`. No package dependency today. |
| **dr-llm pool data** | Pool rows include `model`, `provider`, `finish_reason` — fields aligned with what dr-providers returns on new runs. Historical replay skips generation entirely. |

## Starting-state summary

- **Strength**: minimal typed OpenRouter client, explicit errors, reasoning model support, frozen API surface
- **Gap**: OpenRouter only; no built-in logging, profiling, or multi-provider routing
- **Natural role in a stack**: shared generation primitive between orchestration (dr-bottleneck) and post-processing (code-eval → nl-code)
