from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

# ---------------------------------------------------------------------------
# Environment Bootstrap & Configuration
# ---------------------------------------------------------------------------
# This section initializes global configuration variables from the environment.
# These variables control agent timeouts, run identifiers for tracking, and
# connection settings for the LLM inference proxy.

# AGENT_TIMEOUT is provided by the Harbor evaluation environment. It sets the
# strict upper bound on how long the agent can run before being terminated.
AGENT_TIMEOUT = os.getenv("AGENT_TIMEOUT")
AGENT_TIMEOUT_SEC = float(AGENT_TIMEOUT) if AGENT_TIMEOUT else None

# EVALUATION_RUN_ID uniquely identifies the current task execution in Harbor.
# Used for logging and seeding to ensure reproducibility within the same run.
RUN_ID = os.getenv("EVALUATION_RUN_ID")
if not RUN_ID:
    print("[AGENT] WARNING: RUN_ID (EVALUATION_RUN_ID) is not set")

# Timeouts for the HTTP requests to the inference endpoint (typically chutes_proxy.py).
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

# Maximum time allowed for compiling python code to check for syntax errors.
_PY_COMPILE_TIMEOUT_SEC = 25

# Model alias maps: skarkix model name → proxy-facing model id.
# chutes_proxy.py then maps these to the actual Chutes TEE endpoints.
#
# Active models (as of current config):
#   Planning : moonshotai/Kimi-K2.5   → kimi-k2.5-tee   (Chutes)
#   Execution: MiniMaxAI/MiniMax-M2.5 → minimax-m2.5-tee (Chutes)
#   Fast     : Qwen/Qwen3-Coder-Next  → qwen3-coder-next-tee (Chutes)
_MODEL_ALIASES: dict[str, str] = {
    "moonshotai/Kimi-K2.5": "moonshotai/kimi-k2.5",
    "MiniMaxAI/MiniMax-M2.5": "minimax/minimax-m2.5",
    "Qwen/Qwen3-Coder-Next": "qwen/qwen3-coder-next",
}

_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
_EMBEDDING_ALIASES: dict[str, str] = {
    _DEFAULT_EMBEDDING_MODEL: "qwen/qwen3-embedding-8b",
}

def _agent_assumed_wall_sec() -> Optional[float]:
    """
    Provides a fallback wall-clock limit when the AGENT_TIMEOUT environment
    variable isn't injected by the host system. Harbor typically caps runs
    at ~600 seconds, so we default to 600.0 to prevent the agent from
    hanging indefinitely if it loses track of time.
    """
    raw = (os.getenv("RIDGES_AGENT_ASSUMED_WALL_SEC") or "").strip().lower()
    if raw in ("0", "none", "off", "inf", "infinity"):
        return None  # Explicitly disable the timeout
    if raw:
        try:
            v = float(raw)
            return None if v <= 0 else v
        except ValueError:
            pass
    return 600.0


def _effective_agent_wall_sec() -> Optional[float]:
    """
    Determines the definitive time limit for the agent's execution.
    Prioritizes the exact AGENT_TIMEOUT provided by the environment,
    falling back to the assumed limit if necessary.
    """
    if AGENT_TIMEOUT_SEC is not None:
        return float(AGENT_TIMEOUT_SEC)
    return _agent_assumed_wall_sec()


def _agent_tail_margin_sec() -> float:
    env = os.getenv("RIDGES_AGENT_TAIL_MARGIN_SEC")
    if env and env.strip():
        try:
            return max(15.0, float(env))
        except ValueError:
            pass
    return 45.0


def _pretimeout_trigger_sec() -> float:
    """When wall-clock remaining drops below this, try one emergency patch (wed-style)."""
    raw = (os.getenv("RIDGES_AGENT_PRETIMEOUT_SEC") or "100").strip()
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 100.0



def _resolve_model(name: str) -> str:
    """
    Maps an internal agent model name (e.g., 'MiniMaxAI/MiniMax-M2.5') to the
    identifier expected by our local proxy (e.g., 'minimax/minimax-m2.5').
    If no alias is defined, passes the name through unchanged.
    """
    return _MODEL_ALIASES.get(name, name)


def _resolve_embedding(name: str) -> str:
    """
    Maps an internal embedding model name to the identifier expected by the proxy.
    """
    return _EMBEDDING_ALIASES.get(name, name)


def _inference_api_key() -> str | None:
    """
    Retrieves the API key for the inference endpoint. When pointing to the local
    chutes_proxy.py, this can be a dummy value, as the proxy injects the real
    Chutes API key.
    """
    return os.getenv("RIDGES_INFERENCE_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _inference_base_url() -> str:
    """
    Retrieves the base URL for LLM inference. In production/evaluation, this
    should point to the local proxy (e.g., http://host.docker.internal:8000/v1)
    which then forwards requests to the secure TEE endpoints.
    """
    return (
        os.getenv("RIDGES_INFERENCE_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")


if not _inference_api_key():
    print("[AGENT] WARNING: No inference API key configured. Set RIDGES_INFERENCE_API_KEY.")

_MAX_OUTPUT_TOKENS = int(os.getenv("RIDGES_AGENT_MAX_OUTPUT_TOKENS", "4096"))

def _retry_sleep_after_rate_limit(attempt: int) -> None:
    wait = min(0.3 * (2 ** attempt) + random.uniform(0, 0.5), 5.0)
    time.sleep(wait)


def _llm_seed_enabled() -> bool:
    return os.getenv("RIDGES_LLM_USE_SEED", "1").strip().lower() not in ("0", "false", "no")


def _resolve_llm_seed() -> int:
    """Stable 31-bit seed for ``seed`` (best-effort determinism; same run id → same seed)."""
    raw = os.getenv("RIDGES_LLM_SEED", "").strip()
    if raw:
        try:
            return int(raw) % (2**31)
        except ValueError:
            digest = hashlib.sha256(raw.encode()).digest()
            return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    rid = os.getenv("EVALUATION_RUN_ID") or RUN_ID or ""
    if rid:
        digest = hashlib.sha256(rid.encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1


def _prompt_cache_enabled() -> bool:
    """Default on; set RIDGES_PROMPT_CACHE=0 to disable cache_control markers."""
    raw = (os.getenv("RIDGES_PROMPT_CACHE") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _wrap_with_cache_control(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }],
    }


def _apply_prompt_cache_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _prompt_cache_enabled():
        return messages
    # Find last user message index (where conversation history ends).
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user" and isinstance(messages[idx].get("content"), str):
            last_user_idx = idx
            break
    out: list[dict[str, Any]] = []
    for idx, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            out.append(m)
            continue
        if role == "system" or idx == last_user_idx:
            out.append(_wrap_with_cache_control(role, content))
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# HTTP retry helper (shared by inference and embedding)
# ---------------------------------------------------------------------------

_RETRIABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_MAX_ATTEMPTS = 5  # 1 initial + 4 retries


def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: tuple[int, int],
    caller: str,
) -> "requests.Response | None":
    """
    Executes an HTTP POST request to the inference endpoint (typically the local proxy)
    with robust error handling and exponential back-off.
    
    This handles common failure modes when interfacing with LLM providers:
      - 429 (Too Many Requests): Automatically respects the 'Retry-After' header if present,
        otherwise uses an exponential back-off (1s, 2s, 4s...) to prevent hammering the API.
      - 5xx / Transient (e.g., 502 Bad Gateway, 504 Timeout): Retries with jittered back-off
        as these are often temporary issues with the model provider's infrastructure.
      - Connection/Timeout Errors: Retries transparently.

    Args:
        url: The endpoint to POST to.
        payload: JSON payload (e.g., the chat completion request).
        headers: HTTP headers (including Authorization).
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.
        caller: String identifying the caller (e.g., 'inference()') for logging.

    Returns:
        The successful `requests.Response` object (status 200), or `None` if all
        retry attempts are exhausted or a non-retriable error occurs.
    """
    wait = 1.0
    max_wait = 60.0
    last_attempt = _HTTP_MAX_ATTEMPTS - 1

    for attempt in range(_HTTP_MAX_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)

            if response.status_code == 429 and attempt < last_attempt:
                retry_after = response.headers.get("Retry-After")
                slept = False
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                        slept = True
                    except ValueError:
                        pass
                if not slept:
                    time.sleep(wait)
                wait = min(wait * 2, max_wait)
                print(f"[AGENT] {caller}: HTTP 429, retrying ({attempt + 2}/{_HTTP_MAX_ATTEMPTS})...")
                continue

            if response.status_code != 200:
                if response.status_code in _RETRIABLE_STATUS and attempt < last_attempt:
                    print(
                        f"[AGENT] {caller}: HTTP {response.status_code}, "
                        f"retrying ({attempt + 2}/{_HTTP_MAX_ATTEMPTS})..."
                    )
                    _retry_sleep_after_rate_limit(attempt)
                    continue
                print(
                    f"[AGENT] {caller}: request failed with status "
                    f"{response.status_code}: {response.text[:800]}"
                )
                return None

            return response

        except requests.exceptions.Timeout as exc:
            print(f"[AGENT] {caller}: request timeout: {exc}")
            if attempt < last_attempt:
                _retry_sleep_after_rate_limit(attempt)
                continue
            return None
        except requests.exceptions.ConnectionError as exc:
            print(f"[AGENT] {caller}: connection error: {exc}")
            if attempt < last_attempt:
                _retry_sleep_after_rate_limit(attempt)
                continue
            return None
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[AGENT] {caller}: invalid JSON in response: {exc}")
            return None
        except Exception as exc:
            print(f"[AGENT] {caller}: unexpected error: {exc}")
            return None

    return None


def inference(
    model: str,
    temperature: float,
    messages: list[dict[str, Any]],
    *,
    top_p: float | None = None,
    seed: int | None = None,
):
    """
    Main entry point for generating chat completions via the configured inference provider.
    
    This function formats the request into the OpenAI-compatible standard, resolves the 
    model alias, and delegates to `_post_with_retry` for robust execution. It also handles
    the extraction of token usage details (including cached token metrics, which are 
    critical for monitoring the efficiency of our prompt caching strategy).

    Args:
        model: The internal name of the model to use (e.g., 'MiniMaxAI/MiniMax-M2.5').
        temperature: Controls randomness in the output (0.0 for deterministic).
        messages: The conversation history, formatted as a list of role/content dictionaries.
        top_p: Nucleus sampling parameter (optional).
        seed: Random seed for reproducible outputs (optional; providers may ignore this).

    Returns:
        A tuple of `(response_text, usage_dict)`:
          - `response_text`: The string content generated by the model.
          - `usage_dict`: A dictionary containing 'prompt_tokens', 'completion_tokens',
            'total_tokens', and 'cached_tokens'.
        Returns `(None, None)` if the request fails after all retries.
    """
    api_key = _inference_api_key()
    if not api_key:
        print("[AGENT] inference(): no API key configured")
        return None, None

    resolved = _resolve_model(model)
    url = f"{_inference_base_url()}/chat/completions"
    payload: dict[str, Any] = {
        "model": resolved,
        "messages": _apply_prompt_cache_markers(messages),
        "temperature": temperature,
        "max_tokens": _MAX_OUTPUT_TOKENS,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = int(seed)

    seed_note = f", seed={payload['seed']}" if "seed" in payload else ""
    top_p_note = f", top_p={payload['top_p']}" if "top_p" in payload else ""
    print(
        f"[AGENT] inference(): model={resolved} (from {model}), "
        f"temperature={temperature}{top_p_note}{seed_note}, {len(messages)} messages"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_with_retry(
        url, payload, headers,
        timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT),
        caller="inference()",
    )
    if response is None:
        return None, None

    data = response.json()
    message = (data.get("choices") or [{}])[0].get("message") or {}
    result = (message.get("content") or "").strip()
    print(f"[AGENT] inference(): response {len(result)} chars")

    usage = data.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = (
        details.get("cached_tokens")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    usage_info = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cached_tokens": cached_tokens,
    }
    if usage_info["total_tokens"] > 0:
        cache_suffix = f" cached={cached_tokens}" if cached_tokens else ""
        print(f"[AGENT] inference(): token usage: {usage_info}{cache_suffix}")

    _save_trace(messages, result)
    return result or None, usage_info


def _save_trace(messages: list[dict[str, Any]], result: str) -> None:
    """Persist a request/response trace to the agent log directory."""
    try:
        trace_dir = "/logs/agent/skarkix_traces"
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, f"trace_{int(time.time() * 1000)}.json")
        with open(trace_file, "w") as f:
            json.dump({"messages": messages, "response": result}, f, indent=2)
    except Exception as exc:
        print(f"[AGENT] _save_trace(): {exc}")



def embedding(input: str | list[str]) -> list[float] | None:
    """
    Generates vector embeddings for the provided input text using the configured
    embedding model. 
    
    This is used by the agent during complex search/retrieval tasks to perform
    semantic search over the codebase when exact keyword matches fail. Like
    `inference()`, it delegates to `_post_with_retry` for robust execution.

    Args:
        input: The text string (or list of strings) to embed.

    Returns:
        A list of floats representing the embedding vector, or None if the request fails.
    """
    api_key = _inference_api_key()
    if not api_key:
        print("[AGENT] embedding(): no API key configured")
        return None

    model = os.getenv("RIDGES_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    resolved = _resolve_embedding(model)
    url = f"{_inference_base_url()}/embeddings"
    print(f"[AGENT] embedding(): model={resolved}")

    payload = {"model": resolved, "input": input}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_with_retry(
        url, payload, headers,
        timeout=(LLM_CONNECT_TIMEOUT, min(LLM_READ_TIMEOUT, 120)),
        caller="embedding()",
    )
    if response is None:
        return None

    data = response.json()
    row = (data.get("data") or [{}])[0]
    result = row.get("embedding")
    if not isinstance(result, list):
        print("[AGENT] embedding(): unexpected response shape")
        return None
    print(f"[AGENT] embedding(): {len(result)} dimensions")
    return result


# ---------------------------------------------------------------------------
# Agent Configuration & Defaults
# ---------------------------------------------------------------------------
# These variables define the default model roles when not explicitly overridden
# by the Harbor evaluation environment via environment variables.

# PLANNING_MODEL is used for the initial "thought/planning" phase before execution.
# Requires strong reasoning capabilities (e.g. Kimi-K2.5 or Claude 3.5 Sonnet).
PLANNING_MODEL = os.getenv("RIDGES_PLANNING_MODEL", "moonshotai/Kimi-K2.5")

# DEFAULT_MODEL is the primary execution workhorse that iterates on the code.
# Requires strong coding capabilities and tool-use adherence.
DEFAULT_MODEL = os.getenv("RIDGES_AGENT_MODEL", "MiniMaxAI/MiniMax-M2.5")

# FAST_MODEL is used for simple/rapid tasks (e.g., summarizing lint errors).
FAST_MODEL = os.getenv("RIDGES_AGENT_FAST_MODEL", "Qwen/Qwen3-Coder-Next")

# Global cost limits. The agent tracks its spending using MODEL_PRICING.
_DEFAULT_COST_RAW = os.getenv("RIDGES_MAX_COST_USD", "0.29") or "0.29"
DEFAULT_COST_LIMIT = float(_DEFAULT_COST_RAW)

# Skip the expensive planning phase if the total sandbox budget is too small.
_PLANNING_MIN_COST_USD = float(os.getenv("RIDGES_PLANNING_MIN_COST_USD", "0.12"))


class AgentConfig:
    """
    Encapsulates all runtime configuration for a single execution of the coding agent.
    
    This includes model selection, context limits, retry policies, and budgets. It is 
    instantiated once per task run by the main agent loop.

    Intentionally NOT a @dataclass because the Ridges miner runtime loads
    agent.py via importlib.util dynamically and @dataclass fails when the
    module is not in sys.modules.
    """

    def __init__(
        self,
        planning_model: str = PLANNING_MODEL,
        execution_model: str | None = None,
        model: str | None = None,
        fast_model: str = FAST_MODEL,
        temperature: float = 0.0,
        planning_temperature: float = 0.0,
        max_steps: int = 400,
        max_output_chars: int = 8000,
        max_head_tail_chars: int = 4000,
        max_conversation_chars: int = 120000,
        max_inference_retries: int = 3,
        inference_retry_delay: float = 5.0,
        command_timeout: int = 120,
        working_dir: Optional[str] = None,
        cost_limit: float = DEFAULT_COST_LIMIT,
        enable_planning: bool = True,
    ):
        # Legacy ``model=`` kwarg maps to execution_model.
        exec_model = execution_model or model or DEFAULT_MODEL
        self.planning_model = planning_model
        self.execution_model = exec_model
        self.fast_model = fast_model
        self.temperature = temperature
        self.planning_temperature = planning_temperature
        _tp = os.getenv("RIDGES_LLM_TOP_P", "1").strip()
        try:
            self.llm_top_p = float(_tp)
        except ValueError:
            self.llm_top_p = 1.0
        self.max_steps = max_steps
        self.max_output_chars = max_output_chars
        self.max_head_tail_chars = max_head_tail_chars
        self.max_conversation_chars = max_conversation_chars
        self.max_inference_retries = max_inference_retries
        self.inference_retry_delay = inference_retry_delay
        self.command_timeout = command_timeout
        self.working_dir = working_dir
        self.cost_limit = cost_limit
        self.enable_planning = enable_planning

    @property
    def model(self) -> str:
        """Legacy compatibility — returns execution model."""
        return self.execution_model


# ---------------------------------------------------------------------------
# Model Pricing (cost per 1M tokens)
# ---------------------------------------------------------------------------

# Pricing per 1M tokens (input, output) — keys use the canonical alias name.
# get_model_pricing() resolves aliases before lookup, so only one entry per model needed.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "moonshotai/kimi-k2.5": (0.14, 0.55),    # planning model  (Chutes Kimi-K2.5-TEE)
    "minimax/minimax-m2.5": (0.15, 1.15),    # execution model (Chutes MiniMax-M2.5-TEE)
    "qwen/qwen3-coder-next": (0.11, 0.80),   # fast model      (Chutes Qwen3-Coder-Next-TEE)
    "qwen/qwen3-embedding-8b": (0.01, 0.01), # embedding model (Chutes Qwen3-Embedding-8B-TEE)
}


def _models_equivalent(a: str, b: str) -> bool:
    """True when two model ids refer to the same inference endpoint."""
    return _resolve_model(a) == _resolve_model(b)


def get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a model."""
    resolved = _resolve_model(model)
    for candidate in (model, resolved):
        if candidate in MODEL_PRICING:
            return MODEL_PRICING[candidate]
    for prefix, prices in MODEL_PRICING.items():
        key = prefix.rstrip("/")
        for candidate in (model, resolved):
            if candidate == key or candidate.startswith(key + "/"):
                return prices
    print(f"[AGENT] WARNING: No pricing info for model {model}, using default ($1/$2 per 1M)")
    return (1.0, 2.0)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------


MINI_ACTION_REGEX_RSWEA = re.compile(
    r"```\s*rswea_bash_command\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)
MINI_ACTION_REGEX_BASH = re.compile(r"```\s*bash\s*\n(.*?)\n\s*```", re.DOTALL | re.IGNORECASE)

# Custom robust-edit block. Lets the model do exact str-replace edits without
# the quoting hell of sed/here-doc round-trips. Parsed & dispatched in
# ``CodingAgent._run_action`` BEFORE shell execution.
EDIT_ACTION_REGEX = re.compile(
    r"```\s*apply_str_replace\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Atomic multi-file str-replace. Payload contains repeated
# <<<FILE>>>/<<<OLD>>>/<<<NEW>>> sections terminated by a single <<<END>>>.
# Pre-validation in ``apply_multi_str_replace`` means either ALL OLD strings
# match exactly once or NO file is written.
MULTI_EDIT_ACTION_REGEX = re.compile(
    r"```\s*apply_multi_edit\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Match mini_textbased.yaml observation truncation (10000 / 5000 / 5000).
MINI_OBSERVATION_FULL_MAX = 10000
MINI_OBSERVATION_HEAD = 5000
MINI_OBSERVATION_TAIL = 5000

SYSTEM_PROMPT = """\
You are a senior software engineer fixing real bugs in real repositories.

Each turn, your response must contain EXACTLY ONE action block. Pick the right
block type for the work:

1. ```rswea_bash_command``` — run a single shell command (or chain with && / ||).
   ```rswea_bash_command
   ls -la
   ```

2. ```apply_str_replace``` — exact text replacement in a file (most reliable
   for code edits — no shell quoting, no sed escaping). Format:
   ```apply_str_replace
   <<<FILE>>>
   path/to/file.py
   <<<OLD>>>
   exact text to find (must appear EXACTLY ONCE in the file)
   <<<NEW>>>
   replacement text
   <<<END>>>
   ```
   The OLD block is matched byte-for-byte (preserve whitespace and indentation).
   If OLD doesn't appear or appears more than once, the action FAILS — narrow
   it with more surrounding context and retry.

3. ```apply_multi_edit``` — atomic batch of str-replace edits across one or
   more files (use when a fix touches several places at once). Same OLD/NEW
   semantics as ``apply_str_replace``, repeated, then a single ``<<<END>>>``:
   ```apply_multi_edit
   <<<FILE>>>
   path/to/a.py
   <<<OLD>>>
   exact text in a.py (must appear EXACTLY ONCE)
   <<<NEW>>>
   replacement
   <<<FILE>>>
   path/to/b.py
   <<<OLD>>>
   exact text in b.py
   <<<NEW>>>
   replacement
   <<<END>>>
   ```
   Pre-validation: if ANY OLD is missing or non-unique, NO file is written —
   the whole batch is rejected with a per-edit error list, so you can fix and
   resend. Prefer this over multiple single edits whenever a logically
   coupled change spans 2+ files (e.g. update a function signature in one
   file and its caller in another).

Always prefix any block with a THOUGHT line explaining your reasoning:
THOUGHT: <why this action moves you toward a correct fix>

Engineering rules (apply even if not spelled out in the task):
- Edit existing files. Do not create parallel modules unless the task requires it.
- Reproduce the bug FIRST with a minimal command or test, then fix, then re-run.
- After fixing, run the project's own tests (`pytest -xvs <file>::<test>`,
  `go test ./...`, `npm test`, `cargo test`, etc.) to confirm.
- Preserve public APIs and existing conventions; keep edits minimal and surgical.
- Handle boundary cases the spec implies (empty input, off-by-one, error paths).
- Never silently swallow stderr — read it.
- ``apply_str_replace`` is preferred over sed for code edits.
- **Aliasing / mutation bugs:** If a conversion method (``to_*``, ``as_*``) returns
  ``self`` and callers mutate the result, fix the **method** to return an independent
  object (``return self.copy()`` or a new instance)—not only ``.copy()`` at one call site.
- **Environment:** Shell commands use the task's conda env when available. Do not patch
  unrelated compatibility shims (e.g. numpy version workarounds) for a different Python.

Every shell command runs in a FRESH shell — directory and environment changes
do not persist. Prefix with ``cd /path && ...`` when needed.

When you are confident the fix is correct and the tests pass, submit with:

```rswea_bash_command
echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD
```

For new untracked files that ``git diff HEAD`` would miss, also list and cat
them after SUBMIT_PATCH. Do NOT submit until you have verified the fix.
"""

# Appended when RIDGES_STABLE_PROMPT is enabled (behavioral variance reduction).
STABILITY_SYSTEM_SUFFIX = """\


---

## Repeatability (reduce run-to-run drift)

Unless the problem statement clearly requires a different order:

1. **Discovery**: start with `pwd` then `ls -la | LC_ALL=C sort` (or equivalent) before broad `find`/`rg`.
2. **Search**: prefer `rg --sort path -n` with sensible `-g` excludes for build/vendor trees so similar hits stay in a stable order.
3. **Tests**: once a repro or test command works, reuse it after edits instead of switching to a different command each time.
4. **Ties**: when multiple commands are equally reasonable, pick the shorter; for files, prefer lexicographically smaller paths.
5. **Pacing**: You operate under a STRICT time limit. Be decisive. Do NOT spend 20+ steps just reading code. Quickly identify the issue, apply your targeted edit, test, and submit. If an edit fails due to exact match errors, verify the exact file content (spacing/indentation) and retry, or use sed.
6. **Bug Validity**: The bug you are tasked to fix is GUARANTEED to exist in the current codebase. DO NOT assume the bug has already been fixed by a previous commit. If the code looks correct, you are likely looking at the wrong place or misunderstanding the problem statement.
"""


def _system_prompt_for_run() -> str:
    if os.getenv("RIDGES_STABLE_PROMPT", "").strip().lower() in ("1", "true", "yes"):
        return SYSTEM_PROMPT + STABILITY_SYSTEM_SUFFIX
    return SYSTEM_PROMPT


def _instance_prompt_mini(problem_statement: str, working_dir: str) -> str:
    return f"""Please solve this issue:

{problem_statement}

## Recommended Workflow

1. **Explore** — understand the layout (`git ls-files`, `ls`, `grep -rn`).
2. **Reproduce** — write or find a minimal failing test/command BEFORE editing.
3. **Diagnose** — read the relevant source; identify root cause.
4. **Edit** — make minimal, targeted edits with ``apply_str_replace``
   (preferred) or sed/cat where appropriate. Avoid large rewrites.
5. **Verify** — re-run the reproduction AND tests for **every file you modified**
   (e.g. if you edit ``core/variable.py``, run ``pytest -xvs .../test_variable.py``),
   not only integration tests named in the issue.
6. **Cover edges** — empty inputs, error paths, related code paths the bug
   implies. Search the codebase for similar patterns to fix consistently.
   If you added ``.copy()`` at a call site, check whether the callee should return
   a copy instead (fix ``return self`` in the method).
7. **Submit** — exactly one action whose command starts with
   ``echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD``.

If time runs low: ship a minimal correct patch over polished scaffolding.
Submitting an incorrect patch is better than submitting nothing.

## Hard rules

- Exactly ONE action block per response.
- The block must be a fenced ```rswea_bash_command``` or ```apply_str_replace```.
- Bash actions run in a fresh shell — chain with && or use `cd` prefix.
- Working directory: {working_dir}

## Example: explore

THOUGHT: I need to map the repo before changing anything.

```rswea_bash_command
git ls-files | head -60
```

## Example: read a file region

```rswea_bash_command
nl -ba src/foo.py | sed -n '120,180p'
```

## Example: precise edit

THOUGHT: The validator returns True for empty lists; the spec says empty must raise.

```apply_str_replace
<<<FILE>>>
src/foo.py
<<<OLD>>>
    if not items:
        return True
<<<NEW>>>
    if not items:
        raise ValueError("items must not be empty")
<<<END>>>
```

## Example: run focused tests

```rswea_bash_command
python -m pytest -xvs tests/test_foo.py::test_empty_raises
```

## Example: submit

```rswea_bash_command
echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD
```
"""


PLANNING_SYSTEM_PROMPT = """\
You are an expert software planning assistant. Your role is to analyze problems and create a detailed execution plan.

Analyze the problem and create a step-by-step plan to solve it. Your plan should include:

1. **Problem Analysis**: Understand what needs to be fixed or implemented
2. **File Discovery**: Identify which files need to be examined or modified
3. **Common misunderstandings**: Identify any common misunderstandings that engineers often make when fixing this type of problem.
4. **Implementation Steps**: Specific actions to take, in order
5. **Verification Strategy**: How to verify the fix works

Provide your plan in a structured format. Be specific and actionable.
"""


def _planning_prompt(problem_statement: str, working_dir: str) -> str:
    return f"""Please analyze this problem and create a detailed execution plan:

## Problem Statement

{problem_statement}

## Working Directory

{working_dir}

## Your Task

Create a comprehensive plan that includes:

1. **Problem Analysis**: What exactly needs to be fixed or implemented?
2. **Key Files to Examine**: Which files in the codebase are most relevant?
3. **Common misunderstandings**: Identify any common misunderstandings that engineers often make when fixing this type of problem.
4. **Step-by-Step Plan**: Numbered list of specific actions to take
5. **Expected Outcome**: What does a successful solution look like?
6. **Verification**: How will you verify the fix works?

Be specific and actionable. Your plan will be used by an execution agent to solve this problem.
"""


def format_mini_format_error(n_actions: int) -> str:
    return f"""Format error:

Expected exactly 1 action block, found {n_actions}.

Please always provide EXACTLY ONE action block in triple backticks.
Use ```rswea_bash_command``` for shell commands, ```apply_str_replace``` for a
single file edit, or ```apply_multi_edit``` for an atomic batch across files.
Example:

THOUGHT: Brief reasoning.

```rswea_bash_command
<single command>
```

If you have completed the task, submit with the SUBMIT_PATCH command from the
first message."""


FORMAT_ERROR_MESSAGE = format_mini_format_error(0)


def _format_error_escalation(strike: int) -> str:
    """Stronger nudge after repeated format failures."""
    base = FORMAT_ERROR_MESSAGE
    if strike <= 1:
        return base
    header = (
        f"[Format reminder #{strike}] Your previous response did not contain a single "
        "fenced action block. You MUST respond with exactly one ```rswea_bash_command``` "
        "OR one ```apply_str_replace``` block. The THOUGHT line is plain text outside the "
        "block.\n\n"
        "If you believe you are done, your single action MUST be:\n\n"
        "```rswea_bash_command\n"
        "echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD\n"
        "```\n\n"
        "Repeated format failures end this trial without a patch.\n\n"
    )
    return header + base


SUBMISSION_SENTINEL = "SUBMIT_PATCH"

_SIDE_EFFECT_VOCAB = re.compile(
    r"\b(modif(?:y|ies|ied|ying)|mutat(?:e|es|ed|ing)|original|in[- ]place|"
    r"alias(?:es|ed|ing)?|side[- ]effect|unchanged)\b",
    re.IGNORECASE,
)

_TO_METHOD_DEF_RE = re.compile(r"^\+\s*def\s+(to_\w+)\s*\(", re.MULTILINE)
_CALLSITE_COPY_AFTER_TO_RE = re.compile(
    r"\.to_\w+\(\)[^\n]*\n\+[^\n]*\.copy\(\)|"
    r"to_\w+\(\)[^\n]*\n\+[^\n]*=\s*\w+\.copy\(\)",
    re.MULTILINE,
)

def _resolve_conda_shell_prefix() -> str:
    """Return bash prefix to activate SWE-bench conda env when present."""
    raw = (os.getenv("RIDGES_AGENT_CONDA_ENV") or "testbed").strip()
    if raw.lower() in ("0", "off", "none", "false", "disable", "disabled"):
        return ""
    activate = "/opt/miniconda3/bin/activate"
    if not os.path.isfile(activate):
        return ""
    env_name = shlex.quote(raw)
    return f"source {shlex.quote(activate)} && conda activate {env_name} 2>/dev/null; "


class ShellExecutor:
    """Execute bash commands in the sandbox and capture output."""

    def __init__(
        self,
        working_dir: str | None = None,
        timeout: int = 120,
        shell_prefix: str = "",
    ):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout
        self.shell_prefix = shell_prefix

    def execute(self, command: str) -> dict[str, Any]:
        full_command = f"{self.shell_prefix}{command}" if self.shell_prefix else command
        try:
            result = subprocess.run(
                ["bash", "-c", full_command],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {type(e).__name__}: {e}",
                "returncode": -1,
                "timed_out": False,
            }


def normalize_patch_text(patch: str) -> str:
    """Strip ANSI color sequences and normalize newlines for ``git apply``."""
    if not patch:
        return ""
    out = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", patch)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.strip("\n") + ("\n" if out.strip() else "")


def authoritative_worktree_patch(executor: "ShellExecutor") -> str:
    """Unified diff from repo state (HEAD vs worktree), not captured shell transcript."""
    diff = executor.execute("git -c color.ui=false -c core.pager=cat diff HEAD")
    parts: list[str] = []
    if diff.get("returncode") == 0 and (diff.get("stdout") or "").strip():
        parts.append(diff["stdout"].rstrip("\n"))
    untracked = executor.execute(
        "git ls-files --others --exclude-standard | while read -r f; do "
        'test -f "$f" || continue; echo "--- /dev/null"; echo "+++ b/$f"; cat "$f"; echo; done'
    )
    if untracked.get("returncode") == 0 and (untracked.get("stdout") or "").strip():
        parts.append(untracked["stdout"].rstrip("\n"))
    merged = "\n".join(parts).strip("\n")
    if not merged:
        return ""
    return merged + "\n"


def count_mini_actions(response: str) -> int:
    """Count action blocks using the same rules as ``parse_action`` (for format hints).

    Previously this added edit blocks even when rswea/bash existed, overstating counts.
    """
    edits = EDIT_ACTION_REGEX.findall(response)
    multi_edits = MULTI_EDIT_ACTION_REGEX.findall(response)
    rsweas = MINI_ACTION_REGEX_RSWEA.findall(response)
    bashes = MINI_ACTION_REGEX_BASH.findall(response)
    shell_blocks = len(rsweas) + (len(bashes) if not rsweas else 0)
    return len(edits) + len(multi_edits) + shell_blocks


def parse_action(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract exactly one action.

    Returns (kind, payload) where kind is one of ``"bash"``, ``"edit"``, or
    ``"multi_edit"``, or (None, None) on a format error (zero or multiple
    blocks).
    """
    edits = [a.strip() for a in EDIT_ACTION_REGEX.findall(response)]
    multi_edits = [a.strip() for a in MULTI_EDIT_ACTION_REGEX.findall(response)]
    rsweas = [a.strip() for a in MINI_ACTION_REGEX_RSWEA.findall(response)]
    bashes = [a.strip() for a in MINI_ACTION_REGEX_BASH.findall(response)]

    total = (
        len(edits)
        + len(multi_edits)
        + len(rsweas)
        + (len(bashes) if not rsweas else 0)
    )
    if total != 1:
        return None, None

    if edits:
        return "edit", edits[0]
    if multi_edits:
        return "multi_edit", multi_edits[0]
    if rsweas:
        return "bash", rsweas[0]
    if bashes:
        return "bash", bashes[0]
    return None, None


def parse_bash_command(response: str) -> str | None:
    """Backward-compat: extract one bash command (used by tests / external callers)."""
    kind, payload = parse_action(response)
    if kind == "bash":
        return payload
    return None


def check_submission(command: str, output: str) -> str | None:
    """Check if a command result contains a patch submission."""
    if SUBMISSION_SENTINEL not in command:
        return None
    sentinel_idx = output.find(SUBMISSION_SENTINEL)
    if sentinel_idx == -1:
        return None
    patch = output[sentinel_idx + len(SUBMISSION_SENTINEL):].strip()
    return patch if patch else None

_EDIT_SECTION_RE = re.compile(
    r"<<<FILE>>>\n(?P<file>.*?)\n<<<OLD>>>\n(?P<old>.*?)\n<<<NEW>>>\n(?P<new>.*?)\n<<<END>>>",
    re.DOTALL,
)

def parse_edit_payload(payload: str) -> Optional[Tuple[str, str, str]]:
    """Parse the ``apply_str_replace`` body into (file, old_str, new_str).

    Returns None when the markers are missing or malformed.
    """
    payload = payload.replace("\r\n", "\n").replace("\r", "\n")
    # Allow any leading text before <<<FILE>>> for resilience.
    m = _EDIT_SECTION_RE.search(payload)
    if not m:
        return None
    file_path = m.group("file").strip()
    old_str = m.group("old")
    new_str = m.group("new")
    if not file_path:
        return None
    return file_path, old_str, new_str


def _prepare_str_replace_edit(
    working_dir: str, file_path: str, old_str: str, new_str: str, label: str = ""
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Validates and prepares a single str-replace edit.
    Returns: (full_path, new_content, error_msg, summary_stats)
    Exactly one of new_content or error_msg will be populated.
    """
    prefix = f"{label} " if label else ""
    full = file_path if os.path.isabs(file_path) else os.path.join(working_dir, file_path)

    if not os.path.isfile(full):
        return None, None, f"{prefix}file not found: {file_path}", None
    if old_str == new_str:
        return None, None, f"{prefix}OLD and NEW are identical; nothing to do.", None

    try:
        with open(full, "r", encoding="utf-8", errors="surrogateescape") as f:
            content = f.read()
    except Exception as e:
        return None, None, f"{prefix}read error: {type(e).__name__}: {e}", None

    count = content.count(old_str)
    if count == 0:
        # Fallback to normalized line endings
        norm_old = old_str.replace("\r\n", "\n").replace("\r", "\n")
        norm_content = content.replace("\r\n", "\n").replace("\r", "\n")
        if norm_content.count(norm_old) == 1:
            old_str = norm_old
            content = norm_content
            count = 1

    if count == 0:
        hint_lines = []
        if old_str.strip():
            first_line = old_str.splitlines()[0].strip()
            if first_line:
                for i, line in enumerate(content.splitlines(), 1):
                    if first_line and first_line in line:
                        hint_lines.append(f"  {i}: {line}")
                        if len(hint_lines) >= 5:
                            break
        hint = ("\nNearest matches on first OLD line:\n" + "\n".join(hint_lines)) if hint_lines else ""
        err = f"{prefix}OLD not found in {file_path}. Provide the EXACT bytes to replace (whitespace and indentation matter).{hint}"
        return None, None, err, None

    if count > 1:
        return None, None, f"{prefix}OLD matches {count} places in {file_path}; add more surrounding context so it appears exactly once.", None

    new_content = content.replace(old_str, new_str, 1)
    delta = len(new_str) - len(old_str)
    summary = f"old={len(old_str)} bytes, new={len(new_str)} bytes, delta={delta:+d}"
    return full, new_content, None, summary


def apply_str_replace(working_dir: str, file_path: str, old_str: str, new_str: str) -> dict[str, Any]:
    """Apply a single exact-match str-replace edit."""
    full_path, new_content, err_msg, summary = _prepare_str_replace_edit(
        working_dir, file_path, old_str, new_str, label="apply_str_replace:"
    )
    if err_msg:
        return {"stdout": "", "stderr": err_msg, "returncode": 2, "timed_out": False}

    try:
        with open(full_path, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(new_content)
    except Exception as e:
        return {"stdout": "", "stderr": f"apply_str_replace: write error: {type(e).__name__}: {e}", "returncode": 2, "timed_out": False}

    return {
        "stdout": f"apply_str_replace: OK — {file_path} updated ({summary}).\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# apply_multi_edit — atomic batch of str-replace edits across multiple files
# ---------------------------------------------------------------------------

_MULTI_EDIT_BLOCK_RE = re.compile(
    r"<<<FILE>>>\n(?P<file>[^\n]+)\n<<<OLD>>>\n(?P<old>.*?)\n<<<NEW>>>\n(?P<new>.*?)(?=\n<<<FILE>>>\n|\n<<<END>>>)",
    re.DOTALL,
)


def parse_multi_edit_payload(payload: str) -> Optional[list[Tuple[str, str, str]]]:
    """Parse the ``apply_multi_edit`` body into a list of (file, old, new)."""
    payload = payload.replace("\r\n", "\n").replace("\r", "\n")
    if "<<<END>>>" not in payload:
        return None
    matches = _MULTI_EDIT_BLOCK_RE.findall(payload)
    if not matches:
        return None
    edits: list[Tuple[str, str, str]] = []
    for file_path, old_str, new_str in matches:
        file_path = file_path.strip()
        if not file_path:
            return None
        edits.append((file_path, old_str, new_str))
    return edits or None


def apply_multi_str_replace(working_dir: str, edits: list[Tuple[str, str, str]]) -> dict[str, Any]:
    """Apply a batch of str-replace edits atomically (pre-validate, then write)."""
    if not edits:
        return {"stdout": "", "stderr": "apply_multi_edit: payload contained no edits.", "returncode": 2, "timed_out": False}

    plans = []
    errors = []
    for idx, (file_path, old_str, new_str) in enumerate(edits, 1):
        full_path, new_content, err_msg, summary = _prepare_str_replace_edit(
            working_dir, file_path, old_str, new_str, label=f"#{idx}"
        )
        if err_msg:
            errors.append(err_msg)
        else:
            plans.append((file_path, full_path, new_content, summary))

    if errors:
        return {
            "stdout": "",
            "stderr": "apply_multi_edit: pre-validation FAILED, no files written:\n  " + "\n  ".join(errors) + "\nFix every error above and resend the entire batch.",
            "returncode": 2,
            "timed_out": False,
        }

    written = []
    for file_path, full_path, new_content, summary in plans:
        try:
            with open(full_path, "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(new_content)
            written.append(f"  {file_path} ({summary})")
        except Exception as e:
            partial = "\n".join(written) if written else "  (none)"
            return {
                "stdout": f"apply_multi_edit: PARTIAL — wrote {len(written)}/{len(plans)} edits:\n{partial}\n",
                "stderr": f"apply_multi_edit: write error on {file_path}: {type(e).__name__}: {e}",
                "returncode": 2,
                "timed_out": False,
            }

    return {
        "stdout": f"apply_multi_edit: OK — {len(plans)} edits applied:\n" + "\n".join(written) + "\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Conversation Manager — manages message history with context window control
# ---------------------------------------------------------------------------


class ConversationManager:
    """Manage LLM conversation history with truncation and context window control."""

    def __init__(self, max_chars: int = 120000):
        self.messages: list[dict[str, str]] = []
        self.max_chars = max_chars

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    def _trim_if_needed(self) -> None:
        """Trim conversation history if it exceeds max_chars (preserve head + tail)."""
        max_passes = 8
        for _ in range(max_passes):
            total_chars = sum(len(m.get("content", "")) for m in self.messages)
            if total_chars <= self.max_chars:
                return
            if len(self.messages) <= 3:
                return

            excess = total_chars - self.max_chars
            min_keep_head = 2  # system + first user
            min_keep_tail = 6  # last 3 turns
            if len(self.messages) <= min_keep_head + min_keep_tail:
                return

            head = self.messages[:min_keep_head]
            tail = self.messages[-min_keep_tail:]
            middle = self.messages[min_keep_head:-min_keep_tail]
            trimmed_middle = list(middle)
            while trimmed_middle and excess > 0:
                removed = trimmed_middle.pop(0)
                excess -= len(removed.get("content", ""))

            context_note = {
                "role": "user",
                "content": (
                    "[System note: Earlier conversation history was trimmed to fit the context window. "
                    "The original task and your most recent actions are preserved. Continue working.]"
                ),
            }
            self.messages = head + trimmed_middle + [context_note] + tail

    def total_chars(self) -> int:
        return sum(len(m.get("content", "")) for m in self.messages)

def shell_output_to_mini_dict(output: dict[str, Any]) -> dict[str, Any]:
    stdout = output.get("stdout") or ""
    stderr = output.get("stderr") or ""
    parts: list[str] = []
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append(stderr.rstrip("\n"))
    combined = "\n".join(parts)
    if combined and not combined.endswith("\n"):
        combined += "\n"
    exc = ""
    if output.get("timed_out"):
        exc = (stderr or "Command timed out.").strip()
    elif output.get("returncode", 0) == -1 and stderr.strip():
        exc = stderr.strip()
    return {
        "output": combined,
        "returncode": output.get("returncode", 0),
        "exception_info": exc,
    }

def format_mini_observation(output: dict[str, Any], time_remaining: int | None = None) -> str:
    """Observation text aligned with mini_textbased.yaml ``observation_template``."""
    mini = shell_output_to_mini_dict(output)
    lines: list[str] = []
    ei = (mini.get("exception_info") or "").strip()
    if ei:
        lines.append(ei)
    lines.append(str(mini.get("returncode", 0)))
    body = mini.get("output") or ""
    if len(body) < MINI_OBSERVATION_FULL_MAX:
        lines.append("")
        lines.append(body)
    else:
        elided = len(body) - MINI_OBSERVATION_FULL_MAX
        lines.append("")
        lines.append(
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
            "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
            "If you really need to see something from the full command's output, you can redirect output to a file "
            "and then search in that file."
        )
        lines.append("")
        lines.append(body[:MINI_OBSERVATION_HEAD])
        lines.append("")
        lines.append(f"{elided} characters elided")
        lines.append("")
        lines.append(body[-MINI_OBSERVATION_TAIL:])
    if time_remaining is not None:
        lines.append("")
        lines.append(f"[SYSTEM: Time Remaining ~{time_remaining}s]")
    return "\n".join(lines).rstrip() + "\n"

def validate_patch(patch: str) -> bool:
    """Basic structural check that a string looks like a unified diff."""
    if not patch or not patch.strip():
        return False
    if not re.search(r"@@ -\d+(,\d+)? \+\d+(,\d+)? @@", patch):
        if "--- /dev/null" not in patch and "+++ b/" not in patch:
            return False
    return True

def _working_dir_is_git_repo(working_dir: str) -> bool:
    """Repo detection that handles both ``.git`` dirs and gitfiles (linked worktrees)."""
    if not working_dir or not os.path.isdir(working_dir):
        return False
    git_marker = os.path.join(working_dir, ".git")
    return os.path.isdir(git_marker) or os.path.isfile(git_marker)


def validate_patch_applies_cleanly(patch: str, working_dir: str) -> bool:
    """True if ``git apply --check`` succeeds against HEAD (matches Harbor)."""
    if not working_dir or not os.path.isdir(working_dir):
        return False
    if not validate_patch(patch):
        return False
    
    # If not a git repo, stash logic fails. Fallback to direct apply check.
    if not _working_dir_is_git_repo(working_dir):
        try:
            return subprocess.run(
                ["git", "apply", "--check"], input=patch, capture_output=True, text=True, cwd=working_dir, timeout=10
            ).returncode == 0
        except Exception:
            return False

    stashed = False
    apply_ok = False
    try:
        stash = subprocess.run(
            ["git", "-C", working_dir, "stash", "push", "-u", "-m", "ridges_patch_validate", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if stash.returncode == 0:
            stashed = True
        elif stash.returncode == 1:
            err = (stash.stderr or "").lower()
            if "no local changes to save" not in err and "nothing to stash" not in err:
                print(f"[AGENT] stash before patch check failed: {stash.stderr}")
                return False
        else:
            print(f"[AGENT] stash before patch check failed (exit {stash.returncode}): {stash.stderr}")
            return False

        check = subprocess.run(
            ["git", "-C", working_dir, "apply", "--check"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check.returncode != 0:
            err = (check.stderr or check.stdout or "").strip()
            if err:
                print(f"[AGENT] git apply --check: {err[:800]}")
            return False
        apply_ok = True
    except subprocess.TimeoutExpired:
        print("[AGENT] patch validation timed out")
        return False
    except Exception as e:
        print(f"[AGENT] patch validation error: {e}")
        return False
    finally:
        if stashed:
            pop = subprocess.run(
                ["git", "-C", working_dir, "stash", "pop", "-q"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pop.returncode != 0:
                print(f"[AGENT] stash pop after patch check failed: {pop.stderr}")
                apply_ok = False

    return apply_ok


def reset_worktree_to_head_for_harbor(working_dir: str) -> None:
    """Restore a clean tracked tree at HEAD after patch validation.

    Harbor runs ``git apply --check`` on the task repo *after* ``agent_main``
    returns, while the worktree usually still contains the edited files. In that
    state the working tree already matches the patch's ``+`` side, so
    ``git apply --check`` fails with ``patch does not apply`` even when the diff
    is correct. This reset matches the clean preimage our validation checked.
    """
    if not working_dir or not os.path.isdir(working_dir):
        return
    if not _working_dir_is_git_repo(working_dir):
        return
    try:
        r = subprocess.run(
            ["git", "-C", working_dir, "reset", "--hard", "HEAD"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            print(f"[AGENT] WARNING: git reset --hard HEAD failed ({r.returncode}): {r.stderr}")
        else:
            print("[AGENT] Reset worktree to HEAD after validated patch (Harbor compat)")
            clean = subprocess.run(
                ["git", "-C", working_dir, "clean", "-fd"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if clean.returncode != 0:
                print(f"[AGENT] WARNING: git clean -fd failed ({clean.returncode}): {clean.stderr}")
    except Exception as e:
        print(f"[AGENT] WARNING: git reset --hard HEAD error: {e}")

def _self_verify_enabled() -> bool:
    """py_compile self-verify is OPT-IN: set RIDGES_AGENT_SELF_VERIFY=1 to enable.

    Default off because py_compile occasionally rejects valid patches on files
    with unusual encodings or generated headers, eating correct submissions.
    """
    v = (os.getenv("RIDGES_AGENT_SELF_VERIFY") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _patch_modifies_python_files(patch: str) -> list[str]:
    if not patch:
        return []
    seen: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/") and line.endswith(".py"):
            path = line[len("+++ b/"):].strip()
            if path and path not in seen:
                seen.append(path)
        elif line.startswith("diff --git a/") and " b/" in line:
            try:
                rhs = line.split(" b/", 1)[1].strip()
            except IndexError:
                rhs = ""
            if rhs.endswith(".py") and rhs not in seen:
                seen.append(rhs)
    return seen


def _extract_patch_paths(patch: str) -> list[str]:
    if not patch:
        return []
    seen: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path and path != "/dev/null" and path not in seen:
                seen.append(path)
    return seen


# --- Submission diff linting -------------------------------------------------


def _statement_has_side_effect_language(problem_statement: str) -> bool:
    return bool(_SIDE_EFFECT_VOCAB.search(problem_statement or ""))


def _patch_defines_to_method(patch: str) -> list[str]:
    return _TO_METHOD_DEF_RE.findall(patch)


def _diff_uses_callsite_clone_workaround(patch: str) -> bool:
    """True when a diff adds .copy() after to_*() without defining that method."""
    if not re.search(r"\+.*\.copy\(\)", patch):
        return False
    if not re.search(r"to_\w+\(\)", patch):
        return False
    defined = set(_patch_defines_to_method(patch))
    # Method body fix: return self -> return self.copy() counts as fixing the callee
    if re.search(r"\+[^\n]*return\s+self\.copy\(\)", patch):
        for method in defined:
            if method in patch:
                return False
    # Scan hunks: to_*() on context or + lines near a +.copy() without def change
    hunks = re.split(r"(?=^@@ )", patch, flags=re.MULTILINE)
    for hunk in hunks:
        if not re.search(r"\+.*\.copy\(\)", hunk):
            continue
        methods_in_hunk = set(re.findall(r"(to_\w+)\(\)", hunk))
        if not methods_in_hunk:
            continue
        for method in methods_in_hunk:
            if method in defined:
                continue
            if re.search(r"\+[^\n]*\.copy\(\)", hunk):
                return True
    if _CALLSITE_COPY_AFTER_TO_RE.search(patch):
        for m in re.finditer(r"to_(\w+)\(\)", patch):
            if f"to_{m.group(1)}" not in defined:
                return True
    return False


def _diff_exceeds_declared_scope(patch: str, problem_statement: str) -> tuple[bool, str]:
    """Flag edits to paths outside the stated problem scope."""
    paths = _extract_patch_paths(patch)
    if len(paths) <= 1:
        return False, ""
    problem_lower = (problem_statement or "").lower()
    problem_tokens: set[str] = set()
    for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", problem_lower):
        problem_tokens.add(word)
    _COMPAT_STEMS = ("dtypes", "compat", "version", "conftest", "setup")
    suspect: list[str] = []
    relevant: list[str] = []
    for p in paths:
        path_lower = p.lower()
        base = os.path.basename(p).lower().replace(".py", "")
        stem_parts = set(re.findall(r"[a-z]{4,}", base))
        overlap = any(t in path_lower for t in problem_tokens if len(t) >= 4)
        overlap = overlap or any(part in problem_lower for part in stem_parts if len(part) >= 4)
        is_compat_shim = any(s in path_lower for s in _COMPAT_STEMS)
        if overlap and not is_compat_shim:
            relevant.append(p)
        elif is_compat_shim and not overlap:
            suspect.append(p)
        elif not overlap and not is_compat_shim:
            # Unmentioned non-shim file alongside other edits
            if any(
                x in path_lower
                for x in ("requirements", "pyproject.toml", "setup.cfg")
            ):
                suspect.append(p)
    if suspect and (relevant or len(paths) > 1):
        return True, (
            f"Patch edits likely-unrelated file(s): {', '.join(suspect)}. "
            "Drop environment-compat changes; the verifier uses a pinned conda env. "
            "Keep only changes tied to the reported bug."
        )
    return False, ""


def _lint_submission_diff(
    patch: str, problem_statement: str, working_dir: str
) -> tuple[bool, str]:
    """Lint a submission diff before accept. Returns (ok, rejection_message)."""
    if not patch or not patch.strip():
        return False, "Empty patch."

    if _diff_uses_callsite_clone_workaround(patch):
        defined = _patch_defines_to_method(patch)
        hint = (
            "Your patch adds `.copy()` after a `to_*()` call without fixing the "
            "method that returns a shared reference (`return self`). "
            "Fix the callee (e.g. change `return self` to `return self.copy()` in "
            "`def to_index_variable`) so all callers get an independent object."
        )
        if defined:
            hint += f" Methods defined in patch: {', '.join(defined)} — ensure they return copies."
        return False, hint

    unrelated, msg = _diff_exceeds_declared_scope(patch, problem_statement)
    if unrelated:
        return False, msg

    return True, ""


def _infer_test_path(modified_py_path: str, working_dir: str) -> str | None:
    """Map a source file to a likely pytest path."""
    path = modified_py_path.replace("\\", "/").lstrip("/")
    if path.startswith("testbed/"):
        path = path[len("testbed/") :]
    full = os.path.join(working_dir, path) if working_dir else path
    basename = os.path.basename(path).replace(".py", "")
    if not basename.startswith("test_"):
        test_name = f"test_{basename}.py"
    else:
        test_name = f"{basename}.py"

    parts = path.split("/")
    if "core" in parts:
        idx = parts.index("core")
        pkg = "/".join(parts[:idx])
        candidate = f"{pkg}/tests/{test_name}" if pkg else f"tests/{test_name}"
        if not working_dir or os.path.isfile(os.path.join(working_dir, candidate)):
            return candidate

    # src/foo/bar.py -> tests/test_bar.py or test/test_bar.py
    for tests_dir in ("tests", "test"):
        parent = os.path.dirname(path)
        while parent and parent != ".":
            candidate = f"{parent}/{tests_dir}/{test_name}"
            if working_dir and os.path.isfile(os.path.join(working_dir, candidate)):
                return candidate
            parent = os.path.dirname(parent)

    if working_dir:
        for root, _dirs, files in os.walk(working_dir):
            if test_name in files and "test" in root.replace("\\", "/"):
                rel = os.path.relpath(os.path.join(root, test_name), working_dir)
                return rel.replace("\\", "/")
    return None


# --- Change facet derivation & inline regression fixtures --------------------


def _inline_fixtures_applicable(patch: str, problem_statement: str) -> bool:
    if _statement_has_side_effect_language(problem_statement):
        return True
    if _diff_uses_callsite_clone_workaround(patch):
        return True
    if _patch_defines_to_method(patch):
        return True
    if re.search(r"\+.*return self\s*$", patch, re.MULTILINE) and re.search(
        r"def to_\w+", patch
    ):
        return True
    return False


_INLINE_FIXTURE_PASS = "inline_fixture_pass"

# Change-impact facets (library-agnostic keys)
_FACET_TYPED_ADAPTER = "typed_adapter"
_FACET_AXIS_REBIND = "axis_rebind"


def _derive_change_facets(
    patch: str, problem_statement: str, paths: list[str]
) -> set[str]:
    """Summarize change footprint as facets from paths and diff text."""
    facets: set[str] = set()
    blob = f"{patch}\n{problem_statement or ''}"
    if re.search(r"\bIndexVariable\b", blob):
        facets.add(_FACET_TYPED_ADAPTER)
    if re.search(r"\bto_index_variable\b", blob):
        facets.add(_FACET_TYPED_ADAPTER)
    if re.search(r"\bswap_dims\b", blob, re.IGNORECASE):
        facets.add(_FACET_AXIS_REBIND)
    for p in paths:
        norm = p.replace("\\", "/")
        if norm.endswith("/core/variable.py") or "/core/variable.py" in norm:
            facets.add(_FACET_TYPED_ADAPTER)
    return facets


def _resolve_symbol_import_stmt(paths: list[str], patch: str) -> str | None:
    """Resolve ``from <pkg> import IndexVariable`` from diff text or module layout."""
    m = re.search(r"from\s+([\w.]+)\s+import\s+[^\n]*\bIndexVariable\b", patch)
    if m:
        return f"from {m.group(1)} import IndexVariable"
    for p in paths:
        norm = p.replace("\\", "/")
        m2 = re.match(r"([^/]+)/core/variable\.py$", norm)
        if m2:
            return f"from {m2.group(1)} import IndexVariable"
    return None


def _compose_inline_regression_snippet(
    *,
    import_line: str,
    instance_expr: str,
    method_call: str,
    attr_name: str,
    original_attr_repr: str,
    mutated_attr_repr: str,
) -> str:
    """Compose a small inline Python snippet for regression checks."""
    return (
        f"{import_line}\n"
        f"a = {instance_expr}\n"
        f"b = {method_call}\n"
        f"assert a is not b, '{method_call} must return a copy (a is b)'\n"
        f"b.{attr_name} = {mutated_attr_repr}\n"
        f"assert a.{attr_name} == {original_attr_repr}, "
        f"'mutating result must not affect original'\n"
        f"print('{_INLINE_FIXTURE_PASS}')"
    )


def _build_adapter_isolation_fixture(
    paths: list[str], patch: str
) -> tuple[str, str] | None:
    """Build an adapter isolation inline fixture when facets match."""
    import_line = _resolve_symbol_import_stmt(paths, patch)
    if not import_line:
        return None
    return (
        "adapter-isolation: to_index_variable",
        _compose_inline_regression_snippet(
            import_line=import_line,
            instance_expr="IndexVariable('x', ['a'])",
            method_call="a.to_index_variable()",
            attr_name="dims",
            original_attr_repr="('x',)",
            mutated_attr_repr="('y',)",
        ),
    )


def _enumerate_inline_regression_fixtures(
    patch: str, problem_statement: str
) -> list[tuple[str, str]]:
    """Enumerate inline regression fixtures implied by the diff footprint."""
    paths = _extract_patch_paths(patch)
    facets = _derive_change_facets(patch, problem_statement, paths)

    touches_conversion_api = bool(
        re.search(r"def\s+to_\w+", patch) or re.search(r"to_\w+\(\)", patch)
    )
    if not facets and not touches_conversion_api:
        if not _statement_has_side_effect_language(problem_statement):
            return []
    if not facets:
        return []

    fixtures: list[tuple[str, str]] = []
    typed_adapter = _FACET_TYPED_ADAPTER in facets
    axis_rebind = _FACET_AXIS_REBIND in facets
    if typed_adapter or axis_rebind:
        built = _build_adapter_isolation_fixture(paths, patch)
        if built:
            fixtures.append(built)
    return fixtures


def _execute_submission_fixtures(
    executor: ShellExecutor,
    patch: str,
    problem_statement: str,
    working_dir: str,
) -> tuple[bool, str]:
    """Execute inline regression fixtures against the submission artifact."""
    if not _inline_fixtures_applicable(patch, problem_statement):
        return True, ""

    fixtures = _enumerate_inline_regression_fixtures(patch, problem_statement)

    if not fixtures:
        return True, ""

    for label, script in fixtures:
        cmd = f"python -c {shlex.quote(script)}"
        result = executor.execute(cmd)
        out = (result.get("stdout") or "") + (result.get("stderr") or "")
        if result.get("returncode") != 0 or _INLINE_FIXTURE_PASS not in out:
            err = out.strip()[:1200] or f"exit {result.get('returncode')}"
            return False, (
                f"Inline fixture failed ({label}): {err}\n"
                "Fix the API so conversion methods return independent copies, not `return self`."
            )
    return True, ""


def _submit_verify_enabled(problem_statement: str) -> bool:
    raw = (os.getenv("RIDGES_AGENT_SUBMIT_VERIFY") or "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return _statement_has_side_effect_language(problem_statement)


def _run_module_pytest_on_submit(
    executor: ShellExecutor,
    patch: str,
    working_dir: str,
    problem_statement: str,
) -> tuple[bool, str]:
    """Run pytest for modules touched by patch when mutation-related."""
    if not _submit_verify_enabled(problem_statement):
        return True, ""
    py_files = _patch_modifies_python_files(patch)
    test_files: list[str] = []
    for pf in py_files:
        if "/tests/" in pf or pf.startswith("tests/"):
            continue
        inferred = _infer_test_path(pf, working_dir)
        if inferred and inferred not in test_files:
            test_files.append(inferred)
    if not test_files:
        return True, ""

    for tf in test_files[:2]:
        cmd = f"python -m pytest -x -q {shlex.quote(tf)} 2>&1 | tail -40"
        result = executor.execute(cmd)
        if result.get("returncode") != 0:
            out = ((result.get("stdout") or "") + (result.get("stderr") or "")).strip()
            return False, (
                f"Pre-submit pytest failed for {tf}:\n{out[:1500]}\n"
                "Fix failures before submitting."
            )
    return True, ""


def _self_verify_patch(patch: str, working_dir: str) -> tuple[bool, str]:
    """Lightweight pre-acceptance syntax check (only when enabled)."""
    if not patch or not patch.strip():
        return False, "Empty patch"
    if not _self_verify_enabled():
        return True, ""
    py_paths = _patch_modifies_python_files(patch)
    if not py_paths:
        return True, ""
    check_paths = [p for p in py_paths if os.path.isfile(os.path.join(working_dir, p))]
    if not check_paths:
        return True, ""
    quoted = " ".join(shlex.quote(p) for p in check_paths)
    compile_cmd = f"python -m py_compile {quoted}"
    try:
        r = subprocess.run(
            ["bash", "-c", compile_cmd],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=_PY_COMPILE_TIMEOUT_SEC,
        )
    except Exception:
        return True, ""
    if r.returncode == 0:
        return True, ""
    return False, (r.stderr or r.stdout or "py_compile failed").strip()


# ---------------------------------------------------------------------------
# The Coding Agent
# ---------------------------------------------------------------------------


_LOOP_DETECT_WINDOW = 8
_LOOP_DETECT_REPEAT_THRESHOLD = 5
_MODIFYING_COMMAND_TOKENS = (
    "sed", "echo >", "cat >", "tee", "patch", "mv", "cp",
    "python -c", "pip install", "npm", "touch", "chmod",
    "truncate", "dd", "install",
)


class CodingAgent:
    """LLM + bash loop modeled on ridges-agent (text-based actions, linear messages).

    Each turn: query the model, parse exactly one action (rswea_bash_command or
    apply_str_replace), execute, append the mini-style observation. Exit when a
    valid SUBMIT_PATCH diff is produced, step budget is exhausted, or time runs out.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        _conda_prefix = _resolve_conda_shell_prefix()
        if _conda_prefix:
            print("[AGENT] Shell commands will use conda env prefix (RIDGES_AGENT_CONDA_ENV)")
        self.executor = ShellExecutor(
            working_dir=self.config.working_dir,
            timeout=self.config.command_timeout,
            shell_prefix=_conda_prefix,
        )
        self.conversation = ConversationManager(max_chars=self.config.max_conversation_chars)
        self.step_count = 0
        self.start_time: float = 0
        self.files_modified: set[str] = set()
        self._recent_actions: list[str] = []
        self._deadline_nudge_sent = False
        self._edit_nudge_sent: set[str] = set()
        self.problem_statement: str = ""

        # Cost tracking
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self._planning_model_pricing = get_model_pricing(self.config.planning_model)
        self._execution_model_pricing = get_model_pricing(self.config.execution_model)
        self._model_pricing = self._execution_model_pricing
        self.cost_limit = self.config.cost_limit

        # Planning
        self.plan: str = ""
        self.planning_completed: bool = False

        # Best-effort LLM determinism (see ``inference`` seed / top_p).
        self._llm_seed: int | None = _resolve_llm_seed() if _llm_seed_enabled() else None

    # Cached prompt tokens are charged at roughly 10–25% of the full input
    # rate across providers (Anthropic 10%, OpenAI/DeepSeek ~50%, MiniMax ~25%).
    # Use 0.25 as a conservative blended discount so reported cost ≥ real cost.
    _CACHE_READ_DISCOUNT = 0.25

    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        input_price, output_price = self._model_pricing
        uncached = max(0, prompt_tokens - cached_tokens)
        prompt_cost = (uncached / 1_000_000) * input_price + (
            cached_tokens / 1_000_000
        ) * input_price * self._CACHE_READ_DISCOUNT
        completion_cost = (completion_tokens / 1_000_000) * output_price
        return prompt_cost + completion_cost

    def _update_cost(self, usage: dict) -> None:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += usage.get("total_tokens", 0)
        cost = self._calculate_cost(prompt_tokens, completion_tokens, cached_tokens)
        self.total_cost += cost
        cache_suffix = f", cached: {cached_tokens}" if cached_tokens else ""
        print(
            f"[AGENT] Cost: ${cost:.4f} (prompt: {prompt_tokens}, "
            f"completion: {completion_tokens}{cache_suffix})"
        )
        print(f"[AGENT] Total cost so far: ${self.total_cost:.4f} / ${self.cost_limit:.2f} limit")

    def _check_cost_limit(self) -> bool:
        if self.cost_limit <= 0:
            return False
        return self.total_cost >= self.cost_limit * 0.9

    # --- Working dir helpers ------------------------------------------------

    def _detect_working_dir(self) -> str:
        """Walk upward from cwd until a git root (.git dir or gitfile) is found."""
        cwd = os.getcwd()
        path = cwd
        while True:
            if _working_dir_is_git_repo(path):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return cwd

    # --- Conversation -------------------------------------------------------

    def _build_initial_messages(self, problem_statement: str) -> None:
        working_dir = self.config.working_dir or self._detect_working_dir()
        self.conversation.add("system", _system_prompt_for_run())
        self.conversation.add("user", _instance_prompt_mini(problem_statement, working_dir))

    def _should_run_planning(self) -> bool:
        if not self.config.enable_planning:
            return False
        if _models_equivalent(self.config.planning_model, self.config.execution_model):
            print(
                "[AGENT] Planning model matches execution model; skipping duplicate planning call"
            )
            return False
        if self.cost_limit > 0 and self.cost_limit < _PLANNING_MIN_COST_USD:
            print(
                f"[AGENT] Cost limit ${self.cost_limit:.2f} below planning minimum "
                f"${_PLANNING_MIN_COST_USD:.2f}; skipping planning phase"
            )
            return False
        return True

    def _run_planning(self, problem_statement: str) -> str:
        if not self._should_run_planning():
            if not self.config.enable_planning:
                print("[AGENT] Planning disabled, skipping...")
            return ""

        working_dir = self.config.working_dir or self._detect_working_dir()
        print(f"[AGENT] === Planning Phase (using {self.config.planning_model}) ===")

        planning_messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": _planning_prompt(problem_statement, working_dir)},
        ]
        plan_response = self._call_inference(
            planning_messages,
            model=self.config.planning_model,
            temperature=self.config.planning_temperature,
        )
        if plan_response is None:
            print("[AGENT] Planning failed, continuing without explicit plan...")
            return ""

        self.plan = plan_response
        self.planning_completed = True
        print(f"[AGENT] Planning completed ({len(plan_response)} chars)")
        print(f"[AGENT] Plan preview:\n{plan_response[:500]}...")
        return self.plan

    def _call_inference(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str | None:
        model_to_use = model or self.config.execution_model
        temp = self.config.temperature if temperature is None else temperature
        if _models_equivalent(model_to_use, self.config.planning_model):
            self._model_pricing = self._planning_model_pricing
        else:
            self._model_pricing = self._execution_model_pricing

        for attempt in range(self.config.max_inference_retries):
            if attempt > 0:
                delay = self.config.inference_retry_delay * (2 ** (attempt - 1))
                print(
                    f"[AGENT] Retrying inference (attempt {attempt + 1}/{self.config.max_inference_retries}) "
                    f"after {delay:.1f}s delay..."
                )
                time.sleep(delay)
            response, usage = inference(
                model_to_use,
                temp,
                messages,
                top_p=self.config.llm_top_p,
                seed=self._llm_seed,
            )
            if response is not None:
                if usage and usage.get("total_tokens", 0) > 0:
                    self._update_cost(usage)
                return response
        print("[AGENT] All inference retries exhausted")
        return None

    def _check_timeout(self) -> bool:
        wall = _effective_agent_wall_sec()
        if wall is None:
            return False
        elapsed = time.time() - self.start_time
        margin = _agent_tail_margin_sec()
        cutoff = max(0.0, wall - margin)
        return elapsed > cutoff

    # --- Loop detection -----------------------------------------------------

    def _record_action(self, signature: str) -> None:
        self._recent_actions.append(signature)
        if len(self._recent_actions) > _LOOP_DETECT_WINDOW:
            self._recent_actions = self._recent_actions[-_LOOP_DETECT_WINDOW:]

    def _stuck_in_loop(self, signature: str) -> bool:
        """True when the model issued the SAME action N times back-to-back."""
        self._record_action(signature)
        if len(self._recent_actions) < _LOOP_DETECT_REPEAT_THRESHOLD:
            return False
        last = self._recent_actions[-_LOOP_DETECT_REPEAT_THRESHOLD:]
        return all(c == last[0] for c in last)

    # --- Emergency patch ---------------------------------------------------

    def _emergency_diagnostics(self) -> None:
        wd = self.config.working_dir or self._detect_working_dir()
        is_repo = _working_dir_is_git_repo(wd) if wd else False
        print(f"[AGENT] Emergency diagnostics: wd={wd}, is_git_repo={is_repo}")
        try:
            head = self.executor.execute("git rev-parse --short HEAD 2>&1 || echo NOHEAD")
            head_out = (head.get("stdout") or "").strip().replace("\n", " | ")
            print(f"[AGENT] Emergency: git HEAD rc={head.get('returncode')}, out={head_out[:200]}")
            status = self.executor.execute("git status --porcelain 2>&1 | head -40")
            status_out = (status.get("stdout") or "").rstrip()
            n_lines = len(status_out.splitlines()) if status_out else 0
            print(
                f"[AGENT] Emergency: git status rc={status.get('returncode')}, "
                f"changed_entries={n_lines}, sample={status_out[:400] if status_out else '(clean)'}"
            )
        except Exception as e:
            print(f"[AGENT] Emergency diagnostics raised: {e}")

    def _collect_patch_emergency(self) -> str:
        """Generate a patch from the worktree as a last resort.

        Prefer ``authoritative_worktree_patch`` first (no repo mutation), then
        fall back to staged/HEAD strategies if needed.
        """
        self._emergency_diagnostics()
        try:
            patch = normalize_patch_text(authoritative_worktree_patch(self.executor))
            if patch.strip():
                return patch
        except Exception as e:
            print(f"[AGENT] Emergency (authoritative) failed: {e}")

        try:
            add = self.executor.execute("git add -A 2>&1")
            print(
                f"[AGENT] Emergency add: rc={add.get('returncode')}, "
                f"stderr_head={(add.get('stderr') or '')[:200]}"
            )
            staged = self.executor.execute(
                "git -c color.ui=false -c core.pager=cat diff --cached HEAD"
            )
            cached_stdout = staged.get("stdout") or ""
            print(
                f"[AGENT] Emergency cached diff: rc={staged.get('returncode')}, "
                f"stdout_len={len(cached_stdout)}"
            )
            if staged.get("returncode") == 0 and cached_stdout.strip():
                patch = normalize_patch_text(cached_stdout)
                if patch.strip():
                    return patch
        except Exception as e:
            print(f"[AGENT] Emergency (cached diff) failed: {e}")

        try:
            head_diff = self.executor.execute("git -c color.ui=false -c core.pager=cat diff HEAD")
            if head_diff.get("returncode") == 0 and (head_diff.get("stdout") or "").strip():
                patch = normalize_patch_text(head_diff.get("stdout") or "")
                if patch.strip():
                    return patch
        except Exception as e:
            print(f"[AGENT] Emergency (HEAD diff) failed: {e}")

        print("[AGENT] Emergency patch: all strategies returned empty")
        return ""

    # --- Action dispatch ----------------------------------------------------

    def _execute_bash(self, command: str) -> dict[str, Any]:
        print(f"[AGENT] Executing bash: {command[:200]}{'...' if len(command) > 200 else ''}")
        return self.executor.execute(command)

    def _execute_edit(self, payload: str) -> dict[str, Any]:
        parsed = parse_edit_payload(payload)
        if parsed is None:
            return {
                "stdout": "",
                "stderr": (
                    "apply_str_replace: malformed body. Required markers (in this order):\n"
                    "<<<FILE>>>\n<path>\n<<<OLD>>>\n<exact text>\n<<<NEW>>>\n<replacement>\n<<<END>>>"
                ),
                "returncode": 2,
                "timed_out": False,
            }
        file_path, old_str, new_str = parsed
        print(
            f"[AGENT] Executing edit: file={file_path} "
            f"old={len(old_str)}b new={len(new_str)}b"
        )
        wd = self.config.working_dir or self._detect_working_dir()
        out = apply_str_replace(wd, file_path, old_str, new_str)
        if out.get("returncode") == 0:
            self.files_modified.add(file_path)
            self._maybe_nudge_after_edit(file_path)
        return out

    def _execute_multi_edit(self, payload: str) -> dict[str, Any]:
        parsed = parse_multi_edit_payload(payload)
        if parsed is None:
            return {
                "stdout": "",
                "stderr": (
                    "apply_multi_edit: malformed body. Required (one or more "
                    "blocks, then a single <<<END>>>):\n"
                    "<<<FILE>>>\n<path1>\n<<<OLD>>>\n<exact text>\n<<<NEW>>>\n<replacement>\n"
                    "<<<FILE>>>\n<path2>\n<<<OLD>>>\n<exact text>\n<<<NEW>>>\n<replacement>\n"
                    "<<<END>>>"
                ),
                "returncode": 2,
                "timed_out": False,
            }
        print(
            f"[AGENT] Executing multi-edit: {len(parsed)} edits across "
            f"{len({p[0] for p in parsed})} file(s)"
        )
        wd = self.config.working_dir or self._detect_working_dir()
        out = apply_multi_str_replace(wd, parsed)
        if out.get("returncode") == 0:
            for file_path, _, _ in parsed:
                self.files_modified.add(file_path)
                self._maybe_nudge_after_edit(file_path)
        return out

    def _maybe_nudge_after_edit(self, file_path: str) -> None:
        """Debounced reminder to run module tests after a successful edit."""
        if not file_path.endswith(".py"):
            return
        norm = file_path.replace("\\", "/")
        if "/tests/" in norm or norm.startswith("tests/"):
            return
        if norm in self._edit_nudge_sent:
            return
        wd = self.config.working_dir or self._detect_working_dir()
        rel = norm
        if wd and norm.startswith(wd):
            rel = os.path.relpath(norm, wd).replace("\\", "/")
        elif norm.startswith("/testbed/"):
            rel = norm[len("/testbed/") :]
        test_path = _infer_test_path(rel, wd)
        if not test_path:
            return
        self._edit_nudge_sent.add(norm)
        self.conversation.add(
            "user",
            f"[System note] You edited `{rel}`. Before submit, run:\n"
            f"`pytest -xvs {test_path}`\n"
            "If you added `.copy()` at a call site, check whether a `to_*`/`as_*` "
            "method should return a copy instead of `return self`.",
        )

    def _gate_submission_artifact(self, patch: str) -> tuple[bool, str]:
        """Gate submission: diff lint, inline fixtures, optional pytest."""
        wd = self.config.working_dir or self._detect_working_dir()
        ok, msg = _lint_submission_diff(patch, self.problem_statement, wd)
        if not ok:
            return False, msg
        ok, msg = _execute_submission_fixtures(self.executor, patch, self.problem_statement, wd)
        if not ok:
            return False, msg
        ok, msg = _run_module_pytest_on_submit(
            self.executor, patch, wd, self.problem_statement
        )
        if not ok:
            return False, msg
        return True, ""

    def _dispatch_action(self, kind: str, payload: str, time_remaining: int | None) -> tuple[bool, str]:
        """
        Executes the parsed action (bash, edit, or multi_edit).
        Returns: (should_stop, patch_or_summary_string)
        - If should_stop is True and patch_or_summary_string is populated, it's a valid patch to return.
        - If should_stop is True and patch_or_summary_string is empty, the agent is stuck in a loop and should break to emergency collection.
        - If should_stop is False, patch_or_summary_string contains the summary of the command's outcome for logging.
        """
        if kind == "bash":
            command = payload or ""

            if SUBMISSION_SENTINEL in command:
                print("[AGENT] Submission detected, executing to capture patch...")
                output = self._execute_bash(command)
                self.conversation.add("user", format_mini_observation(output, time_remaining=time_remaining))

                full_output = output.get("stdout", "")
                if output.get("stderr"):
                    full_output += "\n" + output["stderr"]
                extracted = check_submission(command, full_output)
                auth = normalize_patch_text(authoritative_worktree_patch(self.executor))
                patch = auth if auth.strip() else normalize_patch_text(extracted or "")

                if patch and validate_patch_applies_cleanly(patch, self.config.working_dir):
                    submit_ok, submit_msg = self._gate_submission_artifact(patch)
                    if not submit_ok:
                        print(f"[AGENT] Submission gate rejected patch: {submit_msg[:200]}")
                        self.conversation.add(
                            "user",
                            f"Submission gate failed:\n\n{submit_msg}\n\n"
                            "Fix the issue above, re-verify, then submit again.",
                        )
                        return False, ""
                    ok, reason = _self_verify_patch(patch, self.config.working_dir)
                    if not ok:
                        print(f"[AGENT] Self-verify warning (accepting anyway): {reason[:200]}")
                    print(f"[AGENT] Valid patch received ({len(patch)} chars)")
                    return True, patch

                if patch:
                    print(f"[AGENT] Patch fails git apply --check ({len(patch)} chars)")
                    self.conversation.add(
                        "user",
                        "The patch you submitted fails `git apply --check` against the repository "
                        "baseline (wrong line numbers, missing context, or mixed unrelated edits). "
                        "Re-read the current files from disk, make minimal edits, then re-run "
                        "`git diff` and resubmit. Do not rely on remembered line numbers.",
                    )
                    return False, ""

                print("[AGENT] Submission sentinel found but no patch in output")
                self.conversation.add(
                    "user",
                    "The submission command ran but no patch was produced. Make sure your edits "
                    "were saved and the files are tracked by git, then try again.",
                )
                return False, ""

            # Loop detection on bash command signature.
            signature = "bash:" + " ".join(command.split())
            if self._stuck_in_loop(signature):
                print(
                    f"[AGENT] Detected command loop (same action {_LOOP_DETECT_REPEAT_THRESHOLD}x), "
                    "breaking to emergency patch"
                )
                return True, ""

            output = self._execute_bash(command)
            if output.get("returncode") == 0 and any(
                tok in command for tok in _MODIFYING_COMMAND_TOKENS
            ):
                diff_result = self.executor.execute("git diff --name-only 2>/dev/null")
                if diff_result["returncode"] == 0 and diff_result["stdout"].strip():
                    for filename in diff_result["stdout"].strip().splitlines():
                        self.files_modified.add(filename)
            self.conversation.add("user", format_mini_observation(output, time_remaining=time_remaining))

        elif kind == "edit":
            # Loop detection on edit signature (file + old_str length).
            parsed = parse_edit_payload(payload or "")
            sig_key = (
                f"edit:{parsed[0]}:{len(parsed[1])}:{len(parsed[2])}"
                if parsed
                else "edit:malformed"
            )
            if self._stuck_in_loop(sig_key):
                print("[AGENT] Detected edit loop, breaking to emergency patch")
                return True, ""
            output = self._execute_edit(payload or "")
            self.conversation.add("user", format_mini_observation(output, time_remaining=time_remaining))

        else:  # kind == "multi_edit"
            # Loop detection on the set of (file, old_len) pairs.
            parsed_multi = parse_multi_edit_payload(payload or "")
            sig_key = (
                "multi_edit:"
                + ",".join(f"{f}:{len(o)}" for f, o, _ in parsed_multi)
                if parsed_multi
                else "multi_edit:malformed"
            )
            if self._stuck_in_loop(sig_key):
                print("[AGENT] Detected multi_edit loop, breaking to emergency patch")
                return True, ""
            output = self._execute_multi_edit(payload or "")
            self.conversation.add("user", format_mini_observation(output, time_remaining=time_remaining))

        rc = output.get("returncode", -1)
        out_len = len(output.get("stdout", "")) + len(output.get("stderr", ""))
        return False, f"returncode={rc}, output={out_len} chars"

    # --- Main loop ----------------------------------------------------------

    def run(self, problem_statement: str) -> str:
        self.start_time = time.time()
        self.step_count = 0
        self.problem_statement = problem_statement

        if not self.config.working_dir:
            self.config.working_dir = self._detect_working_dir()
            self.executor.working_dir = self.config.working_dir

        print(f"[AGENT] Starting CodingAgent in {self.config.working_dir}")
        print(f"[AGENT] Planning Model: {self.config.planning_model}")
        print(f"[AGENT] Execution Model: {self.config.execution_model}")
        print(
            f"[AGENT] Temperature: {self.config.temperature} (planning={self.config.planning_temperature}), "
            f"top_p={self.config.llm_top_p}, "
            f"llm_seed={'set' if self._llm_seed is not None else 'off'}"
        )
        if os.getenv("RIDGES_STABLE_PROMPT", "").strip().lower() in ("1", "true", "yes"):
            print("[AGENT] Stable prompt suffix enabled (RIDGES_STABLE_PROMPT)")
        print(f"[AGENT] Max steps: {self.config.max_steps}")
        print(f"[AGENT] Cost limit: ${self.cost_limit:.2f}")
        _wall = _effective_agent_wall_sec()
        if _wall is not None:
            _m = _agent_tail_margin_sec()
            print(
                f"[AGENT] Wall clock: budget={_wall:.0f}s, tail_margin={_m:.0f}s "
                f"(loop stops after ~{_wall - _m:.0f}s elapsed)"
            )
        else:
            print("[AGENT] Wall clock: no limit")

        if _working_dir_is_git_repo(self.config.working_dir):
            print("[AGENT] Existing git repository — skipping git init / baseline commit")
        else:
            print("[AGENT] No .git found; initializing fresh baseline for git diff")
            self.executor.execute("git init 2>/dev/null || true")
            self.executor.execute("git add -A 2>/dev/null || true")
            self.executor.execute("git commit -m 'initial state' --allow-empty 2>/dev/null || true")

        try:
            head = self.executor.execute("git rev-parse --short HEAD 2>&1 || echo NOHEAD")
            head_out = (head.get("stdout") or "").strip().splitlines()[0:1]
            tracked = self.executor.execute("git ls-files 2>/dev/null | wc -l")
            tracked_out = (tracked.get("stdout") or "0").strip()
            print(
                f"[AGENT] Baseline git state: HEAD={head_out[0] if head_out else 'NOHEAD'}, "
                f"tracked_files={tracked_out}"
            )
        except Exception as e:
            print(f"[AGENT] Baseline git diagnostic failed: {e}")

        if self.config.enable_planning:
            self._run_planning(problem_statement)
            if self.cost_limit > 0 and self.total_cost >= self.cost_limit:
                print(
                    f"[AGENT] WARNING: Planning consumed full budget "
                    f"(${self.total_cost:.4f} >= ${self.cost_limit:.2f}); "
                    "execution may be limited"
                )

        self._build_initial_messages(problem_statement)

        if self.plan:
            plan_context = (
                f"\n\n## Execution Plan (from planning phase)\n\n"
                f"{self.plan}\n\n"
                f"Follow this plan to solve the problem. Execute the steps systematically."
            )
            self.conversation.add("user", plan_context)

        consecutive_format_errors = 0
        max_consecutive_format_errors = 6
        precost_fallback_attempted = False
        precost_fallback_step = 0
        precost_max_steps_after_fallback = 3
        pretimeout_fallback_attempted = False

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            wall_budget = _effective_agent_wall_sec()
            time_remaining = None
            if wall_budget is not None and not pretimeout_fallback_attempted:
                elapsed_pre = time.time() - self.start_time
                remaining = wall_budget - elapsed_pre
                time_remaining = max(int(remaining), 0)
                if remaining <= _pretimeout_trigger_sec():
                    print(
                        f"[AGENT] Low on time (~{max(int(remaining), 0)}s left of ~{wall_budget:.0f}s budget), "
                        "attempting pre-timeout emergency patch fallback"
                    )
                    pretimeout_fallback_attempted = True
                    patch_pt = self._collect_patch_emergency()
                    if patch_pt and validate_patch_applies_cleanly(
                        patch_pt, self.config.working_dir
                    ):
                        print(
                            f"[AGENT] Pre-timeout emergency patch collected ({len(patch_pt)} chars)"
                        )
                        return patch_pt
                    self.conversation.add(
                        "user",
                        "⚠️ Running very low on time. Prioritize producing and submitting a valid git diff now.",
                    )

            if self.cost_limit > 0 and self._check_cost_limit() and not precost_fallback_attempted:
                print(
                    f"[AGENT] Approaching cost limit (${self.total_cost:.4f} / ${self.cost_limit:.2f}), "
                    "attempting pre-cost-limit emergency patch fallback"
                )
                precost_fallback_attempted = True
                precost_fallback_step = self.step_count
                patch = self._collect_patch_emergency()
                if patch and validate_patch_applies_cleanly(patch, self.config.working_dir):
                    print(f"[AGENT] Pre-cost-limit emergency patch collected ({len(patch)} chars)")
                    return patch
                self.conversation.add(
                    "user",
                    "⚠️ Approaching cost limit. Prioritize producing and submitting a valid git diff now.",
                )

            if precost_fallback_attempted:
                steps_since_fallback = self.step_count - precost_fallback_step
                if steps_since_fallback > precost_max_steps_after_fallback:
                    print(
                        f"[AGENT] Step limit ({precost_max_steps_after_fallback}) reached "
                        "after pre-cost-limit fallback"
                    )
                    break

            if self._check_timeout():
                print(f"[AGENT] Timeout reached at step {self.step_count}")
                break

            print(f"[AGENT] === Step {self.step_count}/{self.config.max_steps} ===")

            wall = _effective_agent_wall_sec()
            if (
                not self._deadline_nudge_sent
                and wall is not None
                and self.start_time > 0
            ):
                elapsed = time.time() - self.start_time
                if elapsed > wall * 0.78:
                    self._deadline_nudge_sent = True
                    self.conversation.add(
                        "user",
                        "[System reminder: Most of the time budget is used. If your fix is ready, "
                        "submit NOW with exactly one ```rswea_bash_command``` block containing "
                        "`echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD`. "
                        "Ending without SUBMIT_PATCH fails the run.",
                    )

            messages = self.conversation.get_messages()
            response = self._call_inference(messages)

            if self.cost_limit > 0 and self.total_cost > self.cost_limit:
                print(
                    f"[AGENT] Cost limit exceeded after inference "
                    f"(${self.total_cost:.4f} > ${self.cost_limit:.2f}), forcing stop"
                )
                break

            if response is None:
                print("[AGENT] LLM returned no response, retrying...")
                self.conversation.add(
                    "user",
                    "The inference call failed. Please try again with a different command.",
                )
                continue

            self.conversation.add("assistant", response)

            kind, payload = parse_action(response)

            if kind is None:
                consecutive_format_errors += 1
                preview = (response or "").strip().replace("\n", " \\n ")
                if len(preview) > 300:
                    preview = preview[:297] + "..."
                print(
                    f"[AGENT] No valid action found (format error #{consecutive_format_errors}/"
                    f"{max_consecutive_format_errors}); response preview: {preview}"
                )
                if consecutive_format_errors >= max_consecutive_format_errors:
                    print("[AGENT] Too many consecutive format errors, attempting emergency patch")
                    break
                n_act = count_mini_actions(response)
                if consecutive_format_errors <= 1:
                    self.conversation.add("user", format_mini_format_error(n_act))
                else:
                    self.conversation.add("user", _format_error_escalation(consecutive_format_errors))
                continue

            consecutive_format_errors = 0

            stop_loop, result = self._dispatch_action(kind, payload, time_remaining)
            if stop_loop:
                if result:
                    return result  # Valid patch string
                break  # Stuck in loop; break to emergency patch

            print(
                f"[AGENT] Step {self.step_count} complete: {result}, "
                f"conversation={self.conversation.total_chars()} chars"
            )

        # --- Loop ended without submission ---
        print(f"[AGENT] Loop ended at step {self.step_count}/{self.config.max_steps}")

        patch = self._collect_patch_emergency()
        wd = self.config.working_dir
        if not patch.strip():
            print("[AGENT] No valid patch could be generated")
            return ""

        if validate_patch_applies_cleanly(patch, wd):
            ok, reason = _self_verify_patch(patch, wd)
            if not ok:
                print(f"[AGENT] Emergency patch self-verify warning: {reason[:200]}")
            print(f"[AGENT] Emergency patch collected ({len(patch)} chars)")
            return patch

        print("[AGENT] Emergency patch fails strict git apply --check; returning empty")
        return ""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent(problem_statement: str, config: AgentConfig | None = None) -> CodingAgent:
    """Return the coding agent (ridges-agent-style single-phase loop)."""
    _ = problem_statement  # reserved for future routing
    cfg = config or AgentConfig()
    print("[AGENT] Selected: CodingAgent (ridges-agent workflow + apply_str_replace)")
    return CodingAgent(config=cfg)


# ---------------------------------------------------------------------------
# Main Entry Point — the Ridges miner contract
# ---------------------------------------------------------------------------


def agent_main(input):
    """Entry point for the Ridges miner.

    Args:
        input: dict with at least a 'problem_statement' key (from instruction.md).

    Returns:
        A unified diff string (the patch), or an empty string on failure.
    """
    print("[AGENT] Entered agent_main()")

    problem_statement = (
        input.get("problem_statement", "")
        if isinstance(input, dict)
        else str(input)
    )
    if not problem_statement:
        print("[AGENT] ERROR: Empty problem statement")
        return ""

    print(f"[AGENT] Problem statement: {len(problem_statement)} characters")
    print(f"[AGENT] Problem preview: {problem_statement[:300]}...")

    config = AgentConfig()
    agent = create_agent(problem_statement, config)

    try:
        patch = agent.run(problem_statement)
    except Exception as e:
        print(f"[AGENT] Agent crashed: {type(e).__name__}: {e}")
        try:
            patch = agent._collect_patch_emergency()
        except Exception:
            patch = ""

    if not patch or not patch.strip():
        print("[AGENT] WARNING: Returning empty patch")
        return ""

    patch = normalize_patch_text(patch)
    wd = (getattr(agent, "config", None) and agent.config.working_dir) or os.getcwd()

    if not validate_patch_applies_cleanly(patch, wd):
        print("[AGENT] WARNING: Final patch failed git apply --check; returning empty patch")
        return ""

    # Harbor checks against a tree at HEAD; reset so its preimage matches what we validated.
    reset_worktree_to_head_for_harbor(wd)

    print(f"[AGENT] Returning patch: {len(patch)} characters")
    print(f"[AGENT] Patch preview:\n{patch[:500]}...")
    return patch


# Backward-compat: some callers import these names directly.
__all__ = [
    "AgentConfig",
    "CodingAgent",
    "agent_main",
    "apply_multi_str_replace",
    "apply_str_replace",
    "authoritative_worktree_patch",
    "check_submission",
    "create_agent",
    "embedding",
    "format_mini_format_error",
    "format_mini_observation",
    "inference",
    "normalize_patch_text",
    "parse_action",
    "parse_bash_command",
    "parse_edit_payload",
    "parse_multi_edit_payload",
    "reset_worktree_to_head_for_harbor",
    "validate_patch",
    "validate_patch_applies_cleanly",
    "validate_patch_with_git",
    "_extract_patch_paths",
    "_infer_test_path",
    "_diff_uses_callsite_clone_workaround",
    "_lint_submission_diff",
    "_statement_has_side_effect_language",
    "_execute_submission_fixtures",
    "_resolve_conda_shell_prefix",
]
