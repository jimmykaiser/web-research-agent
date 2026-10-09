# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.plugins import BasePlugin
from google.adk.tools import FunctionTool, preload_memory
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.workflow import Workflow, node
from google.genai import types
from opentelemetry import trace
from pydantic import BaseModel, Field



# Strategic model routing: lightweight model for fast classification/planning,
# higher-capacity model for multi-tool research synthesis.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gemini-2.5-flash-lite")
RESEARCHER_MODEL = os.getenv(
    "RESEARCHER_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
)

logger = logging.getLogger("web_research_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

tracer = trace.get_tracer("web_research_agent")
_LAST_STAGE_SPAN_CONTEXT: trace.SpanContext | None = None

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (
        re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(
            r"\b(?:AIza[0-9A-Za-z\-_]{20,}|sk-[0-9A-Za-z]{20,}|ya29\.[0-9A-Za-z\-_]+)\b"
        ),
        "[REDACTED_SECRET]",
    ),
]


def redact_pii(text: str) -> str:
    """Scrubs emails, SSNs, phone numbers, and API secrets from text before logging or LLM calls."""
    scrubbed = text
    for pattern, replacement in _PII_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    return value


def log_structured_event(
    event_type: str,
    *,
    intent: dict[str, Any],
    outcome: dict[str, Any],
    severity: str = "INFO",
    component: str = "web_research_agent",
) -> dict[str, Any]:
    """Emits a structured JSON log record capturing intent vs. outcome and active OTel trace IDs."""
    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context() if current_span else None
    trace_id = (
        f"{span_ctx.trace_id:032x}" if span_ctx and span_ctx.is_valid else None
    )
    span_id = (
        f"{span_ctx.span_id:016x}" if span_ctx and span_ctx.is_valid else None
    )

    record: dict[str, Any] = {
        "severity": severity,
        "event_type": event_type,
        "component": component,
        "intent": _sanitize_value(intent),
        "outcome": _sanitize_value(outcome),
        "trace_id": trace_id,
        "span_id": span_id,
    }
    logger.info(json.dumps(record, sort_keys=True))
    return record


@contextlib.contextmanager
def start_linked_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Iterator[trace.Span]:
    """Starts a custom OpenTelemetry span explicitly linked to the previous workflow stage span."""
    global _LAST_STAGE_SPAN_CONTEXT
    links: list[trace.Link] = []
    if _LAST_STAGE_SPAN_CONTEXT is not None and _LAST_STAGE_SPAN_CONTEXT.is_valid:
        links.append(trace.Link(_LAST_STAGE_SPAN_CONTEXT))

    with tracer.start_as_current_span(
        name, links=links, attributes=attributes or {}
    ) as span:
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.is_valid:
            _LAST_STAGE_SPAN_CONTEXT = span_ctx
        yield span


def scrub_pii_before_model_callback(
    callback_context: CallbackContext, llm_request: object
) -> None:
    """Redacts PII from outgoing LLM request contents before invoking the model."""
    del callback_context
    contents = getattr(llm_request, "contents", None) or []
    redacted_parts = 0
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                scrubbed = redact_pii(text)
                if scrubbed != text:
                    part.text = scrubbed
                    redacted_parts += 1
    if redacted_parts > 0:
        log_structured_event(
            "pii_redaction_guardrail",
            intent={"action": "scrub_llm_request_contents"},
            outcome={"status": "redacted", "redacted_parts": redacted_parts},
        )
    return None


async def add_session_to_memory_callback(callback_context: CallbackContext):
    """Persists session context to Vertex AI Memory Bank when available."""
    with start_linked_span("memory.persist_session"):
        try:
            await callback_context.add_session_to_memory()
            log_structured_event(
                "memory_persistence",
                intent={"action": "add_session_to_memory"},
                outcome={"status": "persisted"},
            )
        except ValueError:
            # Memory service is not configured in local in-memory test runs
            log_structured_event(
                "memory_persistence",
                intent={"action": "add_session_to_memory"},
                outcome={"status": "skipped_in_memory"},
            )
    return None


def normalize_grounding_metadata_callback(
    callback_context: CallbackContext, llm_response: object
) -> None:
    """Coerces dict grounding metadata in temp state back to GroundingMetadata.

    When GoogleSearchTool(bypass_multi_tools_limit=True) runs in parallel with
    other tools, ADK's merge_parallel_function_response_events dumps
    EventActions.state_delta via model_dump(), turning
    temp:_adk_grounding_metadata into a plain dict and causing
    VertexAiSessionService.append_event to fail on grounding_metadata.model_dump().
    """
    del llm_response
    raw = callback_context.state.get("temp:_adk_grounding_metadata")
    if isinstance(raw, dict):
        callback_context.state["temp:_adk_grounding_metadata"] = (
            types.GroundingMetadata.model_validate(raw)
        )
    return None


class ResearchPlan(BaseModel):
    """Classification and search plan for an incoming user query."""

    is_research_query: bool = Field(
        description=(
            "True if the query asks to research, explain, compare, or find information/news/papers "
            "about a topic, concept, technology, science, or event. "
            "False if it is casual chit-chat, a request to write/debug code from scratch without a "
            "research topic, or pure arithmetic."
        )
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2 to 3 targeted search queries to investigate the topic.",
    )
    reasoning: str = Field(
        description="Brief explanation of why the query is or is not a research inquiry."
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True ONLY if the user's research request is too vague or missing a subject "
            "(e.g., 'Research it' or 'Compare the two' with no named technologies) and requires "
            "human clarification before searching. False for any named topic or question."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask the user when needs_clarification is True.",
    )


def search_hackernews(query: str, max_results: int = 5) -> str:
    """Search Hacker News stories and developer discussions for a topic.

    Args:
        query: Search keywords (e.g. "retrieval augmented generation" or "WebAssembly").
        max_results: Maximum number of stories to return (1 to 10).

    Returns:
        Formatted string of matching Hacker News stories with points and URLs.
    """
    safe_query = redact_pii(query)
    limit = max(1, min(max_results, 10))
    with start_linked_span(
        "tool.search_hackernews",
        attributes={"tool.query": safe_query, "tool.max_results": limit},
    ) as span:
        encoded = urllib.parse.quote(safe_query)
        url = f"https://hn.algolia.com/api/v1/search?query={encoded}&tags=story&hitsPerPage={limit}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "adk-web-research-agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("hits", [])
        span.set_attribute("tool.hit_count", len(hits))
        log_structured_event(
            "tool_execution",
            intent={
                "tool": "search_hackernews",
                "query": safe_query,
                "max_results": limit,
            },
            outcome={
                "status": "ok" if hits else "empty",
                "hit_count": len(hits),
            },
        )
        if not hits:
            return f"No Hacker News stories found for: {safe_query}"
        lines = []
        for hit in hits:
            title = hit.get("title") or "Untitled"
            points = hit.get("points") or 0
            object_id = hit.get("objectID", "")
            story_url = (
                hit.get("url")
                or f"https://news.ycombinator.com/item?id={object_id}"
            )
            hn_url = f"https://news.ycombinator.com/item?id={object_id}"
            lines.append(
                f"- {title} ({points} points) | Article: {story_url} | HN Discussion: {hn_url}"
            )
        return "\n".join(lines)


def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for scientific and technical research papers.

    Args:
        query: Search terms for research papers (e.g. "mixture of experts language models").
        max_results: Maximum number of papers to return (1 to 5).

    Returns:
        Formatted string containing paper titles, publication dates, summaries, and arXiv URLs.
    """
    safe_query = redact_pii(query)
    limit = max(1, min(max_results, 5))
    with start_linked_span(
        "tool.search_arxiv",
        attributes={"tool.query": safe_query, "tool.max_results": limit},
    ) as span:
        encoded = urllib.parse.quote(safe_query)
        url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={limit}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "adk-web-research-agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        span.set_attribute("tool.hit_count", len(entries))
        log_structured_event(
            "tool_execution",
            intent={
                "tool": "search_arxiv",
                "query": safe_query,
                "max_results": limit,
            },
            outcome={
                "status": "ok" if entries else "empty",
                "hit_count": len(entries),
            },
        )
        if not entries:
            return f"No arXiv papers found for: {safe_query}"
        results = []
        for entry in entries:
            title = (
                (entry.findtext("atom:title", default="", namespaces=ns) or "")
                .strip()
                .replace("\n", " ")
            )
            summary = (
                (entry.findtext("atom:summary", default="", namespaces=ns) or "")
                .strip()
                .replace("\n", " ")
            )
            published = (
                entry.findtext("atom:published", default="", namespaces=ns) or ""
            )[:10]
            link = entry.findtext("atom:id", default="", namespaces=ns) or ""
            results.append(
                f"- [{published}] {title}\n  URL: {link}\n  Abstract: {summary[:350]}..."
            )
        return "\n".join(results)


def requires_exhaustive_search_approval(max_results: int = 3, **_kwargs: object) -> bool:
    """Requires human confirmation before running large literature sweeps (>5 papers)."""
    return max_results > 5


search_arxiv_tool = FunctionTool(
    search_arxiv,
    require_confirmation=requires_exhaustive_search_approval,
)


# 1. Planner & Classifier Agent (uses lightweight PLANNER_MODEL)
query_planner_agent = LlmAgent(
    name="query_planner_agent",
    model=Gemini(
        model=PLANNER_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a senior research planner. Analyze the user's prompt and determine whether "
        "it is a researchable topic or question.\n\n"
        "Research queries include:\n"
        "- Questions about concepts, science, technology, history, industry trends, or events\n"
        "- Requests to compare technologies, summarize academic papers, or gather community sentiment\n\n"
        "Non-research (unrelated) queries include:\n"
        "- Requests to write boilerplate code or solve pure math problems without a research topic\n"
        "- Casual greetings or small talk with no research topic\n\n"
        "If it is a research query, generate 2 to 3 concise search queries (2-4 keywords each) and output using the ResearchPlan schema. "
        "Set needs_clarification=True only when the research prompt is completely missing a subject (e.g. 'research this for me')."
    ),
    output_schema=ResearchPlan,
    before_model_callback=[scrub_pii_before_model_callback],
)


def route_research(ctx: Context, node_input: dict | ResearchPlan) -> Event | RequestInput:
    """Routes valid research plans, pausing for human-in-the-loop input when clarification or plan approval is required."""
    if isinstance(node_input, dict):
        is_research = node_input.get("is_research_query", False)
        queries = node_input.get("search_queries", [])
        needs_clarification = node_input.get("needs_clarification", False)
        clarification_question = node_input.get("clarification_question")
    else:
        is_research = getattr(node_input, "is_research_query", False)
        queries = getattr(node_input, "search_queries", [])
        needs_clarification = getattr(node_input, "needs_clarification", False)
        clarification_question = getattr(node_input, "clarification_question", None)

    route = "research" if is_research else "unrelated"

    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
        user_text = "".join(p.text for p in ctx.user_content.parts if p.text)
    safe_user_text = redact_pii(user_text)

    state = getattr(ctx, "state", None) or {}
    resume_inputs = getattr(ctx, "resume_inputs", None) or {}
    require_plan_approval = bool(state.get("require_plan_approval", False))

    with start_linked_span(
        "workflow.route_research",
        attributes={
            "research.intent.is_research": bool(is_research),
            "research.intent.query_count": len(queries),
            "research.intent.needs_clarification": bool(needs_clarification),
        },
    ) as span:
        # Programmatic HITL execution stop when a research query needs clarification
        # or explicit plan approval before executing multi-source searches.
        if route == "research" and (needs_clarification or require_plan_approval):
            if "clarify_research_scope" not in resume_inputs:
                planned_preview = ", ".join(queries) if queries else "none yet"
                prompt_msg = clarification_question or (
                    f"Please confirm or refine the planned search queries before execution: {planned_preview}"
                )
                span.set_attribute("research.outcome.decision", "hitl_pause")
                log_structured_event(
                    "workflow_routing",
                    intent={
                        "user_query": safe_user_text,
                        "is_research_query": bool(is_research),
                        "needs_clarification": bool(needs_clarification),
                        "planned_queries": list(queries),
                    },
                    outcome={
                        "decision": "hitl_pause",
                        "interrupt_id": "clarify_research_scope",
                    },
                )
                return RequestInput(
                    interrupt_id="clarify_research_scope",
                    message=prompt_msg,
                )

        clarification_reply = resume_inputs.get("clarify_research_scope")
        if route == "research" and queries:
            planned = ", ".join(queries)
            payload = (
                f"User Question: {safe_user_text}\nPlanned Search Queries: {planned}"
            )
            if clarification_reply:
                payload += f"\nUser Clarification / Approval: {redact_pii(str(clarification_reply))}"
        elif route == "research" and clarification_reply:
            payload = (
                f"User Question: {safe_user_text}\n"
                f"User Clarification / Approval: {redact_pii(str(clarification_reply))}"
            )
        else:
            payload = safe_user_text or redact_pii(str(node_input))

        span.set_attribute("research.outcome.route", route)
        log_structured_event(
            "workflow_routing",
            intent={
                "user_query": safe_user_text,
                "is_research_query": bool(is_research),
                "planned_queries": list(queries),
            },
            outcome={
                "decision": "routed",
                "route": route,
                "resumed_from_hitl": bool(clarification_reply),
            },
        )
        return Event(output=payload, route=route)


route_research_node = node(route_research, rerun_on_resume=True)


# 2. Specialist Web Researcher Agent (uses higher-capacity RESEARCHER_MODEL)
web_researcher_agent = LlmAgent(
    name="web_researcher_agent",
    model=Gemini(
        model=RESEARCHER_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert Web Research Analyst.\n"
        "Your job is to investigate the user's question using your search tools and synthesize "
        "a clear, well-structured Markdown research brief.\n"
        "Check any memories from past conversations (such as the user's role, technical depth, "
        "or formatting preferences) to tailor your brief.\n\n"
        "Available Tools & Guidelines:\n"
        "1. `google_search`: Use for general web facts, recent announcements, and broad background.\n"
        "2. `search_hackernews`: Use for developer sentiment, engineering trade-offs, and tech discussions. "
        "Keep queries short (2-3 keywords, e.g. 'RAG production') because the API uses strict AND matching.\n"
        "3. `search_arxiv`: Only call this when the user asks for academic/scientific papers or theoretical research.\n"
        "4. Efficiency: Aim for 2 to 4 tool calls total before synthesizing your final brief.\n\n"
        "Response Format:\n"
        "- **Executive Summary**: 2-3 sentence answer to the user's question.\n"
        "- **Key Findings**: Bullet points synthesizing evidence from your searches.\n"
        "- **Sources**: Bullet list of real URLs returned by your tools. Never fabricate URLs."
    ),
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        search_hackernews,
        search_arxiv_tool,
        preload_memory,
    ],
    before_model_callback=[scrub_pii_before_model_callback],
    after_model_callback=[normalize_grounding_metadata_callback],
    after_agent_callback=[add_session_to_memory_callback],
)


def decline_node(ctx: Context) -> Event:
    """Politely declines requests that are not research inquiries."""
    decline_message = (
        "Hello! I am a specialized Web Research Agent. "
        "I can help you investigate topics, compare technologies, summarize academic papers on arXiv, "
        "or analyze practitioner discussions on Hacker News and the web. "
        "I'm unable to handle non-research requests—please ask me a topic or question you'd like me to research!"
    )
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=decline_message)],
        ),
        output=decline_message,
    )


class ObservabilityAndPrivacyPlugin(BasePlugin):
    """App-wide ADK plugin enforcing PII redaction and structured intent-vs-outcome logging."""

    def __init__(self, name: str = "observability_and_privacy_plugin") -> None:
        super().__init__(name=name)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: object,
        user_message: types.Content,
    ) -> types.Content | None:
        del invocation_context
        for part in getattr(user_message, "parts", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                part.text = redact_pii(text)
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: object,
    ) -> None:
        scrub_pii_before_model_callback(callback_context, llm_request)
        return None

    async def before_tool_callback(
        self,
        *,
        tool: object,
        tool_args: dict[str, Any],
        tool_context: object,
    ) -> dict[str, Any] | None:
        del tool, tool_context
        for key, value in list(tool_args.items()):
            if isinstance(value, str):
                tool_args[key] = redact_pii(value)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: object,
        tool_args: dict[str, Any],
        tool_context: object,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        del tool_context
        tool_name = getattr(tool, "name", "unknown_tool")
        log_structured_event(
            "plugin_tool_audit",
            intent={"tool": tool_name, "args": tool_args},
            outcome={"status": "completed", "has_result": bool(result)},
        )
        return None


# Keep root_agent name in sync with agents-cli-manifest.yaml (`web_research_agent`).
root_agent = Workflow(
    name="web_research_agent",
    description="Multi-agent web research workflow that plans queries, searches live sources, and synthesizes cited briefs.",
    edges=[
        ("START", query_planner_agent),
        (query_planner_agent, route_research_node),
        (
            route_research_node,
            {
                "research": web_researcher_agent,
                "unrelated": decline_node,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[ObservabilityAndPrivacyPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
)

