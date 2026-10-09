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

import io
from types import SimpleNamespace
from unittest.mock import patch

from google.adk.events.request_input import RequestInput
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import app.agent as agent_module
from app.agent import (
    PLANNER_MODEL,
    RESEARCHER_MODEL,
    ResearchPlan,
    app,
    decline_node,
    log_structured_event,
    normalize_grounding_metadata_callback,
    query_planner_agent,
    redact_pii,
    requires_exhaustive_search_approval,
    route_research,
    scrub_pii_before_model_callback,
    search_arxiv,
    search_hackernews,
    start_linked_span,
    web_researcher_agent,
)


def _make_fake_ctx(
    user_text: str,
    state: dict | None = None,
    resume_inputs: dict | None = None,
):
    return SimpleNamespace(
        user_content=types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)],
        ),
        state=state or {},
        resume_inputs=resume_inputs or {},
    )


def test_route_research_valid_query() -> None:
    ctx = _make_fake_ctx("What is retrieval augmented generation?")
    plan = ResearchPlan(
        is_research_query=True,
        search_queries=["retrieval augmented generation overview", "RAG trade-offs"],
        reasoning="User asks about a technical AI architecture.",
    )
    event = route_research(ctx, plan)
    assert event.actions.route == "research"
    assert "What is retrieval augmented generation?" in event.output
    assert "retrieval augmented generation overview" in event.output


def test_route_research_unrelated_query() -> None:
    ctx = _make_fake_ctx("Tell me a joke")
    plan = ResearchPlan(
        is_research_query=False,
        search_queries=[],
        reasoning="Casual request for a joke.",
    )
    event = route_research(ctx, plan)
    assert event.actions.route == "unrelated"


def test_route_research_hitl_pause_and_resume() -> None:
    plan = ResearchPlan(
        is_research_query=True,
        search_queries=[],
        reasoning="Ambiguous research request missing a subject.",
        needs_clarification=True,
        clarification_question="Which two databases would you like me to compare?",
    )
    # 1. First execution halts and returns RequestInput
    pause_ctx = _make_fake_ctx("Compare the two")
    pause_result = route_research(pause_ctx, plan)
    assert isinstance(pause_result, RequestInput)
    assert pause_result.interrupt_id == "clarify_research_scope"
    assert "Which two databases" in (pause_result.message or "")

    # 2. Resumed execution incorporates human input and routes to researcher
    resume_ctx = _make_fake_ctx(
        "Compare the two",
        resume_inputs={"clarify_research_scope": "PostgreSQL vs DuckDB for analytics"},
    )
    resume_event = route_research(resume_ctx, plan)
    assert resume_event.actions.route == "research"
    assert "PostgreSQL vs DuckDB for analytics" in resume_event.output


def test_strategic_model_routing_and_hitl_tool_gate() -> None:
    assert query_planner_agent.model.model == PLANNER_MODEL
    assert web_researcher_agent.model.model == RESEARCHER_MODEL
    assert PLANNER_MODEL != RESEARCHER_MODEL
    assert app.resumability_config is not None
    assert app.resumability_config.is_resumable is True
    assert requires_exhaustive_search_approval(max_results=3) is False
    assert requires_exhaustive_search_approval(max_results=10) is True


def test_decline_node_message() -> None:
    ctx = _make_fake_ctx("Hello")
    event = decline_node(ctx)
    assert "Web Research Agent" in event.output


def test_search_hackernews_parses_response() -> None:
    fake_payload = b'{"hits": [{"title": "Show HN: Fast RAG", "points": 142, "objectID": "999", "url": "https://example.com/rag"}]}'
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = io.BytesIO(fake_payload)
        result = search_hackernews("RAG", max_results=3)
    assert "Show HN: Fast RAG" in result
    assert "142 points" in result
    assert "https://example.com/rag" in result


def test_search_arxiv_parses_atom_xml() -> None:
    fake_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <published>2024-01-01T00:00:00Z</published>
        <title>Mixture of Experts Survey</title>
        <summary>A comprehensive survey of sparse MoE architectures.</summary>
      </entry>
    </feed>"""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = io.BytesIO(fake_xml)
        result = search_arxiv("mixture of experts", max_results=2)
    assert "Mixture of Experts Survey" in result
    assert "http://arxiv.org/abs/2401.00001v1" in result


def test_normalize_grounding_metadata_callback_coerces_dict() -> None:
    original = types.GroundingMetadata(web_search_queries=["RAG overview"])
    dumped_dict = original.model_dump(exclude_none=True)
    ctx = SimpleNamespace(state={"temp:_adk_grounding_metadata": dumped_dict})
    normalize_grounding_metadata_callback(ctx, llm_response=None)
    restored = ctx.state["temp:_adk_grounding_metadata"]
    assert isinstance(restored, types.GroundingMetadata)
    assert restored.model_dump(exclude_none=True, mode="json")["web_search_queries"] == [
        "RAG overview"
    ]


def test_pii_redaction_and_before_model_guardrail() -> None:
    raw = "Contact jane.doe@example.com or 415-555-0199 with SSN 123-45-6789 and key sk-1234567890abcdefghijklmno"
    scrubbed = redact_pii(raw)
    assert "jane.doe@example.com" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed

    llm_req = SimpleNamespace(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Email me at admin@corp.io about RAG")],
            )
        ]
    )
    scrub_pii_before_model_callback(SimpleNamespace(), llm_req)
    assert llm_req.contents[0].parts[0].text == "Email me at [REDACTED_EMAIL] about RAG"


def test_structured_logging_and_linked_otel_spans() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    old_tracer = agent_module.tracer
    agent_module.tracer = provider.get_tracer("test_web_research_agent")
    agent_module._LAST_STAGE_SPAN_CONTEXT = None
    try:
        with start_linked_span("stage.one", attributes={"stage": 1}) as span1:
            ctx1 = span1.get_span_context()
            record = log_structured_event(
                "test_event",
                intent={"query": "email user@example.com"},
                outcome={"status": "routed"},
            )
            assert record["intent"]["query"] == "email [REDACTED_EMAIL]"
            assert record["outcome"]["status"] == "routed"
            assert record["trace_id"] == f"{ctx1.trace_id:032x}"

        with start_linked_span("stage.two", attributes={"stage": 2}):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        assert spans[0].name == "stage.one"
        assert spans[1].name == "stage.two"
        assert len(spans[1].links) == 1
        assert spans[1].links[0].context.span_id == spans[0].context.span_id
    finally:
        agent_module.tracer = old_tracer


import asyncio


def test_observability_and_privacy_plugin_registered_on_app() -> None:
    assert len(app.plugins) == 1
    plugin = app.plugins[0]
    assert isinstance(plugin, agent_module.ObservabilityAndPrivacyPlugin)

    msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Reach me at bob@example.org")],
    )
    asyncio.run(
        plugin.on_user_message_callback(invocation_context=None, user_message=msg)
    )
    assert msg.parts[0].text == "Reach me at [REDACTED_EMAIL]"

    tool_args = {"query": "RAG papers by alice@university.edu"}
    asyncio.run(
        plugin.before_tool_callback(
            tool=SimpleNamespace(name="search_arxiv"),
            tool_args=tool_args,
            tool_context=None,
        )
    )
    assert tool_args["query"] == "RAG papers by [REDACTED_EMAIL]"

