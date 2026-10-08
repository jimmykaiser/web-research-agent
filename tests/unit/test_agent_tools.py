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

from google.genai import types

from app.agent import (
    ResearchPlan,
    decline_node,
    route_research,
    search_arxiv,
    search_hackernews,
)


def _make_fake_ctx(user_text: str):
    return SimpleNamespace(
        user_content=types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)],
        )
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
