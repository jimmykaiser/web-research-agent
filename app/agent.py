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

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.tools import preload_memory
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel, Field


MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


async def add_session_to_memory_callback(callback_context: CallbackContext):
    """Persists session context to Vertex AI Memory Bank when available."""
    try:
        await callback_context.add_session_to_memory()
    except ValueError:
        # Memory service is not configured in local in-memory test runs
        pass
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


def search_hackernews(query: str, max_results: int = 5) -> str:
    """Search Hacker News stories and developer discussions for a topic.

    Args:
        query: Search keywords (e.g. "retrieval augmented generation" or "WebAssembly").
        max_results: Maximum number of stories to return (1 to 10).

    Returns:
        Formatted string of matching Hacker News stories with points and URLs.
    """
    limit = max(1, min(max_results, 10))
    encoded = urllib.parse.quote(query)
    url = f"https://hn.algolia.com/api/v1/search?query={encoded}&tags=story&hitsPerPage={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "adk-web-research-agent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    hits = data.get("hits", [])
    if not hits:
        return f"No Hacker News stories found for: {query}"
    lines = []
    for hit in hits:
        title = hit.get("title") or "Untitled"
        points = hit.get("points") or 0
        object_id = hit.get("objectID", "")
        story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"
        lines.append(f"- {title} ({points} points) | Article: {story_url} | HN Discussion: {hn_url}")
    return "\n".join(lines)


def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for scientific and technical research papers.

    Args:
        query: Search terms for research papers (e.g. "mixture of experts language models").
        max_results: Maximum number of papers to return (1 to 5).

    Returns:
        Formatted string containing paper titles, publication dates, summaries, and arXiv URLs.
    """
    limit = max(1, min(max_results, 5))
    encoded = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "adk-web-research-agent/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_bytes = resp.read()
    root = ET.fromstring(xml_bytes)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        return f"No arXiv papers found for: {query}"
    results = []
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
        link = entry.findtext("atom:id", default="", namespaces=ns) or ""
        results.append(f"- [{published}] {title}\n  URL: {link}\n  Abstract: {summary[:350]}...")
    return "\n".join(results)


# 1. Planner & Classifier Agent
query_planner_agent = LlmAgent(
    name="query_planner_agent",
    model=Gemini(
        model=MODEL,
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
        "If it is a research query, generate 2 to 3 concise search queries (2-4 keywords each) and output using the ResearchPlan schema."
    ),
    output_schema=ResearchPlan,
)


def route_research(ctx: Context, node_input: dict | ResearchPlan) -> Event:
    """Routes valid research plans to the researcher agent or declines off-topic prompts."""
    if isinstance(node_input, dict):
        is_research = node_input.get("is_research_query", False)
        queries = node_input.get("search_queries", [])
    else:
        is_research = getattr(node_input, "is_research_query", False)
        queries = getattr(node_input, "search_queries", [])

    route = "research" if is_research else "unrelated"

    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
        user_text = "".join(p.text for p in ctx.user_content.parts if p.text)

    if route == "research" and queries:
        planned = ", ".join(queries)
        payload = f"User Question: {user_text}\nPlanned Search Queries: {planned}"
    else:
        payload = user_text or str(node_input)

    return Event(output=payload, route=route)


# 2. Specialist Web Researcher Agent
web_researcher_agent = LlmAgent(
    name="web_researcher_agent",
    model=Gemini(
        model=MODEL,
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
        search_arxiv,
        preload_memory,
    ],
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


# Keep root_agent name in sync with agents-cli-manifest.yaml (`web_research_agent`).
root_agent = Workflow(
    name="web_research_agent",
    description="Multi-agent web research workflow that plans queries, searches live sources, and synthesizes cited briefs.",
    edges=[
        ("START", query_planner_agent),
        (query_planner_agent, route_research),
        (
            route_research,
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
)
