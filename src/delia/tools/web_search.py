# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Web search tool using DuckDuckGo.

Provides free, no-API-key web search capability for local LLMs.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import structlog

log = structlog.get_logger()


async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: Literal["on", "moderate", "off"] = "moderate",
) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-10)
        region: Region for search results (default: worldwide)
            Examples: "us-en", "uk-en", "de-de", "wt-wt" (worldwide)
        safesearch: SafeSearch setting ("on", "moderate", "off")

    Returns:
        Formatted search results as markdown string
    """
    from duckduckgo_search import DDGS

    max_results = min(max(1, max_results), 10)  # Clamp to 1-10

    log.info("web_search_start", query=query, max_results=max_results)

    try:
        # Run in thread pool since DDGS is synchronous
        def do_search():
            with DDGS() as ddgs:
                return list(ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                ))

        results = await asyncio.to_thread(do_search)

        if not results:
            log.info("web_search_no_results", query=query)
            return f"No results found for: {query}"

        # Format results as markdown
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            body = r.get("body", "")

            formatted.append(f"### {i}. {title}\n**URL:** {url}\n\n{body}")

        output = f"## Web Search Results for: {query}\n\n" + "\n\n---\n\n".join(formatted)

        log.info("web_search_complete", query=query, num_results=len(results))
        return output

    except Exception as e:
        log.error("web_search_error", query=query, error=str(e))
        return f"Search failed: {e}"


async def web_news(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: Literal["on", "moderate", "off"] = "moderate",
    timelimit: Literal["d", "w", "m"] | None = None,
) -> str:
    """
    Search news articles using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-10)
        region: Region for search results
        safesearch: SafeSearch setting
        timelimit: Time limit for news ("d"=day, "w"=week, "m"=month)

    Returns:
        Formatted news results as markdown string
    """
    from duckduckgo_search import DDGS

    max_results = min(max(1, max_results), 10)

    log.info("web_news_start", query=query, max_results=max_results, timelimit=timelimit)

    try:
        def do_search():
            with DDGS() as ddgs:
                return list(ddgs.news(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                ))

        results = await asyncio.to_thread(do_search)

        if not results:
            log.info("web_news_no_results", query=query)
            return f"No news found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            body = r.get("body", "")
            source = r.get("source", "")
            date = r.get("date", "")

            header = f"### {i}. {title}"
            if source:
                header += f" ({source})"
            if date:
                header += f" - {date}"

            formatted.append(f"{header}\n**URL:** {url}\n\n{body}")

        output = f"## News Results for: {query}\n\n" + "\n\n---\n\n".join(formatted)

        log.info("web_news_complete", query=query, num_results=len(results))
        return output

    except Exception as e:
        log.error("web_news_error", query=query, error=str(e))
        return f"News search failed: {e}"


async def web_fetch_text(url: str, max_chars: int = 8000) -> str:
    """
    Fetch and extract text content from a URL.

    Uses DuckDuckGo's text extraction (simpler than full scraping).

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return

    Returns:
        Extracted text content
    """
    import httpx

    log.info("web_fetch_start", url=url)

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Delia/1.0)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                # Simple HTML text extraction
                from html.parser import HTMLParser

                class TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                        self.skip_tags = {"script", "style", "nav", "footer", "header"}
                        self.current_tag = None

                    def handle_starttag(self, tag, attrs):
                        self.current_tag = tag

                    def handle_endtag(self, tag):
                        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                            self.text.append("\n")
                        self.current_tag = None

                    def handle_data(self, data):
                        if self.current_tag not in self.skip_tags:
                            text = data.strip()
                            if text:
                                self.text.append(text + " ")

                parser = TextExtractor()
                parser.feed(response.text)
                text = "".join(parser.text)

                # Clean up whitespace
                import re
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip()

            else:
                # Plain text
                text = response.text

            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n... (truncated)"

            log.info("web_fetch_complete", url=url, chars=len(text))
            return f"## Content from: {url}\n\n{text}"

    except Exception as e:
        log.error("web_fetch_error", url=url, error=str(e))
        return f"Failed to fetch URL: {e}"


# Tool definitions for registry
WEB_SEARCH_TOOLS = {
    "web_search": {
        "handler": web_search,
        "description": "Search the web using DuckDuckGo. Returns relevant web pages for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    "web_news": {
        "handler": web_news,
        "description": "Search recent news articles using DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The news search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (1-10)",
                    "default": 5,
                },
                "timelimit": {
                    "type": "string",
                    "enum": ["d", "w", "m"],
                    "description": "Time limit: d=day, w=week, m=month",
                },
            },
            "required": ["query"],
        },
    },
    "web_fetch": {
        "handler": web_fetch_text,
        "description": "Fetch and extract text content from a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return",
                    "default": 8000,
                },
            },
            "required": ["url"],
        },
    },
}
