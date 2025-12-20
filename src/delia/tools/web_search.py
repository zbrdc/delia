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
Includes quality validation to filter out garbage results.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Literal

import structlog

log = structlog.get_logger()


# ============================================================
# Search Result Quality Validation
# ============================================================

@dataclass
class SearchResultQuality:
    """Quality assessment for a single search result."""
    is_valid: bool
    score: float  # 0.0 - 1.0
    issues: list[str]


def validate_search_result(
    title: str,
    url: str,
    body: str,
    query: str,
) -> SearchResultQuality:
    """
    Validate a single search result for quality.

    Checks:
    - Completeness: Has title, URL, and body
    - Content quality: Body is meaningful (not spam/SEO garbage)
    - Basic relevance: Some connection to query terms

    Args:
        title: Result title
        url: Result URL
        body: Result body/snippet
        query: Original search query

    Returns:
        SearchResultQuality with validity, score, and issues
    """
    issues: list[str] = []
    scores: list[float] = []

    # 1. Completeness check
    if not title or len(title.strip()) < 3:
        issues.append("missing_title")
        scores.append(0.0)
    else:
        scores.append(1.0)

    if not url or not url.startswith(("http://", "https://")):
        issues.append("invalid_url")
        scores.append(0.0)
    else:
        scores.append(1.0)

    if not body or len(body.strip()) < 10:
        issues.append("missing_body")
        scores.append(0.3)  # Partial penalty - some results legitimately have short snippets
    else:
        scores.append(1.0)

    # 2. Content quality checks
    if body:
        body_lower = body.lower()

        # Check for repetitive content (spam indicator)
        words = body_lower.split()
        if len(words) >= 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues.append("repetitive_content")
                scores.append(0.3)
            else:
                scores.append(min(1.0, unique_ratio * 1.5))

        # Check for excessive special characters (SEO spam)
        special_char_ratio = len(re.findall(r'[^\w\s]', body)) / max(len(body), 1)
        if special_char_ratio > 0.3:
            issues.append("excessive_special_chars")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # Check for gibberish (high ratio of very short words)
        if words:
            short_word_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
            if short_word_ratio > 0.5:
                issues.append("possible_gibberish")
                scores.append(0.5)
            else:
                scores.append(1.0)

    # 3. Basic relevance check
    if query and body:
        query_terms = set(query.lower().split())
        body_lower = body.lower()
        matches = sum(1 for term in query_terms if term in body_lower)
        relevance = matches / max(len(query_terms), 1)
        if relevance < 0.2:
            issues.append("low_relevance")
            scores.append(0.5)  # Partial penalty - relevance is heuristic
        else:
            scores.append(min(1.0, relevance + 0.5))

    # Calculate overall score
    overall = sum(scores) / len(scores) if scores else 0.0

    # Result is valid if score >= 0.5 and no critical issues
    critical_issues = {"missing_title", "invalid_url"}
    has_critical = bool(critical_issues & set(issues))
    is_valid = overall >= 0.5 and not has_critical

    return SearchResultQuality(
        is_valid=is_valid,
        score=overall,
        issues=issues,
    )


def filter_quality_results(
    results: list[dict],
    query: str,
    min_score: float = 0.5,
) -> tuple[list[dict], int]:
    """
    Filter search results by quality score.

    Args:
        results: Raw search results from DuckDuckGo
        query: Original search query
        min_score: Minimum quality score to keep (0.0 - 1.0)

    Returns:
        Tuple of (filtered_results, num_rejected)
    """
    filtered = []
    rejected = 0

    for r in results:
        title = r.get("title", "")
        url = r.get("href", "") or r.get("url", "")
        body = r.get("body", "")

        quality = validate_search_result(title, url, body, query)

        if quality.is_valid and quality.score >= min_score:
            # Add quality score to result for transparency
            r["_quality_score"] = round(quality.score, 2)
            filtered.append(r)
        else:
            rejected += 1
            log.debug(
                "search_result_rejected",
                title=title[:50] if title else "none",
                score=quality.score,
                issues=quality.issues,
            )

    return filtered, rejected


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

        # Filter out garbage results
        filtered_results, rejected_count = filter_quality_results(results, query)

        if not filtered_results:
            log.warning(
                "web_search_all_filtered",
                query=query,
                total=len(results),
                rejected=rejected_count,
            )
            return f"No quality results found for: {query} ({rejected_count} low-quality results filtered)"

        # Format results as markdown
        formatted = []
        for i, r in enumerate(filtered_results, 1):
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            body = r.get("body", "")
            quality = r.get("_quality_score", 0)

            formatted.append(f"### {i}. {title}\n**URL:** {url}\n\n{body}")

        output = f"## Web Search Results for: {query}\n\n" + "\n\n---\n\n".join(formatted)

        # Add quality summary if any were filtered
        if rejected_count > 0:
            output += f"\n\n*({rejected_count} low-quality results filtered)*"

        log.info(
            "web_search_complete",
            query=query,
            num_results=len(filtered_results),
            rejected=rejected_count,
        )
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

        # Filter out garbage results (news uses 'url' not 'href')
        # Normalize to use 'href' for filter function
        for r in results:
            if "url" in r and "href" not in r:
                r["href"] = r["url"]

        filtered_results, rejected_count = filter_quality_results(results, query)

        if not filtered_results:
            log.warning(
                "web_news_all_filtered",
                query=query,
                total=len(results),
                rejected=rejected_count,
            )
            return f"No quality news found for: {query} ({rejected_count} low-quality results filtered)"

        formatted = []
        for i, r in enumerate(filtered_results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "") or r.get("href", "")
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

        # Add quality summary if any were filtered
        if rejected_count > 0:
            output += f"\n\n*({rejected_count} low-quality results filtered)*"

        log.info(
            "web_news_complete",
            query=query,
            num_results=len(filtered_results),
            rejected=rejected_count,
        )
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
