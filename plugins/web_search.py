"""Web search tools — SearxNG primary, DuckDuckGo fallback."""
from core.tool_registry import tool


@tool("web_search", "Search the web for information", {
    "query":       {"type": "string",  "description": "Search query"},
    "max_results": {"type": "integer", "description": "Number of results (1-20)", "default": 10},
    "fetch_pages": {"type": "boolean", "description": "Also fetch and extract page text", "default": False},
})
def web_search(query: str, max_results: int = 10, fetch_pages: bool = False) -> list[dict]:
    import requests
    results = []
    try:
        r = requests.get(
            "http://localhost:4000/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=12,
        )
        results = [
            {"title": x.get("title", ""), "href": x.get("url", ""), "body": x.get("content", "")}
            for x in r.json().get("results", [])
        ][:max_results]
    except Exception:
        pass
    if not results:
        try:
            from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=max_results))
        except Exception:
            pass
    if fetch_pages and results:
        from bs4 import BeautifulSoup
        for item in results:
            try:
                html = requests.get(
                    item["href"], timeout=8,
                    headers={"User-Agent": "Mozilla/5.0"}
                ).text
                soup = BeautifulSoup(html, "lxml")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                item["page_text"] = " ".join(soup.get_text().split())[:6000]
            except Exception:
                item["page_text"] = ""
    return results
