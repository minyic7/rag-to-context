"""
cleaner.py — HTML → clean plain text
Strips nav, footer, scripts, ads; returns readable content.
"""
from bs4 import BeautifulSoup
import re


# Tags whose entire subtree we discard
_REMOVE_TAGS = {
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header",
    "aside", "form",
}

# CSS classes / ids that are noise (partial match)
_NOISE_PATTERNS = re.compile(
    r"(cookie|banner|popup|modal|overlay|breadcrumb|sidebar|ad-|ads-|promo)",
    re.IGNORECASE,
)


def extract_text(html: str, url: str = "") -> dict:
    """
    Parse HTML and return:
      {
        "title":   page title (str),
        "text":    clean plain text (str),
        "links":   list of absolute-ish href strings found in <a> tags
      }
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── title ──────────────────────────────────────────────────────────────
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # CommBank titles look like "Credit cards | CommBank" — keep left part
    if "|" in title:
        title = title.split("|")[0].strip()

    # ── collect links before destructive removal ───────────────────────────
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("#", "javascript", "mailto", "tel")):
            links.append(href)

    # ── remove noisy subtrees ─────────────────────────────────────────────
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None)
        if not attrs:
            continue
        cls = " ".join(attrs.get("class", []) or [])
        id_ = attrs.get("id", "") or ""
        if _NOISE_PATTERNS.search(cls) or _NOISE_PATTERNS.search(id_):
            tag.decompose()

    # ── extract text ──────────────────────────────────────────────────────
    body = soup.find("body") or soup
    lines = []
    for element in body.descendants:
        if element.name in {"h1", "h2", "h3", "h4"}:
            text = element.get_text(strip=True)
            if text:
                lines.append(f"\n## {text}")
        elif element.name in {"p", "li", "td", "th", "dt", "dd"}:
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 20:   # skip trivial fragments
                lines.append(text)

    text = "\n".join(lines)
    # collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {"title": title, "text": text.strip(), "links": links}
