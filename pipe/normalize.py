def normalize(data: dict | list | str | None, _depth: int = 0) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, list):
        return " ".join(normalize(item, _depth + 1) for item in data if item)
    if isinstance(data, dict):
        parts = []
        for key, val in data.items():
            if val in (None, "", [], {}):
                continue
            label = key.replace("_", " ").title()
            content = normalize(val, _depth + 1)
            if content:
                parts.append(f"{label}: {content}" if _depth == 0 else content)
        return ". ".join(parts)
    return ""


def page_json_to_text(page_json: dict) -> str:
    """
    Entry point: takes the full vision model output for one page
    and returns a single clean string.
    """
    parts = []

    if title := page_json.get("title"):
        parts.append(f"Title: {title}")

    if summary := page_json.get("summary"):
        parts.append(f"Summary: {summary}")

    content = page_json.get("content", {})
    if content:
        parts.append(normalize(content))

    return "\n".join(filter(None, parts))