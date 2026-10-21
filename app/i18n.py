"""
Localization for user-facing text (currently: the camera "Describe" feature).

Each language is a single JSON file in app/locales/<code>.json — to add a
language, copy en.json, translate it and rebuild. See docs/adding-a-language.md.

Language resolution order (resolve_lang):
  1. explicit ?lang=xx (the web UI sends navigator.language)
  2. Accept-Language header
  3. APP_LANG env var (default "en")
  4. "en"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

_LOGGER = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
FALLBACK_LANG = "en"
DEFAULT_LANG = (os.environ.get("APP_LANG") or FALLBACK_LANG).strip().lower()

_REQUIRED_PHRASES = ("seen", "nothing", "separator", "count")
_REQUIRED_ERRORS = ("camera_not_found", "snapshot_failed", "missing_jpeg")
_COCO_CLASS_COUNT = 80


def _load_locales() -> dict[str, dict]:
    locales: dict[str, dict] = {}
    for path in sorted(LOCALES_DIR.glob("*.json")):
        code = path.stem.lower()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _LOGGER.error("Locale %s: cannot load (%s) — skipped", path.name, exc)
            continue
        missing = [k for k in _REQUIRED_PHRASES if k not in data.get("phrases", {})]
        missing += [k for k in _REQUIRED_ERRORS if k not in data.get("errors", {})]
        missing += [f"labels.{i}" for i in range(_COCO_CLASS_COUNT)
                    if str(i) not in data.get("labels", {})]
        if missing:
            _LOGGER.warning("Locale %s: missing %d keys (%s…) — English used for those",
                            code, len(missing), ", ".join(missing[:5]))
        locales[code] = data
    if FALLBACK_LANG not in locales:
        raise RuntimeError(f"Fallback locale {FALLBACK_LANG}.json is missing in {LOCALES_DIR}")
    _LOGGER.info("Locales loaded: %s (default: %s)", ", ".join(locales), DEFAULT_LANG)
    return locales


LOCALES = _load_locales()


def available_languages() -> list[dict]:
    return [{"code": code, "name": d.get("_meta", {}).get("name", code)}
            for code, d in LOCALES.items()]


def _match(tag: str | None) -> str | None:
    """'en-US' → 'en' if only 'en' exists; exact match wins."""
    if not tag:
        return None
    tag = tag.strip().lower().replace("_", "-")
    if tag in LOCALES:
        return tag
    base = tag.split("-")[0]
    return base if base in LOCALES else None


def resolve_lang(request: Request | None = None, lang: str | None = None) -> str:
    if (m := _match(lang)):
        return m
    if request is not None:
        header = request.headers.get("accept-language", "")
        # "en-US,en;q=0.9,bg;q=0.8" → by q-value, highest first
        tags = []
        for part in header.split(","):
            pieces = part.strip().split(";")
            q = 1.0
            for p in pieces[1:]:
                if p.strip().startswith("q="):
                    try:
                        q = float(p.strip()[2:])
                    except ValueError:
                        q = 0.0
            tags.append((q, pieces[0]))
        for _, t in sorted(tags, key=lambda x: -x[0]):
            if (m := _match(t)):
                return m
    return _match(DEFAULT_LANG) or FALLBACK_LANG


def _get(lang: str, section: str, key: str) -> str | dict | None:
    for code in (lang, FALLBACK_LANG):
        val = LOCALES.get(code, {}).get(section, {}).get(key)
        if val is not None:
            return val
    return None


def t(lang: str, key: str) -> str:
    """Error / UI message, e.g. t(lang, 'camera_not_found')."""
    return _get(lang, "errors", key) or key


def label(lang: str, class_id: int) -> str:
    entry = _get(lang, "labels", str(class_id))
    return entry["one"] if isinstance(entry, dict) else str(class_id)


def describe(lang: str, detections: list[dict]) -> str:
    """Builds the sentence for a list of detections ({'class_id': int, ...})."""
    if not detections:
        return _get(lang, "phrases", "nothing")

    from collections import Counter
    counts = Counter(d["class_id"] for d in detections)
    count_tpl = _get(lang, "phrases", "count")
    parts = []
    for cls_id, n in counts.most_common():
        entry = _get(lang, "labels", str(cls_id))
        if isinstance(entry, dict):
            word = entry["one"] if n == 1 else entry.get("other", entry["one"])
        else:
            word = str(cls_id)
        parts.append(count_tpl.format(n=n, word=word))
    items = _get(lang, "phrases", "separator").join(parts)
    return _get(lang, "phrases", "seen").format(items=items)
