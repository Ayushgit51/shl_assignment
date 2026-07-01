"""
catalog/loader.py
-----------------
Loads the SHL product catalog JSON and converts each entry into a clean dict
that the retrieval and agent layers can consume.

test_type mapping (from 'keys' field):
  Knowledge & Skills        -> K
  Personality & Behavior    -> P
  Ability & Aptitude        -> A
  Competencies              -> C
  Development & 360         -> D
  Simulations               -> S
  Biodata & Situational J.  -> B
  Assessment Exercises      -> E
"""

import json
import os
from typing import List, Dict

# Path to catalog JSON (same folder as this file)
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "shl_catalog.json")

KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Development & 360": "D",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Assessment Exercises": "E",
}


def _keys_to_test_type(keys: List[str]) -> str:
    """Convert list of keys to comma-separated test_type codes. e.g. ['Personality & Behavior','Competencies'] -> 'P,C'"""
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def _build_search_text(item: Dict) -> str:
    """Build a single string used for embedding/BM25 search."""
    parts = [
        item["name"],
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def load_catalog() -> List[Dict]:
    """
    Returns list of clean catalog items. Each item has:
      - entity_id, name, link, description
      - job_levels (list), languages (list)
      - duration (str), remote (yes/no), adaptive (yes/no)
      - test_type (str)  e.g. "K" or "P,C"
      - keys (list)      original keys
      - search_text      combined string for retrieval
    """
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    catalog = []
    for item in raw:
        if item.get("status") != "ok":
            continue

        clean = {
            "entity_id": item["entity_id"],
            "name": item["name"].strip(),
            "link": item["link"].strip(),
            "description": item.get("description", "").strip(),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", "").strip(),
            "remote": item.get("remote", "yes"),
            "adaptive": item.get("adaptive", "no"),
            "keys": item.get("keys", []),
            "test_type": _keys_to_test_type(item.get("keys", [])),
        }
        clean["search_text"] = _build_search_text(clean)
        catalog.append(clean)

    return catalog


# Singleton — loaded once at import time
CATALOG: List[Dict] = load_catalog()


def get_by_name(name: str) -> Dict | None:
    """Find a catalog item by exact or partial name match (case-insensitive)."""
    name_lower = name.lower()
    for item in CATALOG:
        if item["name"].lower() == name_lower:
            return item
    # partial match fallback
    for item in CATALOG:
        if name_lower in item["name"].lower():
            return item
    return None


def get_by_entity_id(entity_id: str) -> Dict | None:
    for item in CATALOG:
        if item["entity_id"] == entity_id:
            return item
    return None
