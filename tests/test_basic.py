"""
tests/test_basic.py
--------------------
Basic tests to verify the system works before deploying.
Run: python tests/test_basic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from catalog.loader import CATALOG, get_by_name
from retrieval.engine import get_engine

def test_catalog_loaded():
    assert len(CATALOG) > 300, f"Expected 300+ items, got {len(CATALOG)}"
    print(f"✅ Catalog loaded: {len(CATALOG)} items")

def test_test_type_mapping():
    # Find OPQ32r and check it has P type
    opq = get_by_name("Occupational Personality Questionnaire OPQ32r")
    assert opq is not None, "OPQ32r not found in catalog"
    assert "P" in opq["test_type"], f"Expected P in test_type, got {opq['test_type']}"
    print(f"✅ test_type mapping: OPQ32r -> {opq['test_type']}")

def test_get_by_name_partial():
    item = get_by_name("Java 8")
    assert item is not None, "Java 8 not found"
    print(f"✅ Partial name search: 'Java 8' -> {item['name']}")

def test_retrieval_java():
    engine = get_engine()
    results = engine.search("Java developer mid level", top_k=5)
    assert len(results) > 0
    names = [r["name"] for r in results]
    print(f"✅ Retrieval 'Java developer': {names[:3]}")
    # At least one Java test should appear
    assert any("java" in n.lower() or "Java" in n for n in names), \
        f"No Java test found in: {names}"

def test_retrieval_personality():
    engine = get_engine()
    results = engine.search("personality test for leadership executive", top_k=5)
    assert len(results) > 0
    types = [r["test_type"] for r in results]
    print(f"✅ Retrieval 'personality leadership': {[r['name'] for r in results[:3]]}")

def test_no_invalid_urls():
    from catalog.loader import CATALOG
    valid_urls = {item["link"] for item in CATALOG}
    for item in CATALOG:
        assert item["link"] in valid_urls
    print(f"✅ All {len(CATALOG)} catalog URLs are valid")

if __name__ == "__main__":
    print("Running basic tests...\n")
    test_catalog_loaded()
    test_test_type_mapping()
    test_get_by_name_partial()
    print("\nLoading retrieval engine (first time takes ~30s)...")
    test_retrieval_java()
    test_retrieval_personality()
    test_no_invalid_urls()
    print("\n✅ All tests passed!")
