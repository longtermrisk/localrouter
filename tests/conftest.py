"""Pytest configuration for localrouter tests."""

import asyncio
import pytest


def pytest_collection_modifyitems(items):
    """Automatically add asyncio marker to all async test functions."""
    for item in items:
        if hasattr(item, "function") and asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)