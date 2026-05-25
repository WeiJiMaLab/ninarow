"""Pytest configuration for model_fitting tests."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: BADS-based tests that may take minutes")
