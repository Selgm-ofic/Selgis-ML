#!/usr/bin/env python3
"""Legacy test runner — delegates to pytest.

Usage:
    python test_selgis.py
    selgis test
"""
import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-x", "-v", "tests/"]))
