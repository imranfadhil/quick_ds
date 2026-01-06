from __future__ import annotations

import importlib.metadata

import quick_ds as m


def test_version():
    assert importlib.metadata.version("quick_ds") == m.__version__
