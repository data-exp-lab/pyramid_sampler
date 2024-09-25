from __future__ import annotations

import importlib.metadata

import pyramid_sampler as m


def test_version():
    assert importlib.metadata.version("pyramid_sampler") == m.__version__
