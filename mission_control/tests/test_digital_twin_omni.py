"""OmniView (digital-twin /omni) smoke tests: route + static assets exist."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = os.path.join(ROOT, "digital-twin")
STATIC = os.path.join(DT, "static")

MODULES = ["frames", "net", "bowl", "robot", "pointcloud", "persons", "hud", "views", "demo"]


def test_static_files_exist():
    for rel in ["omni.html", "omni.js", "go2_description/go2_description.urdf",
                "go2_description/meshes/base.dae"] + [f"omni/{m}.js" for m in MODULES]:
        assert os.path.isfile(os.path.join(STATIC, rel)), rel


def test_omni_html_references_modules():
    html = open(os.path.join(STATIC, "omni.html"), encoding="utf-8").read()
    assert "/static/omni.js" in html and "three.js/r128" in html
    js = open(os.path.join(STATIC, "omni.js"), encoding="utf-8").read()
    for m in MODULES:
        assert f"./omni/{m}.js" in js, m


def test_omni_route_serves_html():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    sys.path.insert(0, DT)
    try:
        import server  # noqa: WPS433
    finally:
        sys.path.remove(DT)
    client = TestClient(server.app)
    r = client.get("/omni")
    assert r.status_code == 200
    assert "OmniVision 360" in r.text
    assert client.get("/static/omni/bowl.js").status_code == 200
    assert client.get("/").status_code == 200   # existing route untouched
