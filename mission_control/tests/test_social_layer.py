"""mapping/social_layer.py: shape, asymmetry along velocity, temporal decay."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mapping"))

from social_layer import SocialConfig, SocialLayer, social_cost_grid  # noqa: E402

META = {"resolution": 0.1, "origin_x": -5.0, "origin_y": -5.0, "width": 100, "height": 120}


def at(grid, x, y):
    gx = int(np.floor((x - META["origin_x"]) / META["resolution"]))
    gy = int(np.floor((y - META["origin_y"]) / META["resolution"]))
    return int(grid[gy * META["width"] + gx])


def test_shape_dtype_and_empty():
    g = social_cost_grid([], META)
    assert g.dtype == np.uint8 and g.shape == (100 * 120,) and g.max() == 0


def test_peak_at_person_and_falls_off():
    g = social_cost_grid([{"x": 0.0, "y": 0.0}], META)
    assert at(g, 0.0, 0.0) >= 240
    assert at(g, 0.0, 0.0) > at(g, 0.5, 0.0) > at(g, 1.5, 0.0) > at(g, 4.0, 0.0)
    assert at(g, 4.0, 4.0) == 0


def test_static_person_is_symmetric():
    g = social_cost_grid([{"x": 0.0, "y": 0.0}], META)
    # cell centres at +-0.95 m are symmetric about the person at 0
    assert at(g, 0.95, 0.0) == at(g, -0.95, 0.0) > 0
    assert at(g, 0.0, 0.95) == at(g, 0.0, -0.95) == at(g, 0.95, 0.0)


def test_walking_person_front_larger_than_back_and_sides():
    g = social_cost_grid([{"x": 0.0, "y": 0.0, "vx": 1.2, "vy": 0.0}], META)
    front, back = at(g, 1.5, 0.0), at(g, -1.5, 0.0)
    side = at(g, 0.0, 1.5)
    assert front > back and front > side
    # rotated velocity -> rotated asymmetry
    g2 = social_cost_grid([{"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 1.2}], META)
    assert at(g2, 0.0, 1.5) > at(g2, 0.0, -1.5)


def test_multiple_persons_use_max_and_edge_clipping():
    g = social_cost_grid([{"x": -2.0, "y": 0.0}, {"x": 2.0, "y": 0.0}, {"x": 40.0, "y": 0.0},
                          {"x": float("nan"), "y": 0.0}], META)
    assert at(g, -2.0, 0.0) >= 240 and at(g, 2.0, 0.0) >= 240
    assert g.max() <= 255


def test_weights_scale_cost():
    g = social_cost_grid([{"x": 0.0, "y": 0.0}], META, weights=[0.5])
    assert 120 <= at(g, 0.0, 0.0) <= 130


def test_temporal_decay():
    layer = SocialLayer(META, SocialConfig(decay_s=2.5))
    layer.update([{"gid": 1, "x": 0.0, "y": 0.0}], t=10.0)
    full = at(layer.grid(10.0), 0.0, 0.0)
    half = at(layer.grid(11.25), 0.0, 0.0)
    assert full >= 240 and half == pytest.approx(full / 2, abs=4)
    assert at(layer.grid(12.4), 0.0, 0.0) > 0
    assert layer.grid(12.6).max() == 0 and len(layer) == 0


def test_layer_refresh_keeps_cost_and_moves_with_person():
    layer = SocialLayer(META)
    layer.update([{"gid": 1, "x": 0.0, "y": 0.0}], t=0.0)
    layer.update([{"gid": 1, "x": 2.0, "y": 0.0}], t=2.0)
    g = layer.grid(2.0)
    assert at(g, 2.0, 0.0) >= 240 and at(g, 0.0, 0.0) < 50
