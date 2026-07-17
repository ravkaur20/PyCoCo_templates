"""Tests for run_gp optimizer bound tightening."""

import numpy as np
import pytest

import gp_utils as gu
from run_gp import _clip_theta_to_bounds, _tightened_optimizer_bounds


def test_tighten_log_metric_t_min_single_scale() -> None:
    cfg = gu.KernelConfig(additive_t=False, additive_w=False)
    bounds0 = cfg.default_bounds()
    bounds, msgs = _tightened_optimizer_bounds(cfg, log_metric_t_min=-2.0)
    i = cfg.free_param_names().index("log_metric_t")
    assert bounds0[i][0] < -2.0
    assert bounds[i][0] == -2.0
    assert bounds[i][1] == bounds0[i][1]
    assert any("log_metric_t" in m for m in msgs)


def test_tighten_additive_time_caps() -> None:
    cfg = gu.KernelConfig(additive_t=True, additive_w=False)
    bounds, msgs = _tightened_optimizer_bounds(
        cfg,
        log_metric_t2_max=2.5,
        logit_weight_t_min=-0.25,
    )
    names = cfg.free_param_names()
    assert "log_metric_t2" in names
    i2 = names.index("log_metric_t2")
    iw = names.index("logit_weight_t")
    assert bounds[i2][1] == 2.5
    assert bounds[iw][0] == -0.25
    assert len(msgs) == 2


def test_tighten_invalid_raises() -> None:
    cfg = gu.KernelConfig(additive_t=True, additive_w=False)
    with pytest.raises(ValueError):
        _tightened_optimizer_bounds(cfg, log_metric_t2_max=-10.0)


def test_clip_theta() -> None:
    bounds = [(-1.0, 1.0), (0.0, 2.0)]
    theta = [-5.0, 5.0]
    c = _clip_theta_to_bounds(theta, bounds)
    np.testing.assert_array_equal(c, np.array([-1.0, 2.0]))
