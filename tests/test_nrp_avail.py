"""Scheduling predicates in scripts/nrp_avail.py (pure; no cluster access)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import nrp_avail as na  # noqa: E402

A10 = {"nvidia.com/gpu.product": "NVIDIA-A10", "kubernetes.io/hostname": "n1"}


def _spec(exprs, **kw):
    return {"affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution":
                                          {"nodeSelectorTerms": [{"matchExpressions": exprs}]}}}, **kw}


def test_affinity_in_notin_and_terms():
    spec = _spec([{"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A10"]},
                  {"key": "kubernetes.io/hostname", "operator": "NotIn", "values": ["bad"]}])
    assert na.affinity_ok(A10, spec)
    assert not na.affinity_ok({**A10, "kubernetes.io/hostname": "bad"}, spec)
    assert not na.affinity_ok({**A10, "nvidia.com/gpu.product": "NVIDIA-L4"}, spec)
    # ignoring the product key lets any product through, other expressions still apply
    assert na.affinity_ok({**A10, "nvidia.com/gpu.product": "NVIDIA-L4"}, spec,
                          frozenset({"nvidia.com/gpu.product"}))
    assert na.affinity_ok({}, {})                         # no affinity at all
    two = {"affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
        {"matchExpressions": [{"key": "x", "operator": "Exists"}]},
        {"matchExpressions": [{"key": "y", "operator": "DoesNotExist"}]}]}}}}
    assert na.affinity_ok({"x": "1", "y": "1"}, two) and na.affinity_ok({}, two)
    assert not na.affinity_ok({"y": "1"}, two)


def test_node_selector():
    assert na.affinity_ok({"a": "b"}, {"nodeSelector": {"a": "b"}})
    assert not na.affinity_ok({"a": "c"}, {"nodeSelector": {"a": "b"}})


def test_taints_and_tolerations():
    gpu_ns = {"key": "nvidia.com/gpu", "effect": "NoSchedule"}
    resv = {"key": "nautilus.io/reservation", "value": "csu", "effect": "NoSchedule"}
    soft = {"key": "nvidia.com/gpu", "effect": "PreferNoSchedule"}
    tol_exists = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert na.untolerated_taints([soft], []) == []                       # soft taints never block
    assert na.untolerated_taints([gpu_ns], []) == [gpu_ns]
    assert na.untolerated_taints([gpu_ns], tol_exists) == []
    assert na.untolerated_taints([resv], tol_exists) == [resv]
    assert na.untolerated_taints([resv], [{"key": "nautilus.io/reservation", "operator": "Equal",
                                           "value": "csu", "effect": "NoSchedule"}]) == []
    assert na.untolerated_taints([resv], [{"key": "nautilus.io/reservation", "value": "other"}]) == [resv]
    assert na.untolerated_taints([resv, gpu_ns], [{"operator": "Exists"}]) == []   # tolerate-all


def test_hard_taint_class():
    assert na.hard_taint_class([]) == "open"
    assert na.hard_taint_class([{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]) == "open"
    assert na.hard_taint_class([{"key": "nautilus.io/hardware", "value": "x", "effect": "NoSchedule"}]) \
        == "nautilus.io/hardware=x"


@pytest.mark.parametrize("v,expected", [("2", 2.0), ("500m", 0.5), ("6Gi", 6 * 2**30),
                                        ("1900Mi", 1900 * 2**20), ("128974848", 128974848.0), (None, 0.0)])
def test_qty(v, expected):
    assert na.qty(v) == pytest.approx(expected)


def test_prom_key_and_gpu_keys():
    assert na.prom_key("nvidia.com/mig-small") == "nvidia_com_mig_small"
    assert na.gpu_keys({"nvidia.com/gpu": "4", "nvidia.com/gpu.present": "1", "cpu": "8",
                        "nvidia.com/rtxa6000": "0"}) == {"nvidia.com/gpu": 4}
