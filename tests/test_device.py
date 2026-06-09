"""Device selection: env override, torch-absent CPU fallback, and mocked GPU/MPS."""

from __future__ import annotations

import sys
import types

from pipeline.device import select_device


def _fake_torch(*, cuda: bool, mps: bool) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return t


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("PIPELINE_DEVICE", "CUDA")
    assert select_device() == "cuda"


def test_cpu_when_torch_absent(monkeypatch):
    monkeypatch.delenv("PIPELINE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # None in sys.modules → import raises
    assert select_device() == "cpu"


def test_cuda_detected(monkeypatch):
    monkeypatch.delenv("PIPELINE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
    assert select_device() == "cuda"


def test_mps_fallback(monkeypatch):
    monkeypatch.delenv("PIPELINE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert select_device() == "mps"


def test_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.delenv("PIPELINE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    assert select_device() == "cpu"
