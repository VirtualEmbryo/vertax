"""The `gpu` argument reaches Cellpose from `PbcMesh.from_image` down to `segment`."""

import importlib.util
import sys
import types

import numpy as np
import pytest

from vertax import mask_analysis

pytestmark = pytest.mark.skipif(importlib.util.find_spec("cellpose") is None, reason="Cellpose is not installed")


class _StubCellpose:
    """Stands in for `cellpose.models.Cellpose` and records its constructor arguments."""

    calls: list[dict] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).calls.append(kwargs)

    def eval(self, image: np.ndarray, **kwargs: object) -> tuple:
        mask = np.ones(image.shape[:2], dtype=np.int32)
        return mask, None, None, None


@pytest.fixture
def stub_cellpose(monkeypatch: pytest.MonkeyPatch) -> type[_StubCellpose]:
    # Cellpose itself is not imported: the stub packages take its place for the duration of the test.
    models = types.ModuleType("cellpose.models")
    models.Cellpose = _StubCellpose
    package = types.ModuleType("cellpose")
    package.models = models
    _StubCellpose.calls = []
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return _StubCellpose


def _image() -> np.ndarray:
    return np.zeros((8, 8))


@pytest.mark.parametrize("gpu", [True, False])
def test_segment_forwards_explicit_device(stub_cellpose: type[_StubCellpose], gpu: bool) -> None:
    mask_analysis.segment(_image(), gpu=gpu)
    assert stub_cellpose.calls[-1]["gpu"] is gpu


@pytest.mark.parametrize("cuda", [True, False])
def test_segment_default_uses_cuda_only(
    stub_cellpose: type[_StubCellpose], monkeypatch: pytest.MonkeyPatch, cuda: bool
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    mask_analysis.segment(_image())
    assert stub_cellpose.calls[-1]["gpu"] is cuda


def test_mask_from_image_forwards_device(stub_cellpose: type[_StubCellpose]) -> None:
    mask_analysis.mask_from_image(_image(), gpu=False)
    assert stub_cellpose.calls[-1]["gpu"] is False


def test_from_image_forwards_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from vertax import PbcMesh
    from vertax.meshes import pbc_mesh

    received: dict = {}

    def fake_mask_from_image(image: np.ndarray, gpu: bool | None = None) -> np.ndarray:
        received["gpu"] = gpu
        return np.zeros((2, 2), dtype=int)

    monkeypatch.setattr(pbc_mesh, "mask_from_image", fake_mask_from_image)
    monkeypatch.setattr(PbcMesh, "from_mask", classmethod(lambda cls, mask: "mesh"))
    assert PbcMesh.from_image(_image(), gpu=False) == "mesh"
    assert received["gpu"] is False
    PbcMesh.from_image(_image())
    assert received["gpu"] is None
