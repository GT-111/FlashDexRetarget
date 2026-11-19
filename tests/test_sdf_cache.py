from pathlib import Path

import numpy as np
import pytest
import torch

from utils import sdf


@pytest.fixture
def grid_cache(tmp_path, monkeypatch):
    mesh = str(tmp_path / "object.obj")
    grid = np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2)
    monkeypatch.setattr(sdf, "_preprocess_mesh", lambda *_: None)
    monkeypatch.setattr(sdf, "_build_object_sdf", lambda *_: grid)
    return (mesh, 1.0, 0.2, 2), sdf._cache_path(mesh, 0.2, 2, 1.0), grid


def test_sdf_cache_publishes_only_after_a_complete_save(grid_cache, monkeypatch):
    args, cache, grid = grid_cache
    save = torch.save

    def observe_incomplete_write(tensor, destination):
        Path(destination).write_bytes(b"PK\x03\x04")
        assert not cache.exists()
        save(tensor, destination)

    monkeypatch.setattr(torch, "save", observe_incomplete_write)
    torch.testing.assert_close(sdf.bake_object_sdf_grid(*args, "cpu"), torch.from_numpy(grid))
    torch.testing.assert_close(torch.load(cache, weights_only=True), torch.from_numpy(grid))
    assert list(cache.parent.iterdir()) == [cache]


@pytest.mark.parametrize("prefix_bytes", [0, 4])
def test_sdf_cache_rebuilds_a_truncated_tensor(grid_cache, prefix_bytes):
    args, cache, grid = grid_cache
    cache.parent.mkdir()
    torch.save(torch.zeros_like(torch.from_numpy(grid)), cache)
    cache.write_bytes(cache.read_bytes()[:prefix_bytes])

    torch.testing.assert_close(sdf.bake_object_sdf_grid(*args, "cpu"), torch.from_numpy(grid))
    torch.testing.assert_close(torch.load(cache, weights_only=True), torch.from_numpy(grid))


def test_sdf_cache_does_not_retry_a_failed_device_transfer(grid_cache, monkeypatch):
    args, cache, grid = grid_cache
    data = torch.from_numpy(grid)
    cache.parent.mkdir()
    torch.save(data, cache)

    def fail_transfer(self, **kwargs):
        raise RuntimeError("device transfer failed")

    def unexpected_rebuild(*_):
        pytest.fail("a failed transfer must not trigger a bake")

    monkeypatch.setattr(torch.Tensor, "to", fail_transfer)
    monkeypatch.setattr(sdf, "_build_object_sdf", unexpected_rebuild)
    with pytest.raises(RuntimeError, match="device transfer failed"):
        sdf.bake_object_sdf_grid(*args, "cpu")
