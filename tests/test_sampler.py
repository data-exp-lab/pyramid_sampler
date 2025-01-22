from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from pyramid_sampler.sampler import Downsampler, initialize_test_image


def test_initialize_test_image(tmp_path):
    tmp_zrr = str(tmp_path / "myzarr.zarr")
    zarr_store = zarr.open(tmp_zrr)
    res = (8, 8, 8)
    chunks = (2, 2, 2)
    fieldname = "test_field"
    initialize_test_image(zarr_store, fieldname, res, chunks, overwrite_field=False)

    assert fieldname in zarr_store
    assert zarr_store[fieldname]["0"].shape == res
    assert zarr_store[fieldname]["0"].chunks == chunks
    assert Path.exists(tmp_path / "myzarr.zarr" / fieldname)

    res = (16, 16, 16)
    initialize_test_image(zarr_store, fieldname, res, chunks, overwrite_field=True)
    assert zarr_store[fieldname]["0"].shape == res


@pytest.mark.parametrize("dtype", ["float32", np.float64, "int", np.int32, np.int16])
def test_downsampler(tmp_path, dtype):
    tmp_zrr = str(tmp_path / "myzarr.zarr")
    zarr_store = zarr.open(tmp_zrr)
    res = (32, 32, 32)
    chunks = (8, 8, 8)
    fieldname = "test_field"
    initialize_test_image(
        zarr_store, fieldname, res, chunks, overwrite_field=False, dtype=dtype
    )

    dsr = Downsampler(tmp_zrr, (2, 2, 2), res, chunks)

    dsr.downsample(10, fieldname)
    expected_max_lev = 2
    for lev in range(expected_max_lev + 1):
        assert str(lev) in zarr_store[fieldname]
        assert zarr_store[fieldname][str(lev)].dtype == np.dtype(dtype)

    with pytest.raises(ValueError, match="max_level must exceed 0"):
        dsr.downsample(0, fieldname)


def test_downsampler_defaults():
    dsr = Downsampler("not_a_file.zarr", (2, 2, 2), (128, 128, 128))
    assert np.all(dsr.chunks == (64, 64, 64))
