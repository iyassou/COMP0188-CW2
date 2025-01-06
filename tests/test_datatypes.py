import comp0188_cw2.datatypes as D

import h5py
import numpy as np
import pytest
import typing

N = 100

@pytest.fixture(scope="session")
def sample_chunk(tmp_path_factory):
    RNG = np.random.default_rng(seed=0xCAFE)
    data = {
        "actions": np.hstack((
            RNG.uniform(0, 1, size=(N, 3)).astype(np.float16),
            RNG.integers(
                low=0, high=len(D.GripperAction), size=(N, 1)
            ).astype(np.float16)
        )),
        
        "ee_cartesian_pos_ob": RNG.standard_normal(size=(N, 7)).astype(np.float16),

        "ee_cartesian_vel_ob": RNG.standard_normal(size=(N, 6)).astype(np.float16),
        
        "front_cam_ob": RNG.uniform(
            low=0, high=255 + np.finfo(np.float16).tiny, size=(N, 224, 224)
        ).astype(np.float16),
        
        "joint_pos_ob": RNG.normal(loc=0.8, scale=0.11, size=(N, 2)).astype(np.float16),

        "mount_cam_ob": RNG.uniform(
            low=0, high=255 + np.finfo(np.float16).tiny, size=(N, 224, 224)
        ).astype(np.float16),

        "terminals": RNG.integers(low=0, high=2, size=N),
    }
    expected_keys = set(typing.get_args(D._ObservationBaseAttribute))
    missing = expected_keys ^ data.keys()
    assert not missing, f"missing keys: {missing}"
    filepath = tmp_path_factory.mktemp("data") / "sample.h5"
    with h5py.File(filepath, "w") as f:
        for k, v in data.items():
            f[k] = v
    return D.Chunk(filepath)

def test_chunk_basic(sample_chunk):
    assert sample_chunk.n is None
    assert len(sample_chunk) == N
    assert sample_chunk.n == N

def test_chunk_getitem(sample_chunk):
    bad_index = slice(2, 1)
    with pytest.raises(IndexError):
        sample_chunk[bad_index]
    RNG = np.random.default_rng(seed=0xBEEF)
    M = 100
    starts = RNG.integers(0, N // 2, size=(M,))
    stops = RNG.integers(N // 2, N, size=(M,))
    steps = RNG.integers(low=1, high=N // 10, size=(M,))
    slices = [
        slice(start.item(), stop.item(), step.item())
        for start, stop, step in zip(starts, stops, steps)
    ]
    good_indices = (*list(range(N)), *slices)
    for index in good_indices:
        sample_chunk[index]
