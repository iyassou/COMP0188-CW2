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
        "actions": [
            np.concatenate((
                RNG.uniform(0, 1, size=(3,)).astype(np.float16),
                RNG.integers(
                    low=0, high=len(D.GripperAction), size=(1,)
                ).astype(np.float16)
            )) for _ in range(N)
        ],
        "ee_cartesian_pos_ob": [
            RNG.standard_normal(size=(7,)).astype(np.float16)
            for _ in range(N)
        ],
        "ee_cartesian_vel_ob": [
            RNG.standard_normal(size=(6,)).astype(np.float16)
            for _ in range(N)
        ],
        "front_cam_ob": [
            RNG.uniform(
                low=0, high=255 + np.finfo(np.float16).tiny,
                size=(224, 224)
            ).astype(np.float16) for _ in range(N)
        ],
        "joint_pos_ob": [
            RNG.normal(loc=0.8, scale=0.11, size=(2,)).astype(np.float16)
            for _ in range(N)
        ],
        "mount_cam_ob": [
            RNG.uniform(
                low=0, high=255 + np.finfo(np.float16).tiny,
                size=(224, 224)
            ).astype(np.float16) for _ in range(N)
        ],
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
    c = sample_chunk
    assert c.n is None
    assert len(c) == N
    assert c.n == N

def test_chunk_getitem(sample_chunk):
    c = sample_chunk
    bad_index = slice(2, 1)
    with pytest.raises(IndexError):
        c[bad_index]
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
        c[index]
