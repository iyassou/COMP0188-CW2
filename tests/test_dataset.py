import comp0188_cw2.dataset as D

import pytest

@pytest.mark.parametrize(
    "camera, expected",
    [
        (D.Camera.FRONT, True),
        (D.Camera.MOUNT, True),
        (D.Camera.FRONT | D.Camera.MOUNT, False),
    ]
)
def test_camera_single(camera, expected):
    actual = camera.single
    assert expected == actual

@pytest.mark.parametrize(
    "camera, expected",
    [
        (D.Camera.FRONT, "front"),
        (D.Camera.MOUNT, "mount"),
        (D.Camera.FRONT | D.Camera.MOUNT, "both"),
    ]
)
def test_camera_name(camera, expected):
    actual = camera.name
    assert expected == actual

if __name__ == '__main__':
    import inspect
    print(inspect.getsource(D))