import pycolmap


def test_track_element_default_init() -> None:
    element = pycolmap.TrackElement()
    assert element is not None


def test_track_element_init_with_args() -> None:
    element = pycolmap.TrackElement(image_id=1, point2D_idx=5)
    assert element.image_id == 1
    assert element.point2D_idx == 5


def test_track_element_image_id_readwrite() -> None:
    element = pycolmap.TrackElement()
    element.image_id = 10
    assert element.image_id == 10


def test_track_element_point2d_idx_readwrite() -> None:
    element = pycolmap.TrackElement()
    element.point2D_idx = 20
    assert element.point2D_idx == 20


def test_track_default_init() -> None:
    track = pycolmap.Track()
    assert track is not None


def test_track_length() -> None:
    track = pycolmap.Track()
    assert track.length() == 0


def test_track_add_element() -> None:
    track = pycolmap.Track()
    track.add_element(image_id=1, point2D_idx=0)
    assert track.length() == 1


def test_track_add_elements() -> None:
    track = pycolmap.Track()
    elements = [
        pycolmap.TrackElement(image_id=1, point2D_idx=0),
        pycolmap.TrackElement(image_id=2, point2D_idx=1),
    ]
    track.add_elements(elements)
    assert track.length() == 2


def test_track_element_access() -> None:
    track = pycolmap.Track()
    track.add_element(image_id=5, point2D_idx=10)
    element = track.element(0)
    assert element.image_id == 5
    assert element.point2D_idx == 10


def test_track_set_element() -> None:
    track = pycolmap.Track()
    track.add_element(image_id=1, point2D_idx=0)
    new_element = pycolmap.TrackElement(image_id=99, point2D_idx=88)
    track.set_element(0, new_element)
    assert track.element(0).image_id == 99
    assert track.element(0).point2D_idx == 88


def test_track_delete_element() -> None:
    track = pycolmap.Track()
    track.add_element(image_id=1, point2D_idx=0)
    track.add_element(image_id=2, point2D_idx=1)
    track.delete_element(0)
    assert track.length() == 1


def test_track_reserve() -> None:
    track = pycolmap.Track()
    track.reserve(100)
    assert track.length() == 0


def test_track_compress() -> None:
    track = pycolmap.Track()
    track.reserve(100)
    track.add_element(image_id=1, point2D_idx=0)
    track.compress()
    assert track.length() == 1


def test_track_elements_readwrite() -> None:
    track = pycolmap.Track()
    track.add_element(image_id=1, point2D_idx=0)
    track.add_element(image_id=2, point2D_idx=1)
    elements = track.elements
    assert len(elements) == 2
    new_elements = [pycolmap.TrackElement(image_id=10, point2D_idx=20)]
    track.elements = new_elements
    assert track.length() == 1
