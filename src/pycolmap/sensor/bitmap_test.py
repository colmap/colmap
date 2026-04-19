import numpy as np

import pycolmap


def test_bitmap_rescale_filter_enum():
    assert {m.name: int(m) for m in pycolmap.BitmapRescaleFilter} == {
        "BILINEAR": 0,
        "BOX": 1,
    }


def test_bitmap_default_init():
    bitmap = pycolmap.Bitmap()
    assert bitmap is not None
    assert bitmap.is_empty


def test_bitmap_init_width_height_rgb():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_rgb


def test_bitmap_init_width_height_grey():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=False)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_grey


def test_bitmap_init_with_linear_colorspace():
    bitmap = pycolmap.Bitmap(
        width=64, height=48, as_rgb=True, linear_colorspace=True
    )
    assert bitmap.width == 64
    assert bitmap.height == 48


def test_bitmap_readonly_props():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    assert isinstance(bitmap.width, int)
    assert isinstance(bitmap.height, int)
    assert isinstance(bitmap.channels, int)
    assert isinstance(bitmap.is_rgb, bool)
    assert isinstance(bitmap.is_grey, bool)
    assert isinstance(bitmap.is_empty, bool)
    assert isinstance(bitmap.bits_per_pixel, int)
    assert isinstance(bitmap.pitch, int)
    assert bitmap.channels == 3
    assert bitmap.bits_per_pixel == 24
    assert not bitmap.is_empty


def test_bitmap_readonly_props_grey():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=False)
    assert bitmap.channels == 1
    assert bitmap.bits_per_pixel == 8
    assert bitmap.is_grey
    assert not bitmap.is_rgb


def test_bitmap_from_array_grey():
    array = np.zeros((48, 64), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_grey


def test_bitmap_from_array_rgb():
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    assert bitmap.width == 64
    assert bitmap.height == 48
    assert bitmap.is_rgb


def test_bitmap_to_array_roundtrip_grey():
    array_in = np.random.randint(0, 256, (48, 64), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array_in)
    array_out = bitmap.to_array()
    np.testing.assert_array_equal(array_in, array_out)


def test_bitmap_to_array_roundtrip_rgb():
    array_in = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array_in)
    array_out = bitmap.to_array()
    np.testing.assert_array_equal(array_in, array_out)


def test_bitmap_from_array_linear_colorspace():
    array = np.zeros((48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array, linear_colorspace=True)
    assert bitmap.is_rgb


def test_bitmap_clone():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    cloned = bitmap.clone()
    assert cloned.width == bitmap.width
    assert cloned.height == bitmap.height
    assert cloned.is_rgb == bitmap.is_rgb


def test_bitmap_clone_as_grey():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    grey = bitmap.clone_as_grey()
    assert grey.is_grey
    assert grey.width == bitmap.width
    assert grey.height == bitmap.height


def test_bitmap_clone_as_rgb():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=False)
    rgb = bitmap.clone_as_rgb()
    assert rgb.is_rgb
    assert rgb.width == bitmap.width
    assert rgb.height == bitmap.height


def test_bitmap_rescale_default_filter():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rescale(32, 24)
    assert bitmap.width == 32
    assert bitmap.height == 24


def test_bitmap_rescale_explicit_filter():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rescale(32, 24, filter=pycolmap.BitmapRescaleFilter.BOX)
    assert bitmap.width == 32
    assert bitmap.height == 24


def test_bitmap_rot90_once():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rot90(1)
    assert bitmap.width == 48
    assert bitmap.height == 64


def test_bitmap_rot90_twice():
    bitmap = pycolmap.Bitmap(width=64, height=48, as_rgb=True)
    bitmap.rot90(2)
    assert bitmap.width == 64
    assert bitmap.height == 48


def test_bitmap_write_read(tmp_path):
    array = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    bitmap = pycolmap.Bitmap.from_array(array)
    filepath = str(tmp_path / "test.png")
    bitmap.write(filepath)
    loaded = pycolmap.Bitmap.read(filepath, as_rgb=True)
    assert loaded is not None
    assert loaded.width == 64
    assert loaded.height == 48
    np.testing.assert_array_equal(loaded.to_array(), array)


def test_bitmap_set_jpeg_quality():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    bitmap.set_jpeg_quality(85)


def test_bitmap_exif_orientation():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_orientation()
    # Synthetic bitmap has no EXIF data, so expect None.
    assert result is None


def test_bitmap_exif_camera_model():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_camera_model()
    assert result is None


def test_bitmap_exif_focal_length():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_focal_length()
    assert result is None


def test_bitmap_exif_latitude():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_latitude()
    assert result is None


def test_bitmap_exif_longitude():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_longitude()
    assert result is None


def test_bitmap_exif_altitude():
    bitmap = pycolmap.Bitmap(width=32, height=24, as_rgb=True)
    result = bitmap.exif_altitude()
    assert result is None
