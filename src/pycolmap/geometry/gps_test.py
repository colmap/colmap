import numpy as np

import pycolmap


def test_gps_transform_ellipsoid_enum():
    assert pycolmap.GPSTransfromEllipsoid.GRS80 is not None
    assert pycolmap.GPSTransfromEllipsoid.WGS84 is not None


def test_gps_transform_default_init():
    transform = pycolmap.GPSTransform()
    assert transform is not None


def test_gps_transform_init_with_wgs84():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    assert transform is not None


def test_gps_transform_ellipsoid_to_ecef():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    result = transform.ellipsoid_to_ecef(lat_lon_alt)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_ecef_to_ellipsoid():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    ecef = transform.ellipsoid_to_ecef(lat_lon_alt)
    result = transform.ecef_to_ellipsoid(ecef)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_ellipsoid_ecef_roundtrip():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    ecef = transform.ellipsoid_to_ecef(lat_lon_alt)
    roundtrip = transform.ecef_to_ellipsoid(ecef)
    np.testing.assert_array_almost_equal(
        roundtrip[0], lat_lon_alt[0], decimal=5
    )


def test_gps_transform_ellipsoid_to_enu():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    ref_lat = 47.0
    ref_lon = 8.0
    ref_alt = 500.0
    result = transform.ellipsoid_to_enu(lat_lon_alt, ref_lat, ref_lon, ref_alt)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_ecef_to_enu():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    ecef = transform.ellipsoid_to_ecef(lat_lon_alt)
    ref_ecef = ecef[0]
    result = transform.ecef_to_enu(ecef, ref_ecef)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_enu_to_ellipsoid():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    ref_lat = 47.0
    ref_lon = 8.0
    ref_alt = 500.0
    enu_coords = [np.array([0.0, 0.0, 0.0])]
    result = transform.enu_to_ellipsoid(enu_coords, ref_lat, ref_lon, ref_alt)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_enu_to_ecef():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    ref_lat = 47.0
    ref_lon = 8.0
    ref_alt = 500.0
    enu_coords = [np.array([100.0, 200.0, 50.0])]
    result = transform.enu_to_ecef(enu_coords, ref_lat, ref_lon, ref_alt)
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_gps_transform_ellipsoid_to_utm():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    utm_coords, zone = transform.ellipsoid_to_utm(lat_lon_alt)
    assert len(utm_coords) == 1
    assert utm_coords[0].shape == (3,)
    assert isinstance(zone, int)


def test_gps_transform_utm_to_ellipsoid():
    transform = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    lat_lon_alt = [np.array([47.0, 8.0, 500.0])]
    utm_coords, zone = transform.ellipsoid_to_utm(lat_lon_alt)
    is_north = lat_lon_alt[0][0] >= 0
    result = transform.utm_to_ellipsoid(utm_coords, zone, is_north)
    assert len(result) == 1
    np.testing.assert_array_almost_equal(result[0], lat_lon_alt[0], decimal=5)
