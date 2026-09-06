import numpy as np
import pytest

import pycolmap


def test_imu_calibration_defaults():
    calib = pycolmap.ImuCalibration()
    np.testing.assert_array_equal(calib.gyro_rectification, np.eye(3))
    np.testing.assert_array_equal(calib.accel_rectification, np.eye(3))
    assert calib.imu_rate > 0.0
    assert calib.gravity_magnitude > 0.0


def test_imu_measurement():
    measurement = pycolmap.ImuMeasurement(
        100, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    )
    assert measurement.timestamp == 100
    np.testing.assert_array_equal(measurement.gyro, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(measurement.accel, [4.0, 5.0, 6.0])


def test_imu_measurements_sorted_insert():
    ms = pycolmap.ImuMeasurements()
    assert ms.empty()
    zero = np.zeros(3)
    for t in [100, 300, 200, 500, 400]:
        ms.insert(pycolmap.ImuMeasurement(t, zero, zero))
    assert not ms.empty()
    assert len(ms) == 5
    # Stored sorted by timestamp.
    assert [ms[i].timestamp for i in range(len(ms))] == [
        100,
        200,
        300,
        400,
        500,
    ]


def test_imu_measurements_extract_in_range():
    ms = pycolmap.ImuMeasurements()
    zero = np.zeros(3)
    for t in [100, 200, 300, 400, 500]:
        ms.insert(pycolmap.ImuMeasurement(t, zero, zero))

    # Range brackets the edge: sample at/before t1 through sample at/after t2.
    sub = ms.extract_measurements_in_range(200, 400)
    assert [sub[i].timestamp for i in range(len(sub))] == [200, 300, 400]

    # t1 between samples pulls in the preceding sample.
    sub = ms.extract_measurements_in_range(150, 350)
    assert [sub[i].timestamp for i in range(len(sub))] == [100, 200, 300, 400]

    # A range beyond the available samples cannot bracket the edge.
    assert len(ms.extract_measurements_in_range(50, 300)) == 0


def test_imu_measurements_duplicate_raises():
    ms = pycolmap.ImuMeasurements()
    zero = np.zeros(3)
    ms.insert(pycolmap.ImuMeasurement(100, zero, zero))
    with pytest.raises(ValueError):
        ms.insert(pycolmap.ImuMeasurement(100, zero, zero))


def test_imu():
    imu = pycolmap.Imu()
    imu.imu_id = 1
    imu.camera_id = 2
    assert imu.imu_id == 1
    assert imu.camera_id == 2


def test_imu_state_accessors():
    state = pycolmap.ImuState(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    )
    np.testing.assert_array_equal(state.velocity, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(state.bias_gyro, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(state.bias_accel, [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(
        state.params, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )

    # Writing through an accessor updates the underlying params.
    state.velocity = np.array([10.0, 11.0, 12.0])
    np.testing.assert_array_equal(state.params[:3], [10.0, 11.0, 12.0])
