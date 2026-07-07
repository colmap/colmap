import numpy as np
import pytest

import pycolmap


def _ts(seconds: float) -> int:
    return pycolmap.timestamp_from_seconds(seconds)


def _make_integrator(method, t_end_seconds):
    options = pycolmap.ImuPreintegrationOptions()
    options.method = method
    calib = pycolmap.ImuCalibration()
    return pycolmap.ImuPreintegrator(
        options, calib, _ts(0.0), _ts(t_end_seconds)
    )


def _feed_constant(integrator, accel, gyro, num_steps, dt):
    for i in range(num_steps + 1):
        integrator.feed_imu(
            pycolmap.ImuMeasurement(
                _ts(i * dt), np.asarray(gyro), np.asarray(accel)
            )
        )


def test_integration_method_enum():
    assert (
        pycolmap.ImuIntegrationMethod.MIDPOINT
        != pycolmap.ImuIntegrationMethod.RK4
    )
    # String constructor is registered for the enum.
    assert (
        pycolmap.ImuIntegrationMethod("RK4")
        == pycolmap.ImuIntegrationMethod.RK4
    )


def test_preintegration_options_dataclass():
    options = pycolmap.ImuPreintegrationOptions()
    # Default method is RK4.
    assert options.method == pycolmap.ImuIntegrationMethod.RK4
    options.method = pycolmap.ImuIntegrationMethod.MIDPOINT
    options.integration_noise_density = 0.1
    options.max_condition_number = 1e6
    assert options.method == pycolmap.ImuIntegrationMethod.MIDPOINT
    assert options.integration_noise_density == pytest.approx(0.1)
    assert options.max_condition_number == pytest.approx(1e6)


def test_reintegration_options_dataclass():
    options = pycolmap.ImuReintegrationOptions()
    options.reintegrate_angle_norm_thres = 1e-3
    options.reintegrate_vel_norm_thres = 1e-3
    assert options.reintegrate_angle_norm_thres == pytest.approx(1e-3)
    assert options.reintegrate_vel_norm_thres == pytest.approx(1e-3)


@pytest.mark.parametrize(
    "method",
    [pycolmap.ImuIntegrationMethod.MIDPOINT, pycolmap.ImuIntegrationMethod.RK4],
)
def test_preintegration_constant_acceleration(method):
    num_steps, dt = 10, 0.01
    T = num_steps * dt
    integrator = _make_integrator(method, T)
    assert not integrator.has_started()

    accel = [0.0, 0.0, 9.81]
    gyro = [0.0, 0.0, 0.0]
    _feed_constant(integrator, accel, gyro, num_steps, dt)
    assert integrator.has_started()

    data = integrator.extract()
    assert data.delta_t == pytest.approx(T, abs=1e-9)
    # No rotation, no bias: delta_v = accel * T, delta_p = 0.5 * accel * T^2.
    assert data.delta_v[2] == pytest.approx(9.81 * T, abs=1e-9)
    assert data.delta_p[2] == pytest.approx(0.5 * 9.81 * T * T, abs=1e-9)

    # Covariance is finalized (SPD) and sqrt_information is populated.
    assert data.covariance.shape == (15, 15)
    assert data.sqrt_information.shape == (15, 15)
    assert np.linalg.eigvalsh(data.covariance).min() > 0.0


@pytest.mark.parametrize(
    "method",
    [pycolmap.ImuIntegrationMethod.MIDPOINT, pycolmap.ImuIntegrationMethod.RK4],
)
def test_feed_imu_measurements_batch(method):
    num_steps, dt = 10, 0.01
    T = num_steps * dt
    accel = [0.0, 0.0, 9.81]
    gyro = [0.0, 0.0, 0.0]

    # Feed one measurement at a time.
    integrator_single = _make_integrator(method, T)
    _feed_constant(integrator_single, accel, gyro, num_steps, dt)
    data_single = integrator_single.extract()

    # Feed the whole batch at once.
    integrator_batch = _make_integrator(method, T)
    ms = pycolmap.ImuMeasurements()
    for i in range(num_steps + 1):
        ms.insert(
            pycolmap.ImuMeasurement(
                _ts(i * dt), np.asarray(gyro), np.asarray(accel)
            )
        )
    integrator_batch.feed_imu(ms)
    data_batch = integrator_batch.extract()

    np.testing.assert_allclose(
        data_single.delta_p, data_batch.delta_p, atol=1e-12
    )
    np.testing.assert_allclose(
        data_single.delta_v, data_batch.delta_v, atol=1e-12
    )
    assert len(integrator_batch.measurements) == num_steps + 1


def test_reset():
    num_steps, dt = 5, 0.01
    integrator = _make_integrator(
        pycolmap.ImuIntegrationMethod.RK4, num_steps * dt
    )
    _feed_constant(
        integrator, [1.0, 2.0, 9.81], [0.1, -0.2, 0.05], num_steps, dt
    )

    integrator.reset()
    assert not integrator.has_started()
    data = integrator.extract()
    assert data.delta_t == pytest.approx(0.0)
    assert np.linalg.norm(data.delta_p) == pytest.approx(0.0)
    assert np.linalg.norm(data.delta_v) == pytest.approx(0.0)


def test_reintegrate_matches_fresh():
    num_steps, dt = 10, 0.01
    T = num_steps * dt
    accel = [1.0, 0.0, 9.81]
    gyro = [0.0, 0.1, 0.0]

    integrator = _make_integrator(pycolmap.ImuIntegrationMethod.RK4, T)
    _feed_constant(integrator, accel, gyro, num_steps, dt)
    data_before = integrator.extract()

    # Reintegrating with the same biases must reproduce the same result.
    integrator.reintegrate()
    data_after = integrator.extract()
    np.testing.assert_allclose(
        data_before.delta_p, data_after.delta_p, atol=1e-12
    )
    np.testing.assert_allclose(
        data_before.delta_v, data_after.delta_v, atol=1e-12
    )


def test_update_in_place():
    num_steps, dt = 10, 0.01
    integrator = _make_integrator(
        pycolmap.ImuIntegrationMethod.RK4, num_steps * dt
    )
    _feed_constant(
        integrator, [1.0, -0.5, 9.81], [0.05, 0.1, -0.02], num_steps, dt
    )

    data_extract = integrator.extract()
    data_update = pycolmap.PreintegratedImuData()
    integrator.update(data_update)
    np.testing.assert_allclose(
        data_extract.delta_p, data_update.delta_p, atol=1e-12
    )
    np.testing.assert_allclose(
        data_extract.sqrt_information, data_update.sqrt_information, atol=1e-10
    )


def test_set_linearization_biases_changes_result():
    num_steps, dt = 10, 0.01
    accel = [0.5, -0.3, 9.81]
    gyro = [0.1, -0.05, 0.02]

    integrator = _make_integrator(
        pycolmap.ImuIntegrationMethod.RK4, num_steps * dt
    )
    _feed_constant(integrator, accel, gyro, num_steps, dt)
    data_zero_bias = integrator.extract()

    # Setting nonzero biases and reintegrating must change the result.
    integrator.set_linearization_biases(
        np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )
    integrator.reintegrate()
    data_biased = integrator.extract()
    assert np.linalg.norm(data_biased.delta_v - data_zero_bias.delta_v) > 1e-6

    # The reintegrate(biases) overload is equivalent to the two calls above.
    integrator.reintegrate(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
    data_overload = integrator.extract()
    np.testing.assert_allclose(
        data_biased.delta_v, data_overload.delta_v, atol=1e-12
    )
