import numpy as np


def simulate(target, radar, revisit_interval_policy, tracker, horizon):
    states = np.zeros((horizon, target.n))
    prediction = np.zeros((horizon, target.n))
    corrections = np.zeros((horizon, target.n))
    measurements = np.zeros((horizon, radar.n))

    target.reset()
    tracker.x = target.x
    tracker.P = np.eye(target.n)
    for t in range(horizon):
        dt = revisit_interval_policy.choose()
        tracker.predict(F=target.F(dt), Q=target.Q(dt))
        prediction[t, :] = tracker.x.flatten()

        target.update(dt)
        states[t, :] = target.x.flatten()

        z, R = radar.measure(target, tracker.x)
        measurements[t, :] = z.flatten()

        tracker.update(z, H=radar.H, R=R)
        corrections[t, :] = tracker.x.flatten()

    data = dict(
        states=states,
        predictions=prediction,
        corrections=corrections,
        measurements=measurements
    )

    return data


def simulate_on_trajectory(trajectory, tracker, sensor, interval_policy):
    N = trajectory.shape[0]
    predictions = np.zeros(trajectory.shape)
    measurement_indexes = list()
    measurements = list()
    intervals = np.zeros(N)

    k_counter = 0
    for n in range(N):
        x = trajectory[n, :]
        tracker.predict()
        predictions[n, :] = tracker.x.flatten()
        z, R = sensor.measure(x, tracker.x.flatten())

        if k_counter >= interval_policy.K:
            K = interval_policy.update(x[0], z.flatten()[0], r=sensor.R(0)[0, 0])

            measurements.append(z.flatten())
            measurement_indexes.append(n)
            tracker.update(z, R=R)

            k_counter = 0
        else:
            k_counter += 1

        intervals[n] = interval_policy.K

    data = dict(
        predictions=predictions,
        measurement_indexes=np.array(measurement_indexes, dtype=int),
        measurements=np.array(measurements),
        intervals=intervals,
    )
    return data


def create_trajectory(target, dt, horizon):
    target.reset()
    trajectory = np.zeros((horizon, target.n))
    for n in range(horizon):
        target.update(dt)
        trajectory[n, :] = target.x.flatten()
    return trajectory
