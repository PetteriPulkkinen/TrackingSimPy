from collections import namedtuple
import numpy as np


OutputData = namedtuple(
    'output_data',
    [
        'states',
        'innovations',
        'measurements',
        'predictions',
        'corrected',
        'update_indexes',
        'revisit_intervals'
     ])


def _pre_process_output_data(data):
    return OutputData(
        np.array(data.states),
        np.array(data.innovations),
        data.measurements,
        np.array(data.predictions),
        np.array(data.corrected),
        np.array(data.update_indexes),
        np.array(data.revisit_intervals)
    )


def simulation_loop(target, radar, tracker, update_policy, horizon):
    data = OutputData(*(list() for _ in range(7)))

    for t in range(horizon):
        tracker.predict()
        data.predictions.append(tracker.x.flatten())

        target.update()
        data.states.append(target.x.flatten())

        if update_policy.is_revisit():
            data.revisit_intervals.append(update_policy.get_revisit_interval())

            measurement = radar.measure(target, tracker)
            data.measurements.append(measurement)

            innovation = measurement.z - radar.H @ tracker.x
            data.innovations.append(innovation.flatten())

            tracker.update(measurement.z)
            data.corrected.append(tracker.x)

            update_policy.update(tracker, measurement)

            data.update_indexes.append(t)
        else:
            pass
        update_policy.roll_forward()

    return _pre_process_output_data(data)
