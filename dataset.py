import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate

from tensorflow.keras.utils import Sequence
from scipy.spatial.transform import Rotation


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


def load_cea_dataset(imu_data_filename, gt_data, length=500):
    imu_data = pd.read_csv(imu_data_filename).values

    if imu_data.shape[0] >= length:
        gyro_data = imu_data[:length, 4:7]
        acc_data = imu_data[:length, 1:4]
    else:
        gyro_data, acc_data = np.zeros((length, 3)), np.zeros((length, 3))
        gyro_data[:imu_data.shape[0]] = imu_data[:length, 4:7]
        acc_data[:imu_data.shape[0]] = imu_data[:length, 1:4]
    return gyro_data, acc_data, gt_data


def force_quaternion_uniqueness(q):

    if np.absolute(q[3]) > 1e-05:
        if q[3] < 0:
            return -q
    elif np.absolute(q[0]) > 1e-05:
        if q[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q[1]) > 1e-05:
        if q[1] < 0:
            return -q
        else:
            return q
    else:
        if q[2] < 0:
            return -q
        else:
            return q


def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0