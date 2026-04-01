

import dataset as dataset
import numpy as np
from math import pi

def test_load():
    filePath = "IK\\IK_gait_14_filtered.mot"
    data, labels = dataset.loadData(filePath)
    assert data is not None, "Data should not be None"
    assert labels is not None, "Labels should not be None"

def test_convert_to_radians():
    filePath = "IK\\IK_gait_27_filtered.mot"
    data, labels = dataset.loadData(filePath)
    dataRadians = dataset.convertKineToRadians(labels, data)
    assert dataRadians is not None, "Data in radians should not be None"
    for i in range(1, dataRadians.shape[0]):
        for j in range(1, dataRadians.shape[1]):
            if labels[j] != "time":
                assert dataRadians[i,j] > -2*pi , f"Value at {i},{j} should be greater than -2*pi"
                assert dataRadians[i,j] < 2*pi , f"Value at {i},{j} should be less than 2*pi"

def test_reinterpolate():
    filePath = "GRF\\GRF_gait_60_filtered.mot"
    data, labels = dataset.loadData(filePath)
    indexTime = labels.index('time')
    time = data[:,indexTime]
    timeInterpolating = [0, 1, 2, 3, 4, 5,6]
    forcesInterpolated = dataset.reinterpolateGRF(timeInterpolating, time, data)
    assert forcesInterpolated is not None, "Reinterpolated forces should not be None"
    assert len(forcesInterpolated) == len(timeInterpolating), "Reinterpolated forces should match the length of interpolating time"

    

