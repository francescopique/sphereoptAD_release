# Utilities for handling, selecting, preprocessing experimental data (GRF and kinematic).
# There are some tiny tests (compatible with pytest) in the test folder.
# Francesco 05/2025

from opensim_sto_reader import readMotionFile
import numpy as np

def loadData(filePath):
    _, labels, data = readMotionFile(filePath)
    data = np.array(data)
    return data, labels

def convertKineToRadians(labels, data):
    # We select only angular coordinates, i.e. not pelvis x,y,z positions
    labelsAngularCoordinates = [ 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                'hip_flexion_r','hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
                'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 
                'hip_flexion_l', 'hip_adduction_l','hip_rotation_l',
                'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
                'mtp_angle_l', 'lumbar_extension', 'lumbar_bending',
                'lumbar_rotation','arm_flex_r', 'arm_add_r','arm_rot_r',
                'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l',
                'arm_rot_l', 'elbow_flex_l', 'pro_sup_l']
    indexLabelsAngularCoordinates = (
        [labels.index(i) for i in labelsAngularCoordinates])
    datarad = data.copy()
    datarad[:,indexLabelsAngularCoordinates] = (
        np.deg2rad(data[:,indexLabelsAngularCoordinates]))
    return datarad
    
def convertKineToDegrees(labels, data):
    # We select only angular coordinates, i.e. not pelvis x,y,z positions
    labelsAngularCoordinates = [ 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                'hip_flexion_r','hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
                'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 
                'hip_flexion_l', 'hip_adduction_l','hip_rotation_l',
                'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
                'mtp_angle_l', 'lumbar_extension', 'lumbar_bending',
                'lumbar_rotation','arm_flex_r', 'arm_add_r','arm_rot_r',
                'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l',
                'arm_rot_l', 'elbow_flex_l', 'pro_sup_l']
    indexLabelsAngularCoordinates = (
        [labels.index(i) for i in labelsAngularCoordinates])
    datadeg = data.copy()
    datadeg[:,indexLabelsAngularCoordinates] = (
        np.rad2deg(data[:,indexLabelsAngularCoordinates]))
    return datadeg

def selectDesiredGRF(labels,data):
    labelsDesiredQuantities = [
        '1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz',
        '1_ground_force_px',  '1_ground_torque_y','1_ground_force_pz',]
    indexLabelsDesiredQuantities = (
        [labels.index(i) for i in labelsDesiredQuantities])
    forces = data[:, indexLabelsDesiredQuantities]
    return forces

def reinterpolateGRF(interpolatingTime, time, forces):
    numberTimeSamples = len(interpolatingTime)
    numberForceComponents = forces.shape[1]
    forcesInterpolated = np.zeros((numberTimeSamples, numberForceComponents))
    for i in range(numberForceComponents):
        forcesInterpolated[:,i] = np.interp(interpolatingTime, time, forces[:,i])
    return forcesInterpolated

def downsample(data, factor):
    return data[::factor]


def process_data(trialName, downsamplingFactor):
    pathKine =   'IK\\IK_' + trialName + '_filtered'+ '.mot'
    dataKine, labelsKine = loadData(pathKine)
    dataKineRadians = convertKineToRadians(labelsKine, dataKine)

    indexTimeKine = labelsKine.index('time')
    timeKine = dataKineRadians[:,indexTimeKine]

    # Ground reaction forces
    pathGRF = 'GRF//GRF_' + trialName + '.mot'
    dataGRF, labelsGRF = loadData(pathGRF)
    indexTimeGRF = labelsGRF.index('time')
    timeGRF = dataGRF[:,indexTimeGRF]
    dataGRF = selectDesiredGRF(labelsGRF, dataGRF)

    
    timeKine = downsample(timeKine, downsamplingFactor)
    dataKineRadians = downsample(dataKineRadians, downsamplingFactor)
    dataGRF = reinterpolateGRF(timeKine, timeGRF, dataGRF)

    indexNoContact = np.nonzero(dataGRF[:,1])
    GRF = dataGRF[indexNoContact,:].squeeze()
    dataKineRadians = dataKineRadians[indexNoContact,:].squeeze()
    timeKine = timeKine[indexNoContact].squeeze()

    


    return labelsKine, timeKine, dataKineRadians, GRF