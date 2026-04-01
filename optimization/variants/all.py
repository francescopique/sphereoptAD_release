import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import dataset as dataset
from sklearn.preprocessing import MinMaxScaler
import osim_utilities as osim_utilities
from filters import filters_casadi
from filters import filters_scipy
from scipy import signal
import casadiUtilities
from utils import LSE




def define_opti_variables(opti,numberSpheres):
    vars = {}

    # VARIABLES
    vars["E"] = opti.variable(numberSpheres)
    vars["c"] = opti.variable(numberSpheres)
    vars["cx"] = opti.variable(numberSpheres)
    vars["cy"] = opti.variable(numberSpheres)
    vars["cz"] = opti.variable(numberSpheres)
    vars["f"] = opti.variable()
    vars["virtual_E"] = opti.variable()
    vars["virtual_c"] = opti.variable()

    # PARAMETERS
    vars["r"] = opti.parameter(numberSpheres)

    return vars

def apply_initials(opti,vars, data, init_params):

    scales = data["scales"]

    for i in range(len(data["center"])):
        opti.set_initial(vars["E"][i], init_params["E"][i] / scales["E"])
        opti.set_initial(vars["c"], init_params["c"][i]  / scales["c"])
    opti.set_initial(vars["f"], init_params["f"]  / scales["f"])
    opti.set_initial(vars["cx"], init_params["cx"]  / scales["cx"])
    opti.set_initial(vars["cy"], init_params["cy"]  / scales["cy"])
    opti.set_initial(vars["cz"], init_params["cz"]  / scales["cz"])

    # Hardcode the values of the radii
    radii = [0.02 for _ in range(len(data["center"]))]
    opti.set_value(vars["r"],radii)
    
    # opti.set_initial(vars["virtual_E"],  init_params["E"] / scales["E"])
    # opti.set_initial(vars["virtual_c"],  init_params["c"]  / scales["c"])

def apply_bounds(opti,vars,data):

    scales = data["scales"]
    opti.subject_to(opti.bounded(
        data["E_lb"]/scales["E"], vars["E"], data["E_ub"]/scales["E"]
    ))
    opti.subject_to(opti.bounded(
        data["c_lb"]/scales["c"], vars["c"], data["c_ub"]/scales["c"]
    ))
    opti.subject_to(opti.bounded(
        data["f_lb"]/scales["f"], vars["f"], data["f_ub"]/scales["f"]
    ))
    opti.subject_to(opti.bounded(
        data["cx_lb"]/scales["cx"], vars["cx"], data["cx_ub"]/scales["cx"]
    ))
    opti.subject_to(opti.bounded(
        data["cy_lb"]/scales["cy"], vars["cy"], data["cy_ub"]/scales["cy"]
    ))
    opti.subject_to(opti.bounded(
        data["cz_lb"]/scales["cz"], vars["cz"], data["cz_ub"]/scales["cz"]
    ))

    # opti.subject_to(opti.bounded(
    #     data["E_lb"]/scales["E"], vars["virtual_E"], data["E_ub"]/scales["E"]
    # ))
    # opti.subject_to(opti.bounded(
    #     data["c_lb"]/scales["c"], vars["virtual_c"], data["c_ub"]/scales["c"]
    # ))

def build_objective(opti,data,init_params,vars,trialNames,ExternalFunction,downsamplingFactor=1):
    ### Casadi Cost function
    errx, erry, errz, err_copx, err_My, err_copz = [0 for _ in range(6)]
    obj1, obj2, obj3, obj4, obj5, obj6 =  [0 for _ in range(6)]
  
    labels_q1 = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                'pelvis_tx', 'pelvis_ty', 'pelvis_tz','hip_flexion_r',
                'hip_adduction_r', 'hip_rotation_r' ]
    labels_q_r = ['knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']
    labels_q2 = ['hip_flexion_l', 'hip_adduction_l','hip_rotation_l']
    labels_q_l = ['knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'] 
    labels_q3 = ['lumbar_extension', 'lumbar_bending',
                'lumbar_rotation','arm_flex_r', 'arm_add_r','arm_rot_r',
                'elbow_flex_r' , 'arm_flex_l', 'arm_add_l',
                'arm_rot_l', 'elbow_flex_l','pro_sup_l','pro_sup_r']
    vars["q1"] = {}
    vars["q2"] = {}
    vars["q3"] = {}
    vars["q_r"] = {}
    vars["q_l"] = {}

    scales = data["scales"]

    for i in range(len(trialNames)):
        trialName = trialNames[i]
        labelsKine, timeKine, dataKineRadians, GRF = dataset.process_data(trialName,downsamplingFactor)

        numberSamples = len(timeKine)
        q1 = opti.parameter(numberSamples,len(labels_q1))
        q2 = opti.parameter(numberSamples,len(labels_q2))
        q3 = opti.parameter(numberSamples,len(labels_q3))

        # THIS CAN BE CHANGED TO VARIABLE (it depends)
        q_r = opti.variable(numberSamples, len(labels_q_r)) 
        q_l = opti.parameter(numberSamples, len(labels_q_l))


        vars["q1"][trialName] = q1
        vars["q2"][trialName] = q2
        vars["q3"][trialName] = q3
        vars["q_r"][trialName] = q_r
        vars["q_l"][trialName] = q_l

        idx_q1 = [labelsKine.index(i) for i in labels_q1]
        idx_q2 = [labelsKine.index(i) for i in labels_q2]
        idx_q3 = [labelsKine.index(i) for i in labels_q3]
        idx_q_r = [labelsKine.index(i) for i in labels_q_r]
        idx_q_l = [labelsKine.index(i) for i in labels_q_l]

        opti.set_value(q1, dataKineRadians[:,idx_q1])
        opti.set_value(q2, dataKineRadians[:,idx_q2])
        opti.set_value(q3, dataKineRadians[:,idx_q3])
        opti.set_initial(q_r, dataKineRadians[:,idx_q_r])
        opti.set_value(q_l, dataKineRadians[:,idx_q_l])

        # bound_vals = [3,2,3,3] # knee,ankle,subtalar,mtp
        bound_vals = data["q_bounds"]
        # BOUNDS Q
        for j in range(len(bound_vals)):
            val = np.deg2rad(bound_vals[j])
            qr_lb = dataKineRadians[:,idx_q_r[j]] - val
            qr_ub = dataKineRadians[:,idx_q_r[j]] + val
            opti.subject_to(opti.bounded(qr_lb, q_r[:,j], qr_ub))
            
        # butterworth filter parameters
        fc = 6  # cut off
        fs = 100/downsamplingFactor # sampling
        n = 2 # myfiltfilt only has butterworth order 2 
        b, a = signal.butter(n, fc/(fs/2))
        
        q_rfilt = ca.MX(q_r.shape[0], q_r.shape[1])

        for j in range(q_r.shape[1]):
            q_rfilt[:,j] = filters_casadi.myfiltfilt(b,a,q_r[:,j])

    
        dt =1/fs

        dq1 = casadiUtilities.derive(q1,dt)
        dq2 = casadiUtilities.derive(q2,dt)
        dq3 = casadiUtilities.derive(q3,dt)
        dq_r = casadiUtilities.derive(q_rfilt,dt)
        dq_l = casadiUtilities.derive(q_l,dt)

        scaler = MinMaxScaler()
        scaler.fit(GRF)

        for j in range(numberSamples):
            x1 = ca.MX(2*q1.shape[1],1)
            x1[::2] = q1[j,:]
            x1[1::2] = dq1[j,:]
            x2 = ca.MX(2*q2.shape[1],1)
            x2[::2] = q2[j,:]
            x2[1::2] = dq2[j,:]
            x3 = ca.MX(2*q3.shape[1],1)
            x3[::2] = q3[j,:]
            x3[1::2] = dq3[j,:]
            x_r = ca.MX(2*q_r.shape[1],1)
            x_r[::2] = q_rfilt[j,:]
            x_r[1::2] = dq_r[j,:]
            x_l = ca.MX(2*q_l.shape[1],1)
            x_l[::2] = q_l[j,:]
            x_l[1::2] = dq_l[j,:]
            
            vecInput = ca.vertcat(x1,x_r,x2,x_l,x3,
                                vars["E"]*scales["E"],
                                vars["c"]*scales["c"],
                                vars["r"],
                                vars["cx"]*scales["cx"],
                                vars["cy"]*scales["cy"],
                                vars["cz"]*scales["cz"],
                                vars["f"]*scales["f"])
            
            # prediction
            output = ExternalFunction(vecInput)

            ground_truth = GRF[j,:]
            
            # unscale
            output = (output - scaler.data_min_) / scaler.data_range_
            ground_truth = (ground_truth - scaler.data_min_)/scaler.data_range_    

            # squared diff
            errx += (output[0,:]-ground_truth[0])**2 
            erry += (output[1,:]-ground_truth[1])**2 
            errz += (output[2,:]-ground_truth[2])**2 
            err_copx += (output[3,:] - ground_truth[3])**2
            err_My += (output[4,:] - ground_truth[4])**2
            err_copz += (output[5,:] - ground_truth[5])**2

        # MSE
        obj1 += errx/numberSamples
        obj2 += erry/numberSamples
        obj3 += errz/numberSamples
        obj4 += err_copx/numberSamples
        obj5 += err_My/numberSamples
        obj6 += err_copz/numberSamples

    # weights = np.array((0.3,0.3,0.3,0.04,0.02,0.04))
    weights = np.array(data["weights"])
    weighted_objectives =  ca.vertcat(weights[0]*obj1,
          weights[1]*obj2,
          weights[2]*obj3,
          weights[3]*obj4,
          weights[4]*obj5,
          weights[5]*obj6)
    

    loss = LSE(weighted_objectives, t_LSE=500)

    # Regularization term (da identificazione)
    E_exp = 2.26e9/scales["E"]
    c_exp = 0.59/scales["c"]

    # t.c. il termine è 2 o.d.g. sotto il costo
    lam = 1e-2

    # le sfere sul tallone sono le 1-7 comprese
    # regTermStiff = ca.sum((sphereStiffness[:7] - E_exp)**2) + ca.sum((sphereStiffness[7:] - virtualStiffness)**2)
    # regTermDiss = ca.sum((sphereDissipation[:7] - c_exp)**2) + ca.sum((sphereDissipation[7:] - virtualDissipation)**2)

    regTermStiff = ca.sum((vars["E"][:7] - E_exp)**2) + 0.1*ca.sum((vars["E"][7:] - E_exp)**2)
    regTermDiss = ca.sum((vars["c"][:7] - c_exp)**2) + 0.1*ca.sum((vars["c"][7:] - c_exp)**2)
    to_min = loss + lam*(regTermStiff + regTermDiss)
    # to_min = loss
    return to_min

def filter_resulting_q(q_r,downsamplingFactor):

    # butterworth filter parameters
    fc = 6  # cut off
    fs = 100/downsamplingFactor # sampling
    n = 2 # myfiltfilt only has butterworth order 2 
    b, a = signal.butter(n, fc/(fs/2))

    for j in range(q_r.shape[1]):
        q_r[:,j] = filters_scipy.myfiltfilt(b,a,q_r[:,j])
        # q_l[:,j] = myfiltfilt(b,a,q_l[:,j])
    return q_r


def extract_solution(sol, vars, data, trialNames,downsamplingFactor):
    scales = data["scales"]

    kine_sol = {}

    for i in trialNames:
        kine_sol[i] = {
        "q_r": sol.value(vars["q_r"][i]),
        "q_l": sol.value(vars["q_l"][i]),
        "q1": sol.value(vars["q1"][i]),
        "q2": sol.value(vars["q2"][i]),
        "q3": sol.value(vars["q3"][i])
        }
        kine_sol[i]["q_r"] = filter_resulting_q(kine_sol[i]["q_r"],downsamplingFactor)
        
    
        


    return {
        "E": sol.value(vars["E"]) * scales["E"],
        "c": sol.value(vars["c"]) * scales["c"],
        "f": sol.value(vars["f"]) * scales["f"],
        "cx": sol.value(vars["cx"]) * scales["cx"],
        "cy": sol.value(vars["cy"]) * scales["cy"],
        "cz": sol.value(vars["cz"]) * scales["cz"],
        "r": sol.value(vars["r"]),
        "kine": kine_sol
    }


