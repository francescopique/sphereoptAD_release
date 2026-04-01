import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import dataset
from utils import compute_rmse, interpolate_timeseries, LSE
from datetime import datetime
import os

def extract_true_kinematics(labelsKine, data):

    labels_q1 = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                'pelvis_tx', 'pelvis_ty', 'pelvis_tz', ]
    labels_q_r = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']
    labels_q_l = ['hip_flexion_l', 'hip_adduction_l','hip_rotation_l','knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'] 
    
    labels_q3 = ['lumbar_extension', 'lumbar_bending',
                'lumbar_rotation','arm_flex_r', 'arm_add_r','arm_rot_r',
                'elbow_flex_r' , 'arm_flex_l', 'arm_add_l',
                'arm_rot_l', 'elbow_flex_l','pro_sup_l','pro_sup_r']


    idx = lambda names: [labelsKine.index(n) for n in names]

    return {
        "q1": data[:, idx(labels_q1)],
        # "q2": data[:, idx(labels_q2)],
        "q3": data[:, idx(labels_q3)],
        "q_r": data[:, idx(labels_q_r)],
        "q_l": data[:, idx(labels_q_l)],
    }

def build_state_vector(q, dq):
    x = np.zeros(2 * q.shape[0])
    x[::2] = q
    x[1::2] = dq
    return x

def compute_GRF(solution, GRF, kine, dkine,ExternalFunction):
    
    errx = 0
    erry = 0 
    errz = 0
    err_copx = 0
    err_My = 0
    err_copz = 0
    GRF_sav = np.zeros((GRF.shape[0],6))

    scaler = MinMaxScaler()
    scaler.fit(GRF)

    for i in range(GRF.shape[0]):

   
        
        x1 = build_state_vector(kine["q1"][i], dkine["dq1"][i])
        # x2 = build_state_vector(kine["q2"][i], dkine["dq2"][i])
        x3 = build_state_vector(kine["q3"][i], dkine["dq3"][i])
        x_r = build_state_vector(kine["q_r"][i], dkine["dq_r"][i])
        x_l = build_state_vector(kine["q_l"][i], dkine["dq_l"][i])
  
        vecInput = np.concatenate((x1,x_r,x_l,x3,
                                    solution["E"]*np.ones(1),
                                    solution["c"]*np.ones(1),
                        
                                    solution["f"]*np.ones(1)))


        output = ExternalFunction(vecInput)

        ground_truth = GRF[i,:]
            
        # normalization
        output = (output - scaler.data_min_) / scaler.data_range_
        ground_truth = (ground_truth - scaler.data_min_)/scaler.data_range_    

        errx += (output[0,:]-ground_truth[0])**2 
        erry += (output[1,:]-ground_truth[1])**2 
        errz += (output[2,:]-ground_truth[2])**2 
        err_copx += (output[3,:] - ground_truth[3])**2
        err_My += (output[4,:] - ground_truth[4])**2
        err_copz += (output[5,:] - ground_truth[5])**2

        # denormalize
        output = output * scaler.data_range_ + scaler.data_min_
        ground_truth = ground_truth * scaler.data_range_ + scaler.data_min_
        GRF_sav[i,:] = output[:,:].full().flatten()

    single_errors = {
        "x" : errx,
        "y" : erry,
        "z" : errz,
        "copx": err_copx,
        "My": err_My,
        "copz": err_copz,
    }
    return GRF_sav, single_errors

def print_results(trialName,single_errors):
    print('\n', trialName)
    print('errx:', single_errors["x"])
    print('erry: ',single_errors["y"])
    print('errz: ',single_errors["z"])
    print('err cop x: ',single_errors["copx"])
    print('err My: ',single_errors["My"])
    print('err cop z: ',single_errors["copz"])

def print_nrmse(truth, computed):

    print('NRMSE %: ')
    len = truth.shape[1]
    nrmse = np.empty(len)
    for i in range(len):
        nrmse[i] = compute_rmse(truth[:,i],computed[:,i])
        print(np.round(nrmse[i],3)*100)
    
    return nrmse
    
def plot_grf(time, GRF_true, GRF_pred, axs = None):


    created_fig = False

    if axs is None:
        fig, axs = plt.subplots(2, 3)
        created_fig = True
    else:
        fig = axs[0, 0].figure
    
    titles = ["GRFx", "GRFy", "GRFz", "COPx", "My", "COPz"]

    GRF_true, time = interpolate_timeseries(GRF_true)
    GRF_pred, _ = interpolate_timeseries(GRF_pred)

    for i, ax in enumerate(axs.flat):
        ax.plot(time, GRF_true[:, i], label="Experimental", color = 'b', linewidth = 1)
        ax.plot(time, GRF_pred[:, i], label="Predicted", color = 'r', linewidth = 1)
        ax.set_title(titles[i])
        ax.grid()

    axs[0,0].set_ylabel('force [N]')
    axs[1,1].set_xlabel('% gait cycle')
    axs[1,0].set_ylabel('[m]')
    axs[1,1].set_ylabel('[N*m]')
    axs[1,2].set_ylabel('[m]')

    
    if created_fig:
        fig.legend(["Experimental", "Predicted"], loc="lower center", ncol=2)
        fig.tight_layout()

    return fig, axs
  
def plot_kine(time, q_pred, q_true, axs=None):
    created_fig = False

    if axs is None:
        fig, axs = plt.subplots(1,4)
        created_fig = True
    else:
        fig = axs[0].figure
    titles = ["knee","ankle","subtalar","mtp"]

    
    q_true, time = interpolate_timeseries(q_true)
    q_pred, _ = interpolate_timeseries(q_pred)

    for i, ax in enumerate(axs.flat):
        ax.plot(time, q_true[:, i], label="Experimental",color = 'b', linewidth = 1)
        ax.plot(time, q_pred[:, i], label="Predicted",color = 'r', linewidth = 1)

        ax.set_title(titles[i])
        ax.grid()

    axs[0].set_ylabel('deg')
    axs[1].set_xlabel('% gait cycle')

    if created_fig:
        fig.legend(["Experimental", "Predicted"], loc="lower center", ncol=2)
        fig.tight_layout()
    return fig, axs


def plot_offset(time, q_pred, q_true, ax=None):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure
    
    q_true, time = interpolate_timeseries(q_true)
    q_pred, _ = interpolate_timeseries(q_pred)

    ax.plot(time, q_true, label="Experimental",color = 'b', linewidth = 1)
    ax.plot(time, q_pred, label="Predicted",color = 'r', linewidth = 1)

    ax.set_title('pelvis_ty')
    ax.grid()

    ax.set_ylabel('[m]')
    ax.set_xlabel('% gait cycle')

    if created_fig:
        fig.legend(["Experimental", "Predicted"], loc="lower center", ncol=2)
        fig.tight_layout()
    return fig, ax

def plot_mean_std_grf(true,computed):
    """ 
    plots mean value and +- SD with shaded areas of true value and computed (GRF)
    true and computed are dicts
    """
    x = 100
    y = 6
    z = len(true)  # for example
    true_sav = np.zeros((x,y,z))
    computed_sav = np.zeros((x,y,z))

    for i,value in enumerate(true.values()):
        tmp_val, time = interpolate_timeseries(value)
        true_sav[:,:,i] = tmp_val
        true_sav[:,3,i] -= true_sav[0,3,i]
        true_sav[:,5,i] -= true_sav[0,5,i]

    for i,value in enumerate(computed.values()):
        tmp_val, time = interpolate_timeseries(value)
        computed_sav[:,:,i] = tmp_val
        computed_sav[:,3,i] -= computed_sav[0,3,i]
        computed_sav[:,5,i] -= computed_sav[0,5,i]

    
    true_mean = true_sav.mean(axis=2)
    true_std = true_sav.std(axis=2)
    computed_mean = computed_sav.mean(axis=2)
    computed_std = computed_sav.std(axis=2)

    fig, axs = plt.subplots(2,3)

    titles = ['GRFx','GRFy','GRFz','COPx','COPy','COPz']
    for i,ax in enumerate(axs.flat):
        ax.plot(time,true_mean[:,i],color='black')
        ax.fill_between(
            time,
            true_mean[:,i] - true_std[:,i],
            true_mean[:,i] + true_std[:,i],
            color='gray',
            alpha=0.3,
            # label='± SD'
        )
        ax.plot(time,computed_mean[:,i],color='red')
        ax.fill_between(
            time,
            computed_mean[:,i] - computed_std[:,i],
            computed_mean[:,i] + computed_std[:,i],
            color='red',
            alpha=0.3,
            # label='± SD'
        )
        ax.set_title(titles[i])

    axs[0,0].set_ylabel('Force [N]')
    axs[1,0].set_ylabel('[m]')
    axs[1,1].set_ylabel('[Nm]')
    axs[1,2].set_ylabel('[m]')
    axs[1,1].set_xlabel('% gait cycle')


    fig.tight_layout()
    return fig

def plot_mean_std_kine(true,computed,save_fig=False):
    """ 
    plots mean value and +- SD with shaded areas of true value and computed q
    true and computed are dicts
    """
    x = 100
    y = 7
    z = len(true)  # for example
    true_sav = np.zeros((x,y,z))
    computed_sav = np.zeros((x,y,z))

    for i,value in enumerate(true.values()):
        tmp_val, time = interpolate_timeseries(value)
        true_sav[:,:,i] = tmp_val

    for i,value in enumerate(computed.values()):
        tmp_val, time = interpolate_timeseries(value)
        computed_sav[:,:,i] = tmp_val

    
    true_mean = true_sav.mean(axis=2)
    true_std = true_sav.std(axis=2)
    computed_mean = computed_sav.mean(axis=2)
    computed_std = computed_sav.std(axis=2)

    fig, axs = plt.subplots(2,4)

    titles = ['hip flexion','hip adduction','hip rotation','knee flexion','ankle angle','subtalar angle','mtp angle']
    # for i,ax in enumerate(axs.flat):
    for ax, i in zip(axs.flat, range(true_mean.shape[1])): # by chatty
        ax.plot(time,true_mean[:,i],color='black')
        ax.fill_between(
            time,
            true_mean[:,i] - true_std[:,i],
            true_mean[:,i] + true_std[:,i],
            color='gray',
            alpha=0.3,
            # label='± SD'
        )
        ax.plot(time,computed_mean[:,i],color='red')
        ax.fill_between(
            time,
            computed_mean[:,i] - computed_std[:,i],
            computed_mean[:,i] + computed_std[:,i],
            color='red',
            alpha=0.3,
            # label='± SD'
        )
        ax.set_title(titles[i])
        # ax.set_xlabel('% gait cycle')
    axs[0,0].set_ylabel('[deg]')
    axs[0,1].set_ylabel('[deg]')
    axs[0,2].set_ylabel('[deg]')
    axs[0,3].set_ylabel('[deg]')
    axs[1,0].set_ylabel('[deg]')
    axs[1,1].set_ylabel('[deg]')
    axs[1,2].set_ylabel('[deg]')
    axs[1,1].set_xlabel('                              % gait cycle')
    fig.tight_layout()

    return fig
    
def main_evaluation(solution, externalFunctionFolder, trialNames,plot=True,save_fig=False):

    labels_q1 = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    idx_ty = labels_q1.index('pelvis_ty')

    ExternalFunction = ca.external('F','opensimAD' +'\\'  + externalFunctionFolder +'\\model.dll')
    downsamplingFactor = 1
    fs = 100
    dt = (1/fs) / downsamplingFactor
    grf_fig, grf_axs = plt.subplots(2,3)
    q_fig, q_axs = plt.subplots(1,4,figsize=(6,3))
    off_fig, off_axs = plt.subplots() 

    nrmse_total = np.zeros(6)
    evaluation_output = {}
    evaluation_output["computed_grf"] = {}
    evaluation_output["true_grf"] = {}
    evaluation_output["computed_q"] = {}
    evaluation_output["true_q"] = {}
    evaluation_output["computed_ty"]= {}
    evaluation_output["true_ty"] = {}
    # for test only:
    obj1, obj2, obj3, obj4, obj5, obj6 =  [0 for _ in range(6)]

    for i in range(len(trialNames)):
        trialName = trialNames[i]
        labelsKine, timeKine, dataKineRadians, GRF = dataset.process_data(trialName,downsamplingFactor)

        q_true = extract_true_kinematics(labelsKine, dataKineRadians)
        dq_true = {
        "dq1" : np.gradient(q_true["q1"],dt,axis=0),
        # "dq2" : np.gradient(q_true["q2"],dt,axis=0),
        "dq3" : np.gradient(q_true["q3"],dt,axis=0),
        "dq_r" : np.gradient(q_true["q_r"],dt,axis=0),
        "dq_l" : np.gradient(q_true["q_l"],dt,axis=0)}

        if trialName in solution["kine"]:
            q_computed = solution["kine"][trialName]
            dq_computed = {
            "dq1" : np.gradient(q_computed["q1"],dt,axis=0),
            # "dq2" : np.gradient(q_computed["q2"],dt,axis=0),
            "dq3" : np.gradient(q_computed["q3"],dt,axis=0),
            "dq_r" : np.gradient(q_computed["q_r"],dt,axis=0),
            "dq_l" : np.gradient(q_computed["q_l"],dt,axis=0)}
        else:
            q_computed = q_true
            dq_computed = dq_true


        computed_GRF, single_errors = compute_GRF(solution, GRF, q_computed, dq_computed, ExternalFunction)

        print_results(trialName,single_errors)
        # nrmse_total = nrmse_total + print_nrmse(GRF, computed_GRF)
        nrmse_total = nrmse_total + print_nrmse(GRF, computed_GRF)

        grf_fig, grf_axs = plot_grf(timeKine, GRF, computed_GRF, axs=grf_axs)

        q_fig, q_axs = plot_kine(timeKine, np.rad2deg(q_computed["q_r"]), np.rad2deg(q_true["q_r"]), axs=q_axs)

        off_fig, off_axs = plot_offset(timeKine, q_computed["q1"][:,idx_ty],q_true["q1"][:,idx_ty], ax = off_axs)
        # dq = np.rad2deg(dkine["dq_r"])
        # dq_fig, dq_axs = plot_kine(timeKine, dq, dq)

        evaluation_output["computed_grf"][trialName] = computed_GRF
        evaluation_output["true_grf"] [trialName] = GRF
        evaluation_output["computed_q"][trialName] = np.rad2deg(q_computed["q_r"])
        evaluation_output["true_q"][trialName] = np.rad2deg(q_true["q_r"])
        
        evaluation_output["computed_ty"][trialName] = q_computed["q1"][:,idx_ty]
        evaluation_output["true_ty"][trialName] = q_true["q1"][:,idx_ty]


    evaluation_output["nrmse"] = nrmse_total
    fig_std_grf = plot_mean_std_grf(evaluation_output["true_grf"], evaluation_output["computed_grf"])
    fig_std_kine = plot_mean_std_kine(evaluation_output["true_q"],evaluation_output["computed_q"])
    # fig_std_off = plot_mean_std_kine(evaluation_output["true_ty"],evaluation_output["computed_ty"])

    # BY chatty:
    # --- Fill the empty subplot (axs[1,3]) with pelvis_ty mean ± std ---
    x = 100
    z = len(evaluation_output["true_ty"])
    true_ty_sav = np.zeros((x, z))
    computed_ty_sav = np.zeros((x, z))

    for i, value in enumerate(evaluation_output["true_ty"].values()):
        tmp_val, time_ty = interpolate_timeseries(value.reshape(-1, 1))
        true_ty_sav[:, i] = tmp_val[:, 0]

    for i, value in enumerate(evaluation_output["computed_ty"].values()):
        tmp_val, _ = interpolate_timeseries(value.reshape(-1, 1))
        computed_ty_sav[:, i] = tmp_val[:, 0]

    true_ty_mean = true_ty_sav.mean(axis=1)
    true_ty_std  = true_ty_sav.std(axis=1)
    computed_ty_mean = computed_ty_sav.mean(axis=1)
    computed_ty_std  = computed_ty_sav.std(axis=1)

    ax_ty = fig_std_kine.axes[-1]  # last (empty) subplot
    ax_ty.plot(time_ty, true_ty_mean, color='black')
    ax_ty.fill_between(time_ty,
                    true_ty_mean - true_ty_std,
                    true_ty_mean + true_ty_std,
                    color='gray', alpha=0.3)
    ax_ty.plot(time_ty, computed_ty_mean, color='red')
    ax_ty.fill_between(time_ty,
                    computed_ty_mean - computed_ty_std,
                    computed_ty_mean + computed_ty_std,
                    color='red', alpha=0.3)
    ax_ty.set_title('pelvis_ty')
    ax_ty.set_ylabel('[m]')
    fig_std_kine.tight_layout()

    print('\nmean NRMSE over trials')
    for i in range(len(nrmse_total)):
        mean = nrmse_total[i]/len(trialNames)
        print(np.round(mean*100,2))
    
    if plot==True:
        plt.show()

    if save_fig==True:

        grf_fig.tight_layout()
        q_fig.tight_layout()
        # grf_fig.savefig('grf_plt.pdf')
        # q_fig.savefig('kine_plt.pdf')

        fig_std_grf.savefig('grf_std.png', dpi = 300)
        fig_std_kine.savefig('kine_std.png', dpi = 300)

    plt.close(grf_fig)
    plt.close(q_fig)
    plt.close(off_fig)

    plt.close(fig_std_grf)
    plt.close(fig_std_kine)
    # plt.close(fig_std_off)

    return evaluation_output