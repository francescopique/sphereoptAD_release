import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import evaluation
import dataset
import seaborn as sns
from utils import interpolate_timeseries

def boxplot_loso_nrmse(loso_summary,save_fig=False):
    """
    Boxplot of LOSO NRMSE values.
    Assumes each loso_summary[key]['nrmse'] has 6 values:
    [grfx, grfy, grfz, copx, my, copz]
    """

    labels = ["GRFx", "GRFy", "GRFz", "COPx", "My", "COPz"]

    # collect NRMSEs across folds
    nrmse_matrix = np.array([
        v["nrmse"] for v in loso_summary.values()
    ])  # shape: (n_folds, 6)

    if nrmse_matrix.shape[1] != 6:
        raise ValueError("Expected 6 NRMSE values per LOSO fold")

    plt.figure(figsize=(3.45, 2.5)) 
    plt.boxplot(nrmse_matrix*100, patch_artist=True,tick_labels=labels)
    plt.ylabel("NRMSE")
    plt.grid()
    plt.tight_layout()
    if save_fig==True:
        plt.savefig("boxplot_nrmse_loso.png",dpi=300)
    plt.show()


def boxplot_loso_parms(loso_summary,parm="E"):
    """
    Boxplot of LOSO NRMSE values.
    Assumes each loso_summary[key]['nrmse'] has 6 values:
    [grfx, grfy, grfz, copx, my, copz]
    """
    parm_matrix = np.array([
        v["sol"][parm] for v in loso_summary.values()
    ]) 


    # plt.figure(figsize=(8, 5))
    plt.boxplot(parm_matrix, patch_artist=True)
    plt.title(parm)
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_grf_kine_folds(loso_summary, save_fig = False):

    _, grf_axs = plt.subplots(2,3)
    _, kine_axs = plt.subplots(1,4,figsize=(6,3))

    labels_q1 = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    idx_ty = labels_q1.index('pelvis_ty')
    
    evaluation_output = {}
    evaluation_output["computed_grf"] = {}
    evaluation_output["true_grf"] = {}
    evaluation_output["computed_q"] = {}
    evaluation_output["true_q"] = {}

    
    evaluation_output["computed_ty"]= {}
    evaluation_output["true_ty"] ={}

    for i in loso_summary:
        labelsKine, timeKine, dataKineRadians, GRF = dataset.process_data(i,downsamplingFactor=1)

        tmp = evaluation.extract_true_kinematics(labelsKine, dataKineRadians)
        q_true = tmp["q_r"]
        q1_true = tmp["q1"]

        if i in loso_summary[i]['sol']['kine']:
            computed_kine = loso_summary[i]['sol']['kine'][i]['q_r']
            temp = loso_summary[i]['sol']['kine'][i]['q1'] # angles are still in rad here
            evaluation_output["computed_ty"][i] =  temp[:,idx_ty]

        else:
            computed_kine = q_true
        computed_grf = loso_summary[i]['grf']

        fig_grf, grf_axs = evaluation.plot_grf(timeKine, GRF, computed_grf, axs=grf_axs)
        fig_kine, kine_axs = evaluation.plot_kine(timeKine, computed_kine, q_true, axs=kine_axs)

        evaluation_output["computed_grf"][i] = computed_grf
        evaluation_output["true_grf"] [i] = GRF
        evaluation_output["computed_q"][i] = np.rad2deg(computed_kine)
        evaluation_output["true_q"][i] = np.rad2deg(q_true)
 

        evaluation_output["true_ty"][i] = q1_true[:,idx_ty]


    fig_std_grf = evaluation.plot_mean_std_grf(evaluation_output["true_grf"], evaluation_output["computed_grf"])
    fig_std_kine = evaluation.plot_mean_std_kine(evaluation_output["true_q"],evaluation_output["computed_q"])

    # by chatty, fill last subplot with pelvis
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


    fig_std_grf.tight_layout()
    fig_std_grf.tight_layout()

    fig_grf.tight_layout()
    fig_kine.tight_layout()

    if save_fig == True:
        # fig_grf.savefig('grf_plt_LOSO.pdf')
        # fig_kine.savefig('kine_plt_LOSO.pdf')
        fig_std_grf.savefig('grf_plt_LOSO_std.png',dpi=300)
        fig_std_kine.savefig('kine_plt_LOSO_std.png',dpi=300)

    plt.show()

def loso_heatmap(loso_summary,save_fig=False):

    labels = ["GRFx", "GRFy", "GRFz", "COPx", "My", "COPz"]
    nrmse_matrix = np.array([
        loso_summary[key]["nrmse"] for key in loso_summary
    ])
    trialNames = np.array(list(loso_summary.keys()))
    plt.figure(figsize=(3.5,2.5))

    sns.heatmap(nrmse_matrix*100, xticklabels=labels,
                yticklabels=trialNames,cmap="crest",
                annot=True, fmt=".1f")
    plt.tight_layout()
    if save_fig == True:
        plt.savefig('loso_heatmap.png',dpi=300)
    plt.show()

def plot_loso_parm_mean_std(loso_summary,save_fig=False):

    E = []
    c = []
    f = []

    for key in loso_summary:
        parms = loso_summary[key]['sol']
        E.append(parms["E"])
        c.append(parms["c"])
        f.append(parms["f"])




    E_mean = np.mean(E,axis=0)
    c_mean = np.mean(c,axis=0)
    f_mean = np.mean(f,axis=0)

    E_std = np.std(E,axis=0)
    c_std = np.std(c,axis=0)
    f_std = np.std(f,axis=0)


    print(f"E mean: {E_mean}, std_dev: {E_std}")
    print(f"c mean: {c_mean}, std_dev: {c_std}")
    print(f"f mean: {f_mean}, std_dev: {f_std}")

    # fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig, axes = plt.subplots(1, 3,figsize=(3.5,2.5))
    
    rc('text', usetex=True)
    rc('font', family='serif')
    data   = [E, c, f]
    labels = ["K", "c", "$\mu_s$"]
    colors = ["#378ADD", "#1D9E75", "#D85A30"]
    ylabels = ["$Pa \times m^{-3/2}$", "s/m", ""]
    for ax, d, lbl, col in zip(axes, data, labels, colors):
        # bp = ax.boxplot(d, patch_artist=True, medianprops=dict(color=col, linewidth=2))
        bp = ax.boxplot(d,patch_artist=True)

        # bp["boxes"][0].set_facecolor(col + "44")
        # bp["boxes"][0].set_edgecolor(col)
        # for flier in bp["fliers"]:
            # flier.set(marker="o", color=col, markersize=4)
        ax.set_title(lbl)
        ax.set_xticks([])
        # ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.grid()

    axes[0].set_ylim([1.8e8,1.8e10])
    axes[0].set_ylabel("$Pa\cdot m^{-3/2}$")
    axes[1].set_ylim([0.059,2.5])
    axes[1].set_ylabel("$s/m$")

    axes[2].set_ylim([0,1])


    # fig.suptitle("LOSO parameter distributions", fontsize=13, y=1.01)
    fig.tight_layout()

    if save_fig:
        fig.savefig("loso_parms_boxplots.png",dpi=300, bbox_inches="tight")
    plt.show()

    