# Plot mean of experimental GRFs trials
# Note: all the trials are reinterpolated to be of the same length (for visualization purposes)

# Add parent dir to sys for importing modules
import sys
from pathlib import Path
# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path
sys.path.append(str(parent_dir))

import pickle
import numpy as np
import dataset as dataset
import matplotlib.pyplot as plt
import evaluategrf as evaluategrf
import casadi as ca


## stuff to load to make function work
# consider making this into a function
# Kine





trialNames = (
    'gait_14', 'gait_15', 'gait_23', 'gait_25',
    'gait_27', 'gait_60', 'gait_61', 'gait_63','gait_64', 'gait_65')

GRF = []
for i in range(len(trialNames)):

    trialName = trialNames[i]
    pathKine =   'IK\\IK_' + trialName + '_filtered'+ '.mot'
    dataKine, labelsKine = dataset.loadData(pathKine)
    numberJointCoordinates = len(dataKine[0,:]) - 1
    dataKineRadians = dataset.convertKineToRadians(labelsKine, dataKine)
    indexTimeKine = labelsKine.index('time')
    timeKine = dataKineRadians[:,indexTimeKine]
    # Ground reaction forces
    pathGRF = 'GRF//GRF_' + trialName + '.mot'
    dataGRF, labelsGRF = dataset.loadData(pathGRF)
    indexTimeGRF = labelsGRF.index('time')
    timeGRF = dataGRF[:,indexTimeGRF]
    dataGRF = dataset.selectDesiredGRF(labelsGRF, dataGRF)
    dataGRF = dataset.reinterpolateGRF(timeKine, timeGRF, dataGRF)
    indexNoContact = np.nonzero(dataGRF[:,1])
    GRF.append(dataGRF[indexNoContact,:].squeeze())
    dataKineRadians = dataKineRadians[indexNoContact,:].squeeze()
    timeKine = timeKine[indexNoContact].squeeze()
    numberSamples = len(timeKine)
    Kine = {'time' : timeKine, 'labels': labelsKine, 'datakine': dataKineRadians}
    


linewidth = 0.2
fig, axs = plt.subplots(2,3)

sav = np.zeros([100,6,len(trialNames)])

for i in range(len(trialNames)):
    
    n = len(GRF[i])
    new_indices = np.linspace(0, n-1, 100)  # target positions
    to_plot = np.zeros([100,6])
    for j in range(6): to_plot[:,j] = np.interp(new_indices, np.arange(n), GRF[i][:, j])
    # ax.plot(stretched,color='gray',linewidth=0.3)
    
    axs[0,0].plot(to_plot[:,0],label='Experimental',linewidth=linewidth,color='grey')
    axs[0,1].plot(to_plot[:,1],label='Experimental',linewidth=linewidth,color='grey')
    axs[0,2].plot(to_plot[:,2],label='Experimental',linewidth=linewidth,color='grey')

    copx = to_plot[:,3] - to_plot[0,3]
    copz = to_plot[:,5] - to_plot[0,5]


    axs[1,0].plot(copx,label='Experimental',linewidth=linewidth,color='grey')
    axs[1,1].plot(to_plot[:,4],label='Experimental',linewidth=linewidth,color='grey')
    axs[1,2].plot(copz,label='Experimental',linewidth=linewidth,color='grey')
    
    to_plot[:,3] = copx
    to_plot[:,5] = copz

    sav[:,:,i] = to_plot

mean = np.mean(sav,axis=2)
std = np.std(sav,axis=2)

print(mean.shape)
print(std.shape)

axs[0,0].set_title('GRFx')
axs[0,0].grid()
axs[0,0].set_ylabel('force [N]')
axs[0,0].plot(mean[:,0],color='k')

axs[0,1].set_title('GRFy')
axs[0,1].grid()
axs[0,1].set_xlabel('time [s]')
axs[0,1].plot(mean[:,1],color='k')

axs[0,2].set_title('GRFz')
axs[0,2].grid()
axs[0,2].plot(mean[:,2],color='k')

axs[1,0].set_title('COP x')
axs[1,0].grid()
axs[1,0].set_ylabel('[m]')
axs[1,0].set_xlabel('time [s]')
axs[1,0].plot(mean[:,3],color='k')


axs[1,1].set_title('My')
axs[1,1].grid()
axs[1,1].set_ylabel('[N*m]')
axs[1,1].plot(mean[:,4],color='k')

# axs[1,1].set_xlabel('time [s]')
axs[1,2].set_title('COP z')
axs[1,2].grid()
axs[1,2].set_ylabel('[m]')
axs[1,2].set_xlabel('time [s]')
axs[1,2].plot(mean[:,5],color='k')


fig.tight_layout()

plt.show()

# externalFunctionFolder = 'model_smoothmin_last'

# ExternalFunction = ca.external('F','opensimAD' +'\\'  + externalFunctionFolder +'\\model.dll')


# folder = 'multistart_results/'




# name = 'cheb_2509.pickle' # latest sol gait 14

# file = open(folder + name, "rb")
# data = pickle.load(file)
# err = np.zeros((len(data)))

# for i in range(len(data)):
#     err[i] = data[i]['objectives']
# index_min = np.argmin(err)

# # barbatrucco per vedere la seconda miglior soluzione
# # err[index_min] = 30
# # i

# fig, ax = plt.subplots()
# # rescaling?
# ax.plot(err)# GRF
# ax.plot(index_min,err[index_min],'*')   
# ax.set_xlabel('Initial guess number')
# ax.set_ylabel('Objective function')
# fig.savefig('guesses.png',dpi=300)
# plt.show()

# # # print information
# print('minimum err:', err[index_min])
# print('corresponding initial guess:', data[index_min]['init'])
# sol = data[index_min]['solution']

# evaluategrf.printResultsFromParameters(sol,GRF,ExternalFunction,Kine)


# # Kine2 = {'time' : timeKine, 'labels': labelsKine, 'datakine': np.deg2rad(dataKine[:,1:])}
# # evaluategrf.printResultsFromParameters_crossval(sol,GRF,ExternalFunction,Kine2)
# # 