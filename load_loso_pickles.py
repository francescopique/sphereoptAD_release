import loso_plots
import pickle
import numpy as np
import matplotlib.pyplot as plt
results_folder = 'results/'


filename = '100426_1726.pickle'
pickle_file = results_folder + filename

with open(pickle_file, "rb") as f:
        loso_summary = pickle.load(f)

loso_plots.boxplot_loso_nrmse(loso_summary,save_fig=True)


nrmse = np.zeros([6,6])
for i, data in enumerate(loso_summary.values()):
        nrmse[i,:] = data['nrmse']

print('media nrmse folds')
print(np.round(nrmse.mean(axis=0)*100,2))
print('std nrmse fold')
print(np.round(nrmse.std(axis=0)*100,2))

loso_plots.plot_grf_kine_folds(loso_summary,save_fig=True)

loso_plots.loso_heatmap(loso_summary,save_fig=True)


loso_plots.plot_loso_parm_mean_std(loso_summary,save_fig=True)
