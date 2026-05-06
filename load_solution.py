
import pickle
import evaluation

###     SETTINGS     ###

externalFunctionFolder = 'model_new'
# externalFunctionFolder = 'model_vecchio'
results_folder = 'results/'
model_path = 'opensimAD\\' + externalFunctionFolder  + '\model.osim'



full_set = ['gait_14','gait_15','gait_23',
            'gait_25','gait_60','gait_63']
            



# new:
# filename = '130326_1449.pickle' # only kine and mu
filename = '100326_1111.pickle' # all
# filename = '160326_1733.pickle' # only mech

# old: (Standard HC)
# filename = '050326_1028.pickle'
pickle_file = results_folder + filename

with open(pickle_file, "rb") as f:
        result = pickle.load(f)
        

evaluation.main_evaluation(result["solution"],externalFunctionFolder,full_set, save_fig=True)
print(result["solution"]["E"])
print(result["solution"]["c"])
print(result["solution"]["f"])
