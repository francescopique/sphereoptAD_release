
import numpy as np
import evaluation
import pickle
import optimization.main
from datetime import datetime


bounds = {
    # standard HC model
    # 'E' : [1e5, 1e7],
    # 'c' : [0.2, 20],
    
    # new model
    'E' : [1.79e8, 1.79e10],
    'c' : [0.059, 5.9],
    
    'f' : [0.2, 0.8 ],
    'q': [2,2,2,3,2,3,3],
    'ty': [-0.005,0.005],

}
# weights for multiobjective
weights = [1/6,1/6,1/6,1/6,1/6,1/6]


###     SETTINGS     ###

externalFunctionFolder = 'model_new' 
# externalFunctionFolder = 'model_vecchio' 


results_folder = 'results/'
model_path = 'opensimAD\\' + externalFunctionFolder  + '\model.osim'
problem_data = optimization.main.build_problem_data(bounds,model_path,weights)


data = problem_data
scales = data["scales"]
numberSpheres = len(data["center"])
n_guesses = 100

initial_guess = {


    "E": 1.79e9, # raggio 32 mm
    "c": 0.59,

    # "E": np.random.uniform(low=data["E_lb"],high=data["E_ub"]),
    # "c": np.random.uniform(low=data["c_lb"],high=data["c_ub"]),
    "f": np.random.uniform(low=data["f_lb"],high=data["f_ub"]),

}

full_set = ['gait_14','gait_15','gait_23',
            'gait_25','gait_60','gait_63']



print(initial_guess)

# change 'only_mu' to select/deselect mechanical/kinematic variable optimization
result = optimization.main.run_optimization(initial_guess,problem_data,full_set,externalFunctionFolder,'only_mu')
eval_out = evaluation.main_evaluation(result["solution"],externalFunctionFolder,full_set,plot=True)

print(result["solution"]["E"])
print(result["solution"]["c"])
print(result["solution"]["f"])

timestamp = datetime.now().strftime("%d%m%y_%H%M")
filename = f"{timestamp}.pickle"

with open(results_folder + filename, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)