
import numpy as np
import dataset
import evaluation
import opensim
import pickle
import matplotlib.pyplot as plt
import optimization.main
from datetime import datetime
import concurrent
from functools import partial



###     BOUNDS  ###

bounds = {
    'E' : [1e5, 1e7],
    'c' : [0.01, 10],
    # 'c' : [0.2, 20],
    
    'f' : [0.2, 0.8 ],
    'q': [2,2,2,3,2,3,3],
    'ty': [-0.005,0.005],

}
# weights for multiobjective
weights = [1/6,1/6,1/6,1/6,1/6,1/6]


###     SETTINGS     ###

externalFunctionFolder = 'model_vecchio' # idem ma vecchio modello




results_folder = 'results/'
model_path = 'opensimAD\\' + externalFunctionFolder  + '\model.osim'
problem_data = optimization.main.build_problem_data(bounds,model_path,weights)

# initial_guesses = build_initial_guesses(problem_data)

data = problem_data

initial_guess= {


    "E": 1e6,
    "c": 2,
    "f": 0.2,


}

full_set = ['gait_14','gait_15','gait_23',
            'gait_25','gait_60','gait_63']



ini = initial_guess
# print(ini)

# # NORMAL VERSION:
result = optimization.main.run_optimization(ini,problem_data,full_set,externalFunctionFolder,'all')
eval_out = evaluation.main_evaluation(result["solution"],externalFunctionFolder,full_set,plot=True)

print(result["solution"]["E"])
print(result["solution"]["c"])
print(result["solution"]["f"])

timestamp = datetime.now().strftime("%d%m%y_%H%M")
filename = f"{timestamp}.pickle"

with open(results_folder + filename, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

