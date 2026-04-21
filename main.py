
import numpy as np
import evaluation
import pickle
import optimization.main
from datetime import datetime

def loso(full_set,init,problem_data):
    loso_summary = {}
    print(f"Start LOSO, full set:\n{full_set}\n")
    for i in range(len(full_set)):
        training_set = full_set.copy()
        del training_set[i]
        validation_set = [(full_set[i])]
        print(f"LOSO leave out {validation_set}")
        print(f"set {i+1} out of {len(full_set)}\n")

        # train all
        print(f"First training...\n")

       
        sol_for_validation = optimization.main.run_optimization(init,problem_data,training_set,externalFunctionFolder,mode='all')

        if not sol_for_validation["success"]:
            print(f"Skipping {full_set[i]} due to optimization failure")
            continue
        # before validating lets check what happens
        _ = evaluation.main_evaluation(sol_for_validation["solution"],externalFunctionFolder,training_set,plot=True)
        print(sol_for_validation["solution"]["E"])
        print(sol_for_validation["solution"]["c"])
        print(sol_for_validation["solution"]["f"])

        # for validating,if q is in the optimization, we must re-train the q, setting the other parameters from the previous solution
        print(f"Validation...\n")
        result = optimization.main.run_optimization(sol_for_validation["solution"],problem_data,validation_set,externalFunctionFolder,mode='only_mech_q_set')

        if not sol_for_validation["success"]:
            print(f"Skipping {full_set[i]} due to optimization failure")
            continue
        
        eval_out = evaluation.main_evaluation(result["solution"],externalFunctionFolder,validation_set,plot=True)

        loso_summary[full_set[i]]= {
            'sol' : result["solution"],
            'nrmse': eval_out["nrmse"],
            'grf': eval_out["computed_grf"][full_set[i]]
        }
      




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



# print(initial_guess)

# result = optimization.main.run_optimization(initial_guess,problem_data,full_set,externalFunctionFolder,'all')
# eval_out = evaluation.main_evaluation(result["solution"],externalFunctionFolder,full_set,plot=True)

# print(result["solution"]["E"])
# print(result["solution"]["c"])
# print(result["solution"]["f"])

# timestamp = datetime.now().strftime("%d%m%y_%H%M")
# filename = f"{timestamp}.pickle"

# with open(results_folder + filename, 'wb') as handle:
#     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# LOSO


loso(full_set,initial_guess,problem_data)