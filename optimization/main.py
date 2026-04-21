import time
import json
import casadi as ca
import osim_utilities
import numpy as np

def build_problem_data(bounds, model_path,weights):
    center = osim_utilities.getCentersFromModel(model_path)

    data = {
        "weights": weights,
        "center": center,

        "E_lb": bounds['E'][0],
        "E_ub": bounds['E'][1],
        "c_lb": bounds['c'][0],
        "c_ub": bounds['c'][1],
        "f_lb": bounds['f'][0],
        "f_ub": bounds['f'][1],
        "q_bounds": bounds['q'],
        "scal_q": 2*np.array(bounds['q']), # symmetrical bounds
        "ty_lb": bounds['ty'][0],
        "ty_ub": bounds['ty'][1],
    }

    data["scales"] = {
        "E": data["E_ub"] - data["E_lb"],
        "c": data["c_ub"] - data["c_lb"],
        "f": data["f_ub"] - data["f_lb"],
        "ty": data["ty_ub"] - data["ty_lb"],
    }

    return data

def solve_problem(opti, ipopt_options):

    opti.solver("ipopt", ipopt_options)
    return opti.solve()


def run_optimization(init_params,data, trialNames, externalFunctionFolder,mode ='all'):

    if mode=='only_mech_q_set':
        import optimization.variants.only_mech_q_set as variant
    # elif mode == 'only_mu':
    #     import optimization.variants.only_mu_alt_scal_q as variant
    elif mode=='all':
        import optimization.variants.all as variant
    # elif mode=='no_q':
    #     import optimization.variants.no_q as variant
    # elif mode=='no_q_set':
    #     import optimization.variants.no_q_set as variant
    # elif mode=='only_q_set':
    #     import optimization.variants.only_q_set as variant
    # elif mode=='only_K_c':
    #     import optimization.variants.only_K_c as variant
    else:
        print('Wrong mode type!')
        return 1

    numberSpheres = len(data["center"])
    opti = ca.Opti()

    opti_vars = variant.define_opti_variables(opti,numberSpheres)
    variant.apply_initials(opti, opti_vars, data, init_params)
    variant.apply_bounds(opti, opti_vars, data)
    ExternalFunction = ca.external('F','opensimAD' +'\\'  + externalFunctionFolder +'\\model.dll')

    downsamplingFacotr = 1
    objective = variant.build_objective(opti, data, init_params, opti_vars, trialNames, ExternalFunction,downsamplingFacotr,)
    opti.minimize(objective)

    with open('parameters/ipopt_options.json', 'r') as f:
        ipopt_options = json.load(f)

    try:
        tic = time.time()
        sol = solve_problem(opti, ipopt_options)
        toc = time.time()
        print("Elapsed time:", toc - tic, "seconds")        
        print("Optimization succeeded.")
    except Exception as e:        
        print("Optimization failed:", e)
        return {
            "success": False,
            "solution": None,
            "objective": None,
            "error": e
        }
    else:
        extracted_solution = variant.extract_solution(sol, opti_vars, data,trialNames,downsamplingFacotr)
  
        return { 
            "success": True,
            "solution": extracted_solution,
            "objective": sol.value(objective)
            }