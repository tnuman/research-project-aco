import importlib.util
from classes.milp import FlexibleJobShop
from classes.milp_utils import calculate_makespan

def run_solver(fileName, time=0):
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=mod.machineAlternatives, operations=mod.operations,
                          instance=fileName, changeOvers=mod.changeOvers, orders=mod.orders)
    return alg.build_model("solutions/milp/milp_solution_" + fileName + "_" + str(time) + "s" + '.csv', time)


# Set the objective function which will be plotted when running an experiment
def objective_function(s):
    return calculate_makespan(s)


def milp_solve(nr_instances, time):
    solution = []
    for i in range(0, nr_instances):
        file_name = 'FJSP_' + str(i)
        try:
            s = run_solver(file_name, time)
            m = objective_function(s)
            solution.append((i, m))
        except:
            # Currently: store nothing in case no feasible solution is found in the time limit
            pass
    return solution
