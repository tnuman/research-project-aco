import os
import matplotlib.pyplot as plt
from main_milp import milp_solve
from classes import aco


# Runs an experiment. 
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
def run_experiment(name, funcs_times_labels, nr_instances=13, nr_runs_to_average_over = 1):
    path = os.path.join("solutions\experiments\\")
    file = open(path + "\\" + name + ".txt", "a")


    for function, time_limit, label in funcs_times_labels:
        solutions_averaged = [0 for i in range(nr_instances)]
        for i in range(nr_runs_to_average_over):
            solutions = function(nr_instances, time_limit)
            for instance, result in solutions:
                solutions_averaged[instance] += result/nr_runs_to_average_over
        file.write(label + "\n")
        file.write(str(solutions_averaged))
        plt.plot([i for i in range(nr_instances)], solutions_averaged, marker='o', label=label)
        file.write("\n")
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")
    

run_experiment("aco-1-5-15-30-avg5",[(aco.aco_solve, 1, "ACO, limited to 1 second"),
                                (aco.aco_solve, 5, "ACO, limited to 5 seconds"),
                                (aco.aco_solve, 15, "ACO, limited to 15 seconds"),
                                (aco.aco_solve, 30, "ACO, limited to 30 seconds")
                                ], nr_runs_to_average_over=5)