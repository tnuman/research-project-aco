import os
import matplotlib.pyplot as plt
from main_milp import milp_solve
from classes import aco


# Runs an experiment. 
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
# def run_experiment(name, funcs_times_labels, nr_instances=13, nr_runs_to_average_over = 1):
#     path = os.path.join("solutions\experiments\\")
#     file = open(path + "\\" + name + ".txt", "a")
#
#     for function, time_limit, label in funcs_times_labels:
#         solutions_averaged = [0 for i in range(nr_instances)]
#         for i in range(nr_runs_to_average_over):
#             solutions = function(nr_instances, time_limit)
#             for instance, result in solutions:
#                 solutions_averaged[instance] += result/nr_runs_to_average_over
#         file.write(label + "\n")
#         file.write(str(solutions_averaged))
#         plt.plot([i for i in range(nr_instances)], solutions_averaged, marker='o', label=label)
#         file.write("\n")
#     file.close()
#     plt.ylabel("Lowest makespan found")
#     plt.xticks(range(0, nr_instances))
#     plt.xlabel("Instances")
#     plt.legend()
#     plt.savefig(path + "\\" + name + ".png")
def run_experiment(name, funcs_times_labels, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/" + name + ".txt", "a")
    solutions = list(map(lambda x : (x[0](nr_instances, x[1]), x[2]), funcs_times_labels))
    for solution in solutions:
        x_val = list(map(lambda x : x[0], solution[0]))
        y_val = list(map(lambda x : x[1], solution[0]))
        file.write(solution[1] + "\n")
        file.write(str(solution[0]) + "\n")
        plt.plot(x_val, y_val, marker='o', label= solution[1])
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")

def make_plot(solutions):
    for solution in solutions:
        x_val = list(map(lambda x : x[0], solution[0]))
        y_val = list(map(lambda x : x[1], solution[0]))
        plt.plot(x_val, y_val, marker='o', label= solution[1])
    path = os.path.join("solutions/experiments")
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, 13))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\plot.png")

# run_experiment("aco-milp-60-300-900",[
#     # (aco.aco_solve, 60, "ACO, limited to 1 minute"),
#     #                             (aco.aco_solve, 300, "ACO, limited to 5 minutes"),
#     #                             (aco.aco_solve, 900, "ACO, limited to 15 minutes"),
#                                   (milp_solve, 60, "MILP, limited to 1 minute"),
#                                   (milp_solve, 300, "MILP, limited to 5 minutes"),
#                                   (milp_solve, 900, "MILP, limited to 15 minutes"),
#                                 ])

milp60 = [[(0, 22.0), (1, 33.0), (2, 44.0), (3, 58.0), (4, 74.0), (5, 88.0), (6, 105.0),
           (7, 125.0), (8, 132.0), (9, 169.0), (10, 241.0)], "MILP, limited to 1 minute"]
milp300 = [[(0, 22.0), (1, 33.0), (2, 43.0), (3, 56.0), (4, 73.0), (5, 87.0), (6, 101.0),
            (7, 109.0), (8, 128.0), (9, 145.0), (10, 156.0), (11, 179.0), (12, 223.0)], "MILP, limited to 5 minutes"]
milp900 = [[(0, 22.0), (1, 33.0), (2, 43.0), (3, 55.0), (4, 71.0), (5, 87.0), (6, 100.0),
            (7, 109.0), (8, 128.0), (9, 138.0), (10, 156.0), (11, 178.0), (12, 200.0)], "MILP, limited to 15 minutes"]
aco60 = [[(0, 22), (1, 33), (2, 47), (3, 62), (4, 75), (5, 93), (6, 108), (7, 122), (8, 133), (9, 147), (10, 163), (11, 180), (12, 195)], "ACO, limited to 1 minute"]
aco300 = [[(0, 22), (1, 34), (2, 46), (3, 61), (4, 74), (5, 87), (6, 106), (7, 120), (8, 133), (9, 150), (10, 165), (11, 175), (12, 194)], "ACO, limited to 5 minutes"]
aco900 = [[(0, 22), (1, 33), (2, 46), (3, 61), (4, 73), (5, 90), (6, 99), (7, 114), (8, 129), (9, 147), (10, 162), (11, 179), (12, 191)], "ACO, limited to 15 minutes"]

make_plot([milp60, milp900, aco60, aco900])