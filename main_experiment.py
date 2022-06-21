import os
import matplotlib.pyplot as plt
from algorithms.aco import run_solver


# Runs an experiment. 
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
def run_experiment(name, funcs_times_labels, nr_instances=13, nr_runs_to_average_over=3):
    for function, time_limit, label in funcs_times_labels:
        solutions_averaged = [0 for _ in range(nr_instances)]
        for i in range(0, nr_instances):
            file_name = 'FJSP_' + str(i)
            makespan_sum = 0
            for j in range(nr_runs_to_average_over):
                try:
                    solution = run_solver(file_name, time_limit=time_limit)
                    makespan_sum += solution.makespan
                    path = os.path.join(f"solutions/aco/aco_solution_FJSP_{i}_{time_limit}s_{j}.csv")
                    file2 = open(path, "w")
                    file2.write("Machine,Job,Operation,Start,Completion\n")
                    for move in sorted(solution.moves, key=lambda x: (x.o_to.job.number, x.o_to.number)):
                        file2.write(f"{move.machine.number}, {move.o_to.job.number}, {move.o_to.number}, "
                                    f"{move.start_time}, {move.o_to.completion_time}\n")
                    file2.close()
                except Exception as e:
                    print(e)
                    pass
            average_makespan = makespan_sum/nr_runs_to_average_over
            solutions_averaged.append((i, average_makespan))
            print(f"({i}, {average_makespan})")
        # plt.plot([i for i in range(nr_instances)], solutions_averaged, marker='o', label=label)
        path = os.path.join("solutions\experiments\\")
        file = open(path + "\\" + name + ".txt", "a")
        file.write(label + "\n")
        file.write(str(solutions_averaged))
        file.write("\n")
        file.close()
    # plt.ylabel("Lowest makespan found")
    # plt.xticks(range(0, nr_instances))
    # plt.xlabel("Instances")
    # plt.legend()
    # plt.savefig(path + "\\" + name + ".png")

# def run_experiment(name, funcs_times_labels, nr_instances=13):
#     path = os.path.join("solutions/experiments")
#     file = open(path + "/" + name + ".txt", "a")
#     solutions = list(map(lambda x : (x[0](nr_instances, x[1]), x[2]), funcs_times_labels))
#     for solution in solutions:
#         x_val = list(map(lambda x : x[0], solution[0]))
#         y_val = list(map(lambda x : x[1], solution[0]))
#         file.write(solution[1] + "\n")
#         file.write(str(solution[0]) + "\n")
#         plt.plot(x_val, y_val, marker='o', label= solution[1])
#     file.close()
#     plt.ylabel("Lowest makespan found")
#     plt.xticks(range(0, nr_instances))
#     plt.xlabel("Instances")
#     plt.legend()
#     plt.savefig(path + "\\" + name + ".png")


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