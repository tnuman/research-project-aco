import os, statistics
from algorithms.aco import run_solver


def aco_evaluate_hyperparameters(t_0, q_0, max_epochs_without_improvement=3, number_of_runs=3):
    for i in range(0, 13):
        file_name = 'FJSP_' + str(i)
        makespans = []
        times = []
        for j in range(0, number_of_runs):
            try:
                solution = run_solver(file_name, t_0, q_0, max_epochs_with_improvement=max_epochs_without_improvement)

                path = os.path.join(f"solutions/aco/aco_solution_FJSP_{i}_t0={t_0}_q0={q_0}_{j}.csv")
                file2 = open(path, "w")
                file2.write("Machine,Job,Operation,Start,Completion\n")
                for move in sorted(solution.moves, key=lambda x: (x.o_to.job.number, x.o_to.number)):
                    file2.write(f"{move.machine.number}, {move.o_to.job.number}, {move.o_to.number}, "
                                f"{move.start_time}, {move.o_to.completion_time}\n")
                file2.close()

                makespans.append(solution.makespan)
                times.append(solution.time)
            except Exception as e:
                print(e)

        makespan_average = round(sum(makespans)/number_of_runs, 1)
        time_average = round(sum(times)/number_of_runs, 1)
        makespan_stdev = round(statistics.stdev(makespans), 1)
        time_stdev = round(statistics.stdev(times), 1)

        file = open("results/aco-hyperparameter-evaluation/aco_hyperparameter_evaluation.txt", "a")
        file.write(f"t_0: {t_0}, q_0: {q_0}, instance: {i}, makespan: {makespan_average} += {makespan_stdev}, time: {time_average} += {time_stdev}\n")
        file.close()


t_0_values = [0.01, 0.05, 0.1, 0.5, 1]
q_0_values = [0.1, 0.3, 0.5, 0.7, 0.9]

for t_0 in t_0_values:
    q_0 = 0.3
    aco_evaluate_hyperparameters(t_0, q_0)

for q_0 in q_0_values:
    t_0 = 0.1
    aco_evaluate_hyperparameters(t_0, q_0)
