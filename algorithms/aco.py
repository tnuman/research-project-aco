from __future__ import annotations
from numpy.random import choice
from typing import List
import importlib
import sys
import time
import random
import copy

# Weights for respectively pheromone amount and visibility function
alpha = 1
beta = 0.4
# Pheromone evaporation coefficient
rho = 0.12


# t_0: Constant for initial amount of pheromones, q_0: Cutting exploration parameter
def run_solver(fileName, t_0 = 0.1, q_0 = 0.9, max_epochs_with_improvement = -1, time_limit = -1):
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dummy_start: OperationNode = OperationNode(None, None, [], is_dummy_start=True)
    # dummy_finish: OperationNode = OperationNode(None, None, is_dummy_finish=True)

    machines: list[Machine] = []
    for machine_number in mod.machines:
        changeOvers = {(k[1], k[2]): v for k, v in mod.changeOvers.items() if k[0] == machine_number}
        machines.append(Machine(machine_number, dummy_start, changeOvers))

    total_number_of_operations: int = 0
    jobs: list[Job] = []
    for i in range(mod.nr_jobs):
        job: Job = Job(i, mod.orders[i]['product'], mod.orders[i]['due'])
        jobs.append(job)
        current_node = OperationNode(job, mod.operations[i][0], [machines[machine_number] for machine_number in mod.machineAlternatives[(i, mod.operations[i][0])]],
                                     processing_time=mod.processingTimes[(i, mod.operations[i][0], mod.machineAlternatives[(i, mod.operations[i][0])][0])])
        current_node.previous_operation_of_job = dummy_start
        total_number_of_operations += 1
        job.first_unscheduled_operation = current_node
        j = 1
        while j < len(mod.operations[i]):
            previous_node = current_node
            current_node = OperationNode(job, mod.operations[i][j], [machines[machine_number] for machine_number in mod.machineAlternatives[(i, mod.operations[i][j])]],
                                         processing_time=mod.processingTimes[(i, mod.operations[i][j], mod.machineAlternatives[(i, mod.operations[i][j])][0])])
            total_number_of_operations += 1
            previous_node.next_operation_of_job = current_node
            current_node.previous_operation_of_job = previous_node
            j += 1

    return ant_colony_optimization(jobs, total_number_of_operations, t_0, q_0, max_epochs_with_improvement, time_limit)


class Job:
    def __init__(self, number: int, enzyme_type: str, due_date: int, first_unscheduled_operation: OperationNode = None) -> None:
        self.number = number
        self.enzyme_type = enzyme_type
        self.due_date = due_date
        self.first_unscheduled_operation = first_unscheduled_operation


class OperationNode:
    def __init__(self, job: Job, number: int, available_machines: list[Machine], previous_operation_of_job: OperationNode = None,
                 next_operation_of_job: OperationNode = None, processing_time: int = 0, is_dummy_start: bool = False,
                 is_dummy_finish: bool = False) -> None:
        self.job = job
        self.number = number
        self.is_dummy_start = is_dummy_start
        self.is_dummy_finish = is_dummy_finish
        self.completion_time = 0
        self.previous_operation_of_job = previous_operation_of_job
        self.processing_time = processing_time
        self.next_operation_of_job = next_operation_of_job
        self.available_machines = available_machines

    def __repr__(self):
        return "Node(%s, %s)" % (self.job, self.number)

    def __eq__(self, other):
        if isinstance(other, OperationNode):
            return (self.job == other.job and self.number == other.number)
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class Move():
    def __init__(self, o_from: OperationNode, o_to: OperationNode, machine: Machine):
        self.o_from = o_from
        self.o_to = o_to
        self.machine = machine
        self.start_time = -1

    def __hash__(self):
        return hash((self.o_from, self.o_to, self.machine))

    def __eq__(self, other):
        return self.o_from == other.o_from and self.o_to == other.o_to and self.machine == other.machine

    def visibility_function(self):
        self.start_time = self.o_to.previous_operation_of_job.completion_time
        if not (self.o_from.is_dummy_start):
            self.start_time = max(self.o_from.completion_time
                             + self.machine.change_over_times[(self.o_from.job.enzyme_type, self.o_to.job.enzyme_type)],
                             self.start_time)
        return 1/(self.start_time+1)


class Ant:
    def __init__(self, makespan = -1, time = -1) -> None:
        self.moves: list[Move] = []
        self.makespan = makespan
        self.time = time


class Machine:
    def __init__(self, number, dummy_start: OperationNode, change_over_times):
        self.number: int = number
        self.loading_sequence: list[OperationNode] = [dummy_start]
        self.change_over_times = change_over_times


def ant_colony_optimization(jobs: list[Job], total_number_of_operations: int, t_0, q_0, max_epochs_without_improvement = -1, time_limit = -1):
    start = time.time()
    pheromone_amounts: dict[Move, float] = {}
    population_size = len(jobs)
    epochs_without_improvement = 0
    best_ant_overall = Ant(makespan=sys.maxsize)

    # While “optimality condition is not satisfied” do
    # Options for optimality condition:
    #       1. certain number of epochs without improvement of the best solution
    #          (stability condition)
    #       2. time limit
    while ((max_epochs_without_improvement == -1 or epochs_without_improvement < max_epochs_without_improvement)
           and (time_limit == -1 or time.time() - start < time_limit)):
        best_ant_this_epoch = Ant(makespan=sys.maxsize)
        ants: list[Ant] = []

        # For each ant a = 1 to ps, place the ant on a randomly chosen operation O_ij0
        for i in range(population_size):
            ants.append(Ant())

        # For each ant a, a = 1 to ps do
        for ant in ants:
            # S_a <- ∅
            # O <- {O_ijr ∣ i=1,..,n,j=1,..,m,r=1,..,l_i}
            ant_jobs: list[Job] = copy.deepcopy(jobs)

            # For each w = 1 to [Σ_(i=1,..,n) l_i] do
            for w in range(total_number_of_operations):
                # 1. Initialization of Candidate Nodes: build the allowed list AL_w for the current
                # step w: AL_w <- {O_ijr ∈ O ∣ O_ijr-1 ∩ O = ∅}
                # (first unscheduled operation of job)
                allowed_list: List[OperationNode] = []
                for j in ant_jobs:
                    if j.first_unscheduled_operation is not None:
                        allowed_list.append(j.first_unscheduled_operation)

                # 2. Restriction: restriction of the allowed list by means of optimality criteria
                # (i.e. active or non-delay schedule); let the candidate list CL_w be the
                # restricted allowed list
                # First choice: CL_w = AL_w
                # TODO (potentially): implement (more complicated) restriction
                candidate_list: set[OperationNode] = allowed_list

                # 3. Initialization of Feasible Moves: mark as a feasible move each disjunctive arc
                # (O_i'jr', O_ijr) of E_j where O_ijr ∈ CL_w and O_i'jr' is the last operation of
                # the loading sequence of resource j (it creates the possibility for the candidate
                # operation to become the new last operation of that loading sequence)
                feasible_moves: list[Move] = []
                for o in candidate_list:
                    for machine in o.available_machines:
                        o_prime = machine.loading_sequence[-1]
                        feasible_moves.append(Move(o_prime, o, machine))

                # 4. Move Selection: select a feasible move (Oi'jr', Oijr) of E_j using transition
                # probability rule depending on pheromones τ and heuristic visibility function η
                # First choice for ...
                    # Transition probability rule: see Rossi's paper
                    # Local updating rule: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * τ_0
                    # Heuristic visibility function: Earliest Starting Time (EST) dispatching rule
                    # (Blum and Sampels, 2004) + incorporate change-over-times
                # TODO (potentially): implement (more complicated) rule(s)
                # Direct the related disjunctive arc (O_i'jr' = ‘dummy start node 0’ if r = 1)
                q = random.random()
                selected_move: Move = None
                if q <= q_0:
                    # Argmax
                    max_value: float = sys.float_info.min
                    for move in feasible_moves:
                        if not (move in pheromone_amounts):
                            pheromone_amounts[move] = t_0
                        value = pheromone_amounts[move]**alpha * move.visibility_function()**beta
                        if value > max_value:
                            max_value = value
                            selected_move = move
                else:
                    # Random proportional rule of AS
                    weights = []
                    for move in feasible_moves:
                        if not (move in pheromone_amounts):
                            pheromone_amounts[move] = t_0
                        value = pheromone_amounts[move]**alpha * move.visibility_function()**beta
                        weights.append(value)
                    summ = sum(weights)
                    weights = [w / summ for w in weights]
                    selected_move = choice(feasible_moves, p = weights)

                # Local pheromone update
                pheromone_amounts[selected_move] = (1-rho) * pheromone_amounts[selected_move] + rho * t_0

                # Add directed edge (O_i'jr', O_ijr)
                selected_move.machine.loading_sequence.append(selected_move.o_to)
                ant.moves.append(selected_move)

                # 5. Arcs Removal: remove all the remaining disjunctive arcs connected to O_i'jr'
                # (i.e. no other operation can be immediately subsequent to O_i'jr' in the loading
                # sequence)
                # Also remove all the remaining disjunctive arcs of E_h connected to O_ijr,
                # i.e. h ∈ Mij and h != j
                # (i.e. no other loading sequence can include the operation)
                # 6. Computing length: the length of the arc (O_i'jr', O_ijr) is evaluated as the
                # sum of processing and lag times of node O_ijr by means of expression
                # st(O_ijr) = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)}
                # 7. Transferring length: this length is placed on the related arc and on the arc
                # of the job routing (arc of A) which ends at O_ijr;
                # also, the completion time
                # t(O_ijr) = st(O_ijr) + t_ij = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)} + t_ij
                # is placed as a mark of the node O_ijr;
                start_time = selected_move.o_to.previous_operation_of_job.completion_time
                if not (selected_move.o_from.is_dummy_start):
                    start_time = max(selected_move.o_from.completion_time
                                     + selected_move.machine.change_over_times[(selected_move.o_from.job.enzyme_type, selected_move.o_to.job.enzyme_type)],
                                     start_time)
                selected_move.o_to.completion_time = start_time + selected_move.o_to.processing_time
                ant.makespan = max(ant.makespan, selected_move.o_to.completion_time)

                #8. Updating Structures: update O by removing operation O_ijr
                selected_move.o_to.job.first_unscheduled_operation = selected_move.o_to.next_operation_of_job
            # End for

            # At this point we have a schedule S_a (i.e. Acyclic Conjuctive Graph (ACG) with the
            # completion times of the operations), with makespan(S_a) = length of critical path
            # = length of longest path from dummy start 0 to dummy finishing operation *

            # TODO (potentially): Apply local search routine to S_a

            # Best this epoch evaluation: if (makespan(S_a) < makespan(S_best_this_epoch))
            # Then
                # makespan(S_best_this_epoch) = makespan(S_a)
                # S_best_this_epoch <- S_a
            # End if
            if (time_limit == -1 or time.time() - start < time_limit):
                if (ant.makespan < best_ant_this_epoch.makespan):
                    best_ant_this_epoch = ant
            else:
                break
        # End for

        # Global updating: For the schedule of the best ant of this epoch S_b, update pheromone for
        # all the moves of S_b
        # From Rossi papers: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * (1/makespan(S_b))
        if (time_limit == -1 or time.time() - start < time_limit):
            for move in best_ant_this_epoch.moves:
                pheromone_amounts[move] = (1-rho) * pheromone_amounts[move] + rho * (1/best_ant_this_epoch.makespan)

        # Total best evaluation: if (makespan(S_best_this_epoch) < makespan(S_total_best))
        # Then
            # makespan(S_total_best) = makespan(S_best_this_epoch)
            # S_total_best <- S_best_this_epoch
            # epoch <- 0
        # Else
            # epoch++
        # End if
        if (best_ant_this_epoch.makespan < best_ant_overall.makespan):
            best_ant_overall = best_ant_this_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    # End while

    # Final output:
        # The best schedule found S_total_best
            # Can be written to a CSV like milp_solution
        # Lowest makespan found = makespan(S_total_best)
            # Can be used as input data for the plot to compare with MILP (see main_experiments)
    if best_ant_overall.makespan == sys.maxsize:
        raise Exception("No solution found")
    best_ant_overall.time = time.time() - start
    return best_ant_overall