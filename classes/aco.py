from __future__ import annotations

import importlib
import os
import sys
import time
from enum import IntEnum
import random
from numpy.random import choice
from collections import defaultdict

import copy

# Start pheromone amount
from typing import List

t_0 = 0.1
# MAX_EPOCHS_WITHOUT_IMPROVEMENT = 5
# POPULATION_SIZE = 20
# Cutting exploration parameter, between 0 and 1
q_0 = 0.3
# Weights for respectively pheromone amount and visibility function
alpha = 1
beta = 0.4
# Pheromone evaporation coefficient
rho = 0.12



# TODO: Implement better handling of time_limit
def run_solver(fileName, time=0):
    spec = importlib.util.spec_from_file_location('instance', "../instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
    #                       machineAlternatives=mod.machineAlternatives, operations=mod.operations, instance=fileName,
    #                       changeOvers=mod.changeOvers, orders=mod.orders)
    # return alg.build_model("solutions/milp/milp_solution_" + fileName + "_" + str(time) + "s"
    # + '.csv', time)

    dummy_start: OperationNode = OperationNode(None, None, [], is_dummy_start=True)
    # dummy_finish: OperationNode = OperationNode(None, None, is_dummy_finish=True)

    # operationTypes: list[OperationType] = []
    # # print(mod.machineAlternatives.keys())
    # for operation_number in mod.machineAlternatives.keys():
    #     # print(operation_number)
    #     operation_type = OperationType(operation_number)
    #     # print(operation_type)
    #     operationTypes.append(operation_type)
    #     for machine_number in mod.machineAlternatives[operation_number]:
    #         changeOvers = {(k[1], k[2]): v for k, v in mod.changeOvers.items() if k[0] == machine_number}
    #         operation_type.available_machines.append(Machine(machine_number, dummy_start, changeOvers))
    #     # print(operation_type.available_machines)

    machines: list[Machine] = []
    for machine_number in mod.machines:
        changeOvers = {(k[1], k[2]): v for k, v in mod.changeOvers.items() if k[0] == machine_number}
        machines.append(Machine(machine_number, dummy_start, changeOvers))

    # Group nodes per operation type/machine pool (same grouping in this case)
    # nodes_grouped_per_operation: list[list[OperationNode]] = [[], [], []]
    # edges: set[Edge]

    total_number_of_operations: int = 0
    jobs: list[Job] = []
    for i in range(mod.nr_jobs):
        job: Job = Job(i, mod.orders[i]['product'], mod.orders[i]['due'])
        jobs.append(job)
        current_node = OperationNode(job, mod.operations[i][0], [machines[machine_number] for machine_number in mod.machineAlternatives[(i, mod.operations[i][0])]],
                                     processing_time=mod.processingTimes[(i, mod.operations[i][0], mod.machineAlternatives[(i, mod.operations[i][0])][0])])
        current_node.previous_operation_of_job = dummy_start
        total_number_of_operations += 1
        # nodes_grouped_per_operation[mod.operations(i)[0]].append(current_node)
        job.first_unscheduled_operation = current_node
        # dummy_start.outgoing_directed_edges.append(DirectedEdge(dummy_start, current_node))
        # dummy_start.undirected_edges.append(UndirectedEdge())
        # edges.append(Edge(dummy_start, current_node), True)
        j = 1
        while j < len(mod.operations[i]):
            previous_node = current_node
            current_node = OperationNode(job, mod.operations[i][j], [machines[machine_number] for machine_number in mod.machineAlternatives[(i, mod.operations[i][j])]],
                                         processing_time=mod.processingTimes[(i, mod.operations[i][j], mod.machineAlternatives[(i, mod.operations[i][j])][0])])
            total_number_of_operations += 1
            # nodes_grouped_per_operation[mod.operations(i)[j]].append(current_node)
            # edges.append((previous_node, current_node), True)
            previous_node.next_operation_of_job = current_node
            current_node.previous_operation_of_job = previous_node
            j += 1
        # edges.append((current_node, dummy_finish), True)

    # Add undirected edges between pairs of operations to be processed on the same pool j
    # (replicate each as many times as the number of machines in this pool)
    # The cleaning over times are machine specific (also differ between machines in the same pool)
    # -> Can this be incorporated in weights on these edges, instead of weights on the nodes? (the
    # last is the case in Rossi's papers, where the sequence-dependent setup times do not
    # differ between machines in the same pool)
    # for node_set in nodes_grouped_per_operation:
    #     for i in range(len(node_set)):
    #         node = node_set[i]
    #         for j in range(i+1, len(node_set)):
    #             node2 = node_set[j]
    #             edge = UndirectedEdge((node, node2))
    # default_changeOvers = defaultdict(dict)
    # for k, v in mod.changeOvers.items():
    #     default_changeOvers[k] = v
    # print(default_changeOvers)

    best_ant = ant_colony_optimization(jobs, total_number_of_operations, mod.nr_jobs, time_limit=time)

    path = os.path.join("../solutions/aco/aco_solution_" + fileName + "_" + str(time) + "s" + '.csv')
    file = open(path, "w")
    file.write("Machine,Job,Operation,Start,Completion\n")
    for move in sorted(best_ant.moves, key = lambda x: (x.o_to.job.number, x.o_to.number)):
        file.write(f"{move.machine.number}, {move.o_to.job.number}, {move.o_to.number}, "
                   f"{move.start_time}, {move.o_to.completion_time}\n")
    file.close()
    return best_ant.makespan

# Input: a weighted digraph WDG = (N_F, A, E_jh, W_N, W_E) ----------------------------------------
#
#   N_F: partitioning of the set of nodes (all operations plus the dummy start and finishing
#        operations 0 and *) into the subsets of setup activities, where the operations which have
#        the same setup are included in the same subset (i.e. they are associated to the same kind
#        of node)
#
#   A: the set of conjunctive arcs (directed edges) between every pair of operations on a job
#      routing, between dummy start 0 and every first operation on a routing, and between every
#      last operation on a routing and dummy end *
#
#   E_jh: the set of disjunctive arcs (undirected edges) between pairs of operations, O_*j*, that
#         have to be processed on the same machine h (h = 1,...,m_j)
#         [m_j sometimes also referred as k_j] of the pool j (j = 1,...,m). It also includes
#         disjunctive arcs between 0 and O_*j*, between O_*j* and * and between 0 and *
#         for all E_jh
#         Each arc, connecting a pair of operations to be processed on the same pool j, is
#         replicated m_j times (as many times as the number of machines of pool j).
#
#   W_N: the weight on the nodes, represented by the two-dimensional array
#        W_N(O_ijr) = (t_ij, t(f_ij)),which includes, respectively, processing time and
#        sequence-dependent setup time
#
#   W_E: the weights on the disjunctive arcs. The weight on the disjunctive arcs (O_i'j'r', O_ijr)
#        of E_jh is represented by the two-dimensional array of n components each,
#        W_E(O_i'j'r', O_ijr) = (τ_p[O_i'j'r', O_ijr],τ_p[O_ijr, O_i'j'r']), which includes the
#        pheromone amount for selecting the feasible moves, respectively, O_i'j'r' -> O_ijr or
#        O_ijr -> O_i'j'r' in the position p of the loading sequence

# class ProductType():
#     def __init__(self, name: str, operation_processing_times: tuple) -> None:
#         self.name = name
#         self.operation_processing_times = operation_processing_times

class Job:
    def __init__(self, number: int, enzyme_type: str, due_date: int, first_unscheduled_operation: OperationNode = None) -> None:
        self.number = number
        self.enzyme_type = enzyme_type
        self.due_date = due_date
        self.first_unscheduled_operation = first_unscheduled_operation

# class OperationType:
#     # PREPERATION = 0
#     # FILTERING = 1
#     # RECEPTION = 2
#
#     def __init__(self, number: int):
#         self.number = number
#         self.available_machines = []

class OperationNode:
    def __init__(self, job: Job, number: int, available_machines: list[Machine], previous_operation_of_job: OperationNode = None,
                 next_operation_of_job: OperationNode = None, processing_time: int = 0,
                 # outgoing_directed_edges: list[DirectedEdge] = [], undirected_edges: dict[Machine, list[UndirectedEdge]] = [],
                 is_dummy_start: bool = False, is_dummy_finish: bool = False) -> None:
        self.job = job
        # self.operationType = operation_type
        # self.outgoing_directed_edges = outgoing_directed_edges
        # self.undirected_edges = undirected_edges
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

# class Edge:
#     def __init__(self, nodes: tuple[OperationNode, OperationNode]) -> None:
#         self.nodes = nodes
#
# class DirectedEdge(Edge):
#     def __init__(self, o_from: OperationNode, o_to: OperationNode, length: int = -1):
#         self.length = length
#         super.__init__(nodes)
#
# class UndirectedEdge(Edge):
#     def __init__(self, nodes: tuple[OperationNode, OperationNode], machine_number: int):
#         self.machine_number = machine_number
#         super.__init__(nodes)

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
        # print(selected_move.machine.change_over_times)
        if not (self.o_from.is_dummy_start):
            self.start_time = max(self.o_from.completion_time
                             + self.machine.change_over_times[(self.o_from.job.enzyme_type, self.o_to.job.enzyme_type)],
                             self.start_time)
        return 1/(self.start_time+1)
        # if self.o_from.is_dummy_start or self.o_to.is_dummy_finish or self.machine.change_over_times[(self.o_from.job.enzyme_type, self.o_to.job.enzyme_type)] == 0:
        #     change_over = 1
        # else:
        #     change_over = self.machine.change_over_times[(self.o_from.job.enzyme_type, self.o_to.job.enzyme_type)]
        # return 1/change_over

# class WeightedDisjunctiveGraph:
#     def __init__(self, edges: set[Edge], nodes: set[OperationNode], dummy_start: OperationNode, dummy_end: OperationNode) \
#             -> None:
#         self.edges = edges
#         self.nodes = nodes
#         self.dummy_start = dummy_start
#         self.dummy_end = dummy_end

class Ant:
    def __init__(self, makespan = -1) -> None:
        self.moves: list[Move] = []
        self.makespan = makespan

class Machine:
    def __init__(self, number, dummy_start: OperationNode, change_over_times):
        self.number: int = number
        self.loading_sequence: list[OperationNode] = [dummy_start]
        self.change_over_times = change_over_times

# Initialization ----------------------------------------------------------------------------------
def ant_colony_optimization(jobs: list[Job], total_number_of_operations: int, population_size, max_epochs_without_improvement = -1, time_limit: int = None):
    start = time.time()

    pheromone_amounts: dict[Move, float] = {}
    epochs_without_improvement = 0
    best_ant_overall = Ant(makespan=sys.maxsize)

    # Main loop -----------------------------------------------------------------------------------

    # While “optimality condition is not satisfied” do
    # -> First choice: certain number of epochs without improvement of the best solution
    # (stability condition)
    # TODO (potentially): implement better optimality condition
    # Also add option for time_limit: if the algorithm is still running when this is reached,
    # it is stopped and the best solution found up until now is returned
    while ((max_epochs_without_improvement == -1 or epochs_without_improvement < max_epochs_without_improvement)
           and (time_limit == None or time.time() - start < time_limit)):

        ants: list[Ant] = []

        # For each ant a = 1 to ps, place the ant on a randomly chosen operation O_ij0
        for i in range(population_size):
            ants.append(Ant())

        # Epoch Loop ---
        best_ant_this_epoch = Ant(makespan=sys.maxsize)

        # For each ant a, a = 1 to ps do
        for ant in ants:

            # Path Generation using List Scheduler (LS) Algorithm ---

            # S_a <- ∅
            # O <- {O_ijr ∣ i=1,..,n,j=1,..,m,r=1,..,l_i}
            # ant_nodes: set[OperationNode] = copy.deepcopy(wdg.nodes)
            ant_jobs: list[Job] = copy.deepcopy(jobs)

            # For each w = 1 to [Σ_(i=1,..,n) l_i] do
            for w in range(total_number_of_operations):

                # 1. Initialization of Candidate Nodes: build the allowed list AL_w for the current
                # step w: AL_w <- {O_ijr ∈ O ∣ O_ijr-1 ∩ O = ∅}
                # (first unscheduled operation of job)
                allowed_list: List[OperationNode] = []
                for j in ant_jobs:
                    if (j.first_unscheduled_operation != None):
                        allowed_list.append(j.first_unscheduled_operation)

                # print(ant_jobs)

                # allowed_list: set[OperationNode] \
                #     = filter(lambda o: OperationNode(o.job, o.operationType - 1) not in ant_nodes,
                #              ant_nodes)

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
                # print(candidate_list)
                for o in candidate_list:
                    for machine in o.available_machines:
                        o_prime = machine.loading_sequence[-1]
                        feasible_moves.append(Move(o_prime, o, machine))
                        #at the moment no distinction is made for on which actual machine in the machine pool the move is
                        #this could be incorporated in the heuristic visbility function, but maybe also in the pheromone map
                        #(key being [o_from, o_to, machine_number], instead of just [o_from, o_to]
                        #EDIT: now a distinction is made, because I think I actually have to
                    # feasible_moves.append(o_prime.undirected_edges[o])

                # feasible_moves: set[Edge] = []
                # for machine_pool in machine_pools:
                #     o_prime = machine_pool.loading_sequence[-1]
                #     feasible_moves.append(filter(lambda e: (not e.is_directed)
                #                                            and ((e.nodes[0] == o_prime
                #                                                  and e.nodes[1] in candidate_list)
                #                                                 or (e.nodes[0] in candidate_list
                #                                                     and e.nodes[1] == o_prime)),
                #                                  o_prime.outgoing_directed_edges))

                # 4. Move Selection: select a feasible move (Oi'jr', Oijr) of E_j using transition
                # probability rule depending on pheromones τ and heuristic visibility function η
                # First choice for ...
                    # Transition probability rule: see Rossi's paper
                    # Local updating rule: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * τ_0
                    # Heuristic visibility function: Earliest Starting Time (EST) dispatching rule
                    # (Blum and Sampels, 2004)
                    # EDIT: Even simpler and maybe more suitable for this specific problem: use change-over-times as heuristic
                # TODO (potentially): implement (more complicated) rule(s)
                # Direct the related disjunctive arc (O_i'jr' = ‘dummy start node 0’ if r = 1)
                q = random.random()
                selected_move: Move = None
                # print(feasible_moves)
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
                    # print(feasible_moves)
                    # print(weights)
                    summ = sum(weights)
                    weights = [w / summ for w in weights]
                    # print(weights)
                    selected_move = choice(feasible_moves, p = weights)

                # Local pheromone update
                pheromone_amounts[selected_move] = (1-rho) * pheromone_amounts[selected_move] + rho * t_0

                # Add directed edge (O_i'jr', O_ijr)
                # selected_move.o_from.outgoing_directed_edges.add(DirectedEdge(selected_move.o_from, selected_move.o_to))
                selected_move.machine.loading_sequence.append(selected_move.o_to)
                ant.moves.append(selected_move)

                # 5. Arcs Removal: remove all the remaining disjunctive arcs connected to O_i'jr'
                # (i.e. no other operation can be immediately subsequent to O_i'jr' in the loading
                # sequence)
                # selected_move.o_from.undirected_edges = []
                # Also remove all the remaining disjunctive arcs of E_h connected to O_ijr,
                # i.e. h ∈ Mij and h != j
                # (i.e. no other loading sequence can include the operation)
                # selected_move.o_to.undirected_edges = {selected_move.machine,
                #                                        selected_move.o_to.undirected_edges[selected_move.machine]}

                # 6. Computing length: the length of the arc (O_i'jr', O_ijr) is evaluated as the
                # sum of processing and lag times of node O_ijr by means of expression
                # st(O_ijr) = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)}
                # 7. Transferring length: this length is placed on the related arc and on the arc
                # of the job routing (arc of A) which ends at O_ijr;
                # UNNECESARRY???

                # also, the completion time
                # t(O_ijr) = st(O_ijr) + t_ij = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)} + t_ij
                # is placed as a mark of the node O_ijr;

                start_time = selected_move.o_to.previous_operation_of_job.completion_time
                # print(selected_move.machine.change_over_times)
                if not (selected_move.o_from.is_dummy_start):
                    start_time = max(selected_move.o_from.completion_time
                                     + selected_move.machine.change_over_times[(selected_move.o_from.job.enzyme_type, selected_move.o_to.job.enzyme_type)],
                                     start_time)
                selected_move.o_to.completion_time = start_time + selected_move.o_to.processing_time
                # print(selected_move.o_to)
                # print(selected_move.o_to.completion_time)
                ant.makespan = max(ant.makespan, selected_move.o_to.completion_time)
                # print(str(selected_move.o_to.job.number) + ", " + str(selected_move.o_to.operationType.number) + ": " + str(selected_move.o_to.completion_time))

                #8. Updating Structures: update O by removing operation O_ijr
                selected_move.o_to.job.first_unscheduled_operation = selected_move.o_to.next_operation_of_job

            # End for

            # Directing the remaining disjunctive arcs: connect these to the dummy end operation *
            # UNNECESARRY???

            # At this point we have a schedule S_a (i.e. Acyclic Conjuctive Graph (ACG) with the
            # completion times of the operations), with makespan(S_a) = length of critical path
            # = length of longest path from dummy start 0 to dummy finishing operation *

            # TODO (potentially): Apply local search routine to S_a

            # Best this epoch evaluation: if (makespan(S_a) < makespan(S_best_this_epoch))
            # Then
                # makespan(S_best_this_epoch) = makespan(S_a)
                # S_best_this_epoch <- S_a
            # End if
            # print(makespan)
            if (time_limit == None or time.time() - start < time_limit):
                if (ant.makespan < best_ant_this_epoch.makespan):
                    # print(makespan)
                    best_ant_this_epoch = ant
            else:
                break

        # End for

        # Global updating: For the schedule of the best ant of this epoch S_b, update pheromone for
        # all the moves of S_b
        # From Rossi papers: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * (1/makespan(S_b))
        # TODO (potentially): Implement better global updating rule
        if (time_limit == None or time.time() - start < time_limit):
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
    # print(best_ant_overall.makespan)
    if best_ant_overall.makespan == sys.maxsize:
        raise Exception("No solution found")
    return best_ant_overall

def aco_solve(nr_instances, time):
    solution = []
    for i in range(0, nr_instances):
        file_name = 'FJSP_' + str(i)
        try:
            makespan = run_solver(file_name, time)
            solution.append((i, makespan))
            print(f"({i}, {makespan})")
        except Exception as e:
            # Currently: store nothing in case no feasible solution is found in the time limit
            print(e)
            pass
    return solution

# for i in range(0, 13):
#     file_name = 'FJSP_' + str(i)
# run_solver('FJSP_0', time=5)

# aco_solve(13, time=60)
# aco_solve(13, time=300)
# aco_solve(13, time=900)