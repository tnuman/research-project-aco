import importlib
import sys
import time
from enum import IntEnum
from random import random
from __future__ import annotations
import copy

# Start pheromone amount
t_0 = 0.1
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10
POPULATION_SIZE = 10
# Cutting exploration parameter, between 0 and 1
q_0 = 0.5
# Weights for respectively pheromone amount and visibility function
alpha = 1
beta = 1

# TODO: Implement better handling of time_limit
def run_solver(fileName, time=0):
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
    #                       machineAlternatives=
    #                       mod.machineAlternatives, operations=mod.operations, instance=fileName,
    #                       changeOvers=mod.changeOvers, orders=mod.orders)
    # return alg.build_model("solutions/milp/milp_solution_" + fileName + "_" + str(time) + "s"
    # + '.csv', time)
    dummy_start: OperationNode = OperationNode(None, None, is_dummy_start=True)
    dummy_finish: OperationNode = OperationNode(None, None, is_dummy_finish=True)
    # Group nodes per operation type/machine pool (same grouping in this case)
    nodes_grouped_per_operation: list[list[OperationNode]] = [[], [], []]
    # edges: set[Edge]
    for i in range(mod.nr_jobs):
        current_node = OperationNode(Job(mod.orders[i]['product'], mod.orders[i]['due']),
                                     mod.operations(i)[0])
        nodes_grouped_per_operation[mod.operations(i)[0]].append(current_node)
        dummy_start.outgoing_directed_edges.append(DirectedEdge(dummy_start, current_node))
        # edges.append(Edge(dummy_start, current_node), True)
        j = 1
        while j < len(mod.operations(i)):
            previous_node = current_node
            current_node = OperationNode(Job(mod.orders[i]['product'], mod.orders[i]['due']),
                                         mod.operations(i)[j])
            nodes_grouped_per_operation[mod.operations(i)[j]].append(current_node)
            # edges.append((previous_node, current_node), True)
            j += 1
        # edges.append((current_node, dummy_finish), True)

    # TODO: Add undirected edges between pairs of operations to be processed on the same pool j
    # (replicate each as many times as the number of machines in this pool)
    # The cleaning over times are machine specific (also differ between machines in the same pool)
    # -> Can this be incorporated in weights on these edges, instead of weights on the nodes? (the
    # last is the case in Rossi's papers, where the sequence-dependent setup times do not
    # differ between machines in the same pool)
    for node_set in nodes_grouped_per_operation:
        for i in range(len(node_set)):
            node = node_set[i]
            for j in range(i+1, len(node_set)):
                node2 = node_set[j]
                edge = UndirectedEdge((node, node2))

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

class ProductType():
    def __init__(self, name: str, operation_processing_times: tuple) -> None:
        self.name = name
        self.operation_processing_times = operation_processing_times

class Job:
    def __init__(self, type: ProductType, due_date: int, first_unscheduled_operation: OperationNode) -> None:
        self.type = type
        self.due_date = due_date
        self.first_unscheduled_operation = first_unscheduled_operation

class OperationType(IntEnum):
    PREPERATION = 0
    FILTERING = 1
    RECEPTION = 2

class OperationNode:
    def __init__(self, job: Job, operation_type: OperationType, outgoing_directed_edges: set[Edge] = [],
                 undirected_edges: dict[OperationNode, UndirectedEdge] = [], is_dummy_start: bool = False,
                 is_dummy_finish: bool = False) -> None:
        self.job = job
        self.operationType = operation_type
        self.outgoing_directed_edges = outgoing_directed_edges
        self.undirected_edges = undirected_edges
        self.is_dummy_start = is_dummy_start
        self.is_dummy_finish = is_dummy_finish

    def __repr__(self):
        return "Node(%s, %s)" % (self.job, self.operationType)

    def __eq__(self, other):
        if isinstance(other, OperationNode):
            return (self.job == other.job and self.operationType == other.operationType)
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())

class Edge:
    def __init__(self, nodes: tuple[OperationNode, OperationNode]) -> None:
        self.nodes = nodes

class DirectedEdge(Edge):
    def __init__(self, nodes: tuple[OperationNode, OperationNode], length: int = -1):
        self.length = length
        super.__init__(nodes)

class UndirectedEdge(Edge):
    def __init__(self, nodes: tuple[OperationNode, OperationNode], pheromone_amounts: (float, float) = (t_0, t_0)):
        self.pheromone_amounts = pheromone_amounts
        super.__init__(nodes)


class WeightedDisjunctiveGraph:
    def __init__(self, edges: set[Edge], nodes: set[OperationNode], dummy_start: OperationNode, dummy_end: OperationNode) \
            -> None:
        self.edges = edges
        self.nodes = nodes
        self.dummy_start = dummy_start
        self.dummy_end = dummy_end

class Ant:
    def __init__(self, position: OperationNode) -> None:
        self.position = position

class MachinePool:
    def __init__(self):
        self.number: int
        self.loading_sequence: list[OperationNode] = []

def visibility_function(node1: OperationNode, node2: OperationNode):
    #TODO
    pass

# Initialization ----------------------------------------------------------------------------------
def ant_colony_optimization(wdg: WeightedDisjunctiveGraph, jobs: list[Job],
                            machine_pools: list[MachinePool], time_limit: int = None):
    start = time.time()

    epochs_without_improvement = 0
    best_schedule = None
    best_makespan = sys.maxint

    # Main loop -----------------------------------------------------------------------------------

    # While “optimality condition is not satisfied” do
    # -> First choice: certain number of epochs without improvement of the best solution
    # (stability condition)
    # TODO (potentially): implement better optimality condition
    # Also add option for time_limit: if the algorithm is still running when this is reached,
    # it is stopped and the best solution found up until now is returned
    while (epochs_without_improvement < MAX_EPOCHS_WITHOUT_IMPROVEMENT
           and (time_limit == None or time.time() - start < time_limit)):

        ants: list[Ant] = []

        # For each ant a = 1 to ps, place the ant on a randomly chosen operation O_ij0
        for i in range(POPULATION_SIZE):
            ants.append(Ant(random.choice(wdg.dummy_start.outgoing_directed_edges)))

        # Epoch Loop ---

        # For each ant a, a = 1 to ps do
        for ant in ants:

            # Path Generation using List Scheduler (LS) Algorithm ---

            # S_a <- ∅
            # O <- {O_ijr ∣ i=1,..,n,j=1,..,m,r=1,..,l_i}
            ant_nodes: set[OperationNode] = copy.deepcopy(wdg.nodes)

            # For each w = 1 to [Σ_(i=1,..,n) l_i] do
            for w in range(len(ant_nodes)):

                # 1. Initialization of Candidate Nodes: build the allowed list AL_w for the current
                # step w: AL_w <- {O_ijr ∈ O ∣ O_ijr-1 ∩ O = ∅}
                # (first unscheduled operation of job)
                allowed_list: list[OperationNode] = [j.first_unscheduled_operation for j in jobs]

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
                feasible_moves: list[UndirectedEdge] = []
                for o in candidate_list:
                    o_prime = machine_pools[o.operationType].loading_sequence[-1]
                    feasible_moves.append(o_prime.undirected_edges[o])

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
                # TODO (potentially): implement (more complicated) rule(s)
                # Direct the related disjunctive arc (O_i'jr' = ‘dummy start node 0’ if = 1)
                q = random()
                selected_move: Edge = None
                direction_node1_to_node2: bool = True
                if q <= q_0:
                    max: float = sys.float_info.min
                    for index, move in feasible_moves:
                        if move.nodes[0] == candidate_list[0]:
                            value = move.pheromone_amounts[0]**alpha * visibility_function(move.nodes[0], move.nodes[1])**beta
                        else:
                            value = move.pheromone_amounts[1]**alpha * visibility_function(move.nodes[1], move.nodes[0])**beta

                        if (value > max):
                            max = value
                            selected_move = move
                else:
                    weights = []
                    for index, move in feasible_moves:
                        if move.nodes[0] == candidate_list[0]:
                            value = move.pheromone_amounts[0]**alpha * visibility_function(move.nodes[0], move.nodes[1])**beta
                        else:
                            value = move.pheromone_amounts[1]**alpha * visibility_function(move.nodes[1], move.nodes[0])**beta
                        weights.append(value)
                    selected_move = random.choice(feasible_moves, weights)

                selected_move
                #Gebleven bij gekut met direction van move/edge :((((

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
                # of the job routing (arc of A) which ends at O_ijr; also, the completion time
                # t(O_ijr) = st(O_ijr) + t_ij = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)} + t_ij
                # is placed as a mark of the node O_ijr;

                #8. Updating Structures: update O by removing operation O_ijr

            # End for

            # Directing the remaining disjunctive arcs: connect these to the dummy end operation *

            # At this point we have a schedule S_a (i.e. Acyclic Conjuctive Graph (ACG) with the
            # completion times of the operations), with makespan(S_a) = length of critical path
            # = length of longest path from dummy start 0 to dummy finishing operation *

            # TODO (potentially): Apply local search routine to S_a

            # Best this epoch evaluation: if (makespan(S_a) < makespan(S_best_this_epoch))
            # Then
                # makespan(S_best_this_epoch) = makespan(S_a)
                # S_best_this_epoch <- S_a
            # End if

        # End for

        # Global updating: For the schedule of the best ant of this epoch S_b, update pheromone for
        # all the moves of S_b
        # From Rossi papers: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * (1/makespan(S_b))
        # TODO (potentially): Implement better global updating rule

        # Total best evaluation: if (makespan(S_best_this_epoch) < makespan(S_total_best))
        # Then
            # makespan(S_total_best) = makespan(S_best_this_epoch)
            # S_total_best <- S_best_this_epoch
            # epoch <- 0
        # Else
            # epoch++
        # End if

    # End while

    # Final output:
        # The best schedule found S_total_best
            # Can be written to a CSV like milp_solution
        # Lowest makespan found = makespan(S_total_best)
            # Can be used as input data for the plot to compare with MILP (see main_experiments)