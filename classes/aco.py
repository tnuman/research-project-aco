import sys
import time
from enum import IntEnum
from random import random

t_0 = 0.1
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 10

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
#         disjunctive arcs between 0 and O_*j*, between O_*j* and * and between 0 and * for all E_jh
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
    def __init__(self, type: ProductType, due_date: int) -> None:
        self.type = type
        self.due_date = due_date

class OperationType(IntEnum):
    PREPERATION = 1
    FILTERING = 2
    RECEPTION = 3

class Edge:
    def __init__(self, node_1: Node, node_2: Node, is_directed: bool, pheromone_amount: float) -> None:
        self.node_1 = node_1
        self.node_2 = node_2
        self.is_directed = is_directed
        self.pheromone_amount = pheromone_amount

class Node:
    def __init__(self, job: Job, operationType: OperationType, edges: set[Edge]) -> None:
        self.job = job
        self.operationType = operationType
        self.edges = edges

class WeightedDisjunctiveGraph:
    def __init__(self, edges: set[Edge], nodes: set[Node], dummy_start: Node, dummy_end: Node) -> None:
        self.edges = edges
        self.nodes = nodes
        self.dummy_start = dummy_start
        self.dummy_end = dummy_end

    def get_random_first_node(self) -> Node:
        edge: Edge = random.choice(filter(lambda e: e.is_directed and e.node_1 == self.dummy_start, self.dummy_start.edges))
        return edge.node_2

class Ant:
    def __init__(self, position: Node) -> None:
        self.position = position

# Initialization ----------------------------------------------------------------------------------
def ant_colony_optimization(wdg: WeightedDisjunctiveGraph, total_number_of_operations: int, time_limit: int = None):
    start = time.time()

    ants: list[Ant]

    # For each edge of the WDG, deposit a small constant amount of pheromone τ_0
    for edge in wdg.edges:
        edge.pheromone_amount = t_0

    # For each ant a = 1 to ps, place the ant on a randomly chosen operation O_ij0
    for ant in ants:
        ant.position = wdg.get_random_first_node()

    epochs_without_improvement = 0
    best_schedule = None
    best_makespan = sys.maxint


    # Main loop ---------------------------------------------------------------------------------------

    # While “optimality condition is not satisfied” do
    # -> First choice: certain number of epochs without improvement of the best solution
    # (stability condition)
    # TODO (potentially): implement better optimality condition
    # Also add option for time_limit: if the algorithm is still running when this is reached,
    # it is stopped and the best solution found up until now is returned
    while (epochs_without_improvement < MAX_EPOCHS_WITHOUT_IMPROVEMENT
           and (time_limit == None or time.time() - start < time_limit)):

        # Epoch Loop ---

        # For each ant a, a = 1 to ps do
        for ant in ants:

            # Path Generation using List Scheduler (LS) Algorithm ---

            # S_a <- ∅
            # O <- {O_ijr ∣ i=1,..,n,j=1,..,m,r=1,..,l_i}
            ant_nodes = wdg.nodes

            # For each w = 1 to [Σ_(i=1,..,n) l_i] do
            for w in range(total_number_of_operations):

                # 1. Initialization of Candidate Nodes: build the allowed list AL_w for the current
                # step w: AL_w <- {O_ijr ∈ O ∣ O_ijr-1 ∩ O = ∅} (first unscheduled operation of job)
                allowed_list: set[Node] = []
                for node in ant_nodes:
                    if ()

                # 2. Restriction: restriction of the allowed list by means of optimality criteria
                # (i.e. active or non-delay schedule); let the candidate list CL_w be the restricted
                # allowed list
                # First choice: CL_w = AL_w
                # TODO (potentially): implement (more complicated) restriction

                # 3. Initialization of Feasible Moves: mark as a feasible move each disjunctive arc
                # (O_i'jr', O_ijr) of E_j where O_ijr ∈ CL_ w and O_i'jr' is the last operation of
                # the loading sequence of resource j (it creates the possibility for the candidate
                # operation to become the new last operation of that loading sequence)

                # 4. Move Selection: select a feasible move (Oi'jr', Oijr) of E_j using transition
                # probability rule depending on pheromones τ and heuristic visibility function η
                # First choice for ...
                # Transition probability rule: see Rossi's paper
                # Local updating rule: τ(O_ijr, O_i'jr') = (1-ρ) * τ(O_ijr, O_i'jr') + ρ * τ_0
                # Heuristic visibility function: Earliest Starting Time (EST) dispatching rule (Blum and Sampels, 2004)
                # TODO (potentially): implement (more complicated) rule(s)
                # Direct the related disjunctive arc (O_i'jr' = ‘dummy start node 0’ if = 1)

                # 5. Arcs Removal: remove all the remaining disjunctive arcs connected to O_i'jr'
                # (i.e. no other operation can be immediately subsequent to O_i'jr' in the loading sequence)
                # Also remove all the remaining disjunctive arcs of E_h connected to O_ijr, i.e. h ∈ Mij and h != j
                # (i.e. no other loading sequence can include the operation)

                # 6. Computing length: the length of the arc (O_i'jr', O_ijr) is evaluated as the sum
                # of processing and lag times of node O_ijr by means of expression
                # st(O_ijr) = max{t(O_i'jr') + t(f_i,j), t(O_ijr-1)}

                # 7. Transferring length: this length is placed on the related arc and on the arc of
                # the job routing (arc of A) which ends at O_ijr; also, the completion time
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

        # Global updating: For the schedule of the best ant of this epoch S_b, update pheromone for all
        # the moves of S_b
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