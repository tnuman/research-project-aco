
# Input: a weighted digraph WDG = (N_F, A, E_jh, W_N, W_E)
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