from collections import *
from copy import *


g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# Results for compute_rdmst(g0, 0):
# ({0: {1: 2, 2: 2, 3: 2}, 1: {5: 2}, 2: {4: 2}, 3: {}, 4: {}, 5: {}}, 10)

g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# Results for compute_rdmst(g1, 0):
# ({0: {2: 4}, 1: {}, 2: {3: 8}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, 28)

g2 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
# Results for compute_rdmst(g2, 0):
# ({0: {2: 4}, 1: {}, 2: {1: 2}}, 6)

g3 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
# Results for compute_rdmst(g3, 1):
# ({1: {3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {}, 5: {}}, 11.1)

g4 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1, 8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0}, 6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}
# Results for compute_rdmst(g4, 1):
# ({1: {8: 6.1, 3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {9: 6.1, 10: 5.0}, 5: {}, 6: {7: 0.0}, 7: {}, 8: {}, 9: {}, 10: {6: 0.0}}, 28.3)

custom = {0:{1:20,2:4,3:20},1:{2:2},2:{3:8,4:20},3:{4:4},4:{1:4}}

def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist

def reverse_digraph_representation(graph):
    """
    Input: A directed graph in standard representation (dictionary).
    Output: A directed graph in reversed representation (dictionary).
    This function takes a directed graph in standard representation stored in a dictionary
    and returns that same graph in reversed representation stored in a dictionary.
    """
    rgraph = {}
    #Forms the basic structure of the dictionary being created.
    for each in graph.keys():
        rgraph[each]={}

    #Inputs the data for each entry in the new dictionary based on the old dictionary.
    for each in graph.keys():
        for every in graph[each].keys():
            rgraph[every][each]=graph[each][every]
    return rgraph
def modify_edge_weights(rgraph, root):
    """
    Input: A directed graph in reversed representation (dictionary) and a node representing the root of the graph.
    Output: Nothing.
    This function modifies a directed graph, rgraph, by taking each node and reducing the weight
    of each edge for which that node is a head by the value of the minimum weighted edge for which
    that node is a head.
    """

    for each in rgraph:
        minim = float("inf")
        # This loop calculates the minimum edge for which the node each is a head
        for every in rgraph[each].keys():
            if (rgraph[each][every]<minim):
                minim = rgraph[each][every]
        # This loop reduces the weight of each edge for which the node each is a head by the value found
        # in the previous loop.
        for new_every in rgraph[each].keys():
            if each != root:
                rgraph[each][new_every]=rgraph[each][new_every]-minim


# rg0 = reverse_digraph_representation(g0)
# rg1 = reverse_digraph_representation(g1)
# rg2 = reverse_digraph_representation(g2)
# rcustom = reverse_digraph_representation(custom)
#
# modify_edge_weights(rg0, 0)
# modify_edge_weights(rg1, 0)
# modify_edge_weights(rg2, 0)
# modify_edge_weights(rcustom, 0)

#print rg0
#print rg1
#print rg2
#print rcustom

def compute_rdst_candidate(rgraph, root):
    candied_date = {}

    #This loop forms the structure of the tree we are creating.
    for node in rgraph:
        candied_date[node]= {}

    for node in rgraph:
        if node != root:
            minim =float("inf")
            # Attaches nodes according to the lemma
            for every in rgraph[node].keys():
                if (rgraph[node][every] < minim):
                    minim = rgraph[node][every]
                    the_every = every
            candied_date[node][the_every]=minim
    return candied_date

# print compute_rdst_candidate(rg0, 0)
# print compute_rdst_candidate(rg1, 0)
# print compute_rdst_candidate(rg2, 0)
# print compute_rdst_candidate(rcustom, 0)

def help_bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}
    path = {}
    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0
    path[startnode] = [startnode]
    x = list(path[startnode])
    # Initialize search queue
    queue = deque([startnode])
    cycle = False
    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            # This if statement finds the node which is the final node before the completion of the cycle.
            if nbr==startnode:
                cycle = True
                end_cycle = node
            if dist[nbr] == float('inf'):
                path[nbr]= list(path[node])
                path[nbr].append(nbr)
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    if cycle:
        # This returns the calculated path from start node to end_cycle, all nodes in the cycle.
        return path[end_cycle]
    return False

def compute_cycle(rdst_candidate):
    """
    Input: a candidate for being a rdst in the form of a dictionary of dictionaries.
    Output: Either a tuple containing all nodes in a cycle in the candidate, or nothing.
    This function checks the candidate for a cycle and, if found, returns it.
    This function is for a directed graph with an in-degree of either 1 or zero for each node.
    """
    for each in rdst_candidate.keys():
        #bfs has been modified to return a cycle if one exists
        cycle = help_bfs(rdst_candidate,each)
        if cycle != False:
            break
    # If the loop iterates to completion without finding anything, there is no cycle.
    if cycle == False:
        return False
    cycle = tuple(cycle)
    return cycle

#cycg0= compute_cycle(compute_rdst_candidate(rg0,0))
# cycg1= compute_cycle(compute_rdst_candidate(rg1,0))
# cycg2= compute_cycle(compute_rdst_candidate(rg2,0))
# cyccustom = compute_cycle(compute_rdst_candidate(rcustom,0))

def contract_cycle(graph, cycle):
    """
    Input: A graph in standard representation in the form of a dictionary, and a tuple containing
    each node in a cycle in the graph.
    Output. The original graph with the cycle replaced by a node, and the new node.
    This function takes a graph with a cycle in it and then contracts that cycle into a single node so that
    any node that had an edge with nodes on the cycle (in either direction) now has one edge (for that same direction)
    with the contraction node. This edge
    is the minimum of edges the node had with the cycle in that direction.
    """
    # value of the new node
    cstar = max(graph.keys())+ 1

    graph[cstar] = {}

    for each in graph.keys():
        for every in graph[each].keys():

            # This iteration loop assigns the edges going out of the contraction
            if each in cycle:
                if every not in cycle:
                    container =[]
                    # This part is to make sure only the minimum possible edge is attached to the contraction
                    for all in cycle:
                        if (all in graph.keys()) and (every in graph[all].keys()):
                            container.append(graph[all][every])
                    if cstar in graph.keys() and every in graph[cstar].keys() and min(container) < graph[cstar][every]:
                        graph[cstar][every]= min(container)
                    elif every not in graph[cstar].keys():
                        graph[cstar][every] = min(container)


            # This iteration loop assigns the edges coming into the contraction
            elif each not in cycle:
                if every in cycle:
                    container=[]
                    # This part is to make sure only the minimum possible edge is attached to the contraction
                    for all in cycle:
                        if all in graph[each].keys():
                            container.append(graph[each][all])
                    graph[each][cstar]=min(container)
        # This part deletes the outgoing edges of the cycle nodes
        if each in cycle:
            del graph[each]

    # This loop eliminates the last traces of the cycle nodes from the graph
    for each in graph.keys():
        for every in graph[each].keys():
            if every in cycle:
                del graph[each][every]

    return (graph, cstar)
# print cycg1
# print cycg2
# print cyccustom

# print contract_cycle(g1, cycg1)
# print contract_cycle(g2, cycg2)
# print contract_cycle(custom, cyccustom)

def expand_graph(graph, rdst_candidate, cycle, cstar):
    """
    Input:
    (1) The weighted digraph original_graph (in standard representation) whose cycle was contracted;
    (2) the RDST candidate rdst_candidate as a weighted digraph, in standard representation, that was computed on the
    contracted version of original_graph;
    (3) the tuple of nodes on the cycle cycle that was contracted; and
    (4) the number that labels the node that replaces the contracted cycle, cstar.
    Output:
    The function returns a weighted digraph (in standard representation)
    that results from expanding the cycle in rdst_candidate.
    """

    #NEED TO FIX DIFFERENCE BETWEEN GRAPH AND RDST CANDIDATE

    check_graph = deepcopy(graph)
    expanded={}
    # This loop structure creates the base structure for the graph.
    for each in graph.keys():
        expanded[each]={}

    # This loop puts all directed edges from the candidate in the graph that are neither to nor from the cycle.
    for each in rdst_candidate.keys():
        for every in rdst_candidate[each].keys():
            if each != cstar and every != cstar:
                expanded[each][every]=rdst_candidate[each][every]

    # Outside cycle to inside cycle. ASSUMES ONLY ONE.
    for each in rdst_candidate.keys():
        for every in cycle:
            if cstar in rdst_candidate[each].keys() and every in check_graph[each].keys():
                # This if statement compares the edge value in the candidate to the original graph to determine what cycle
                # it was attached to.
                if rdst_candidate[each][cstar] == check_graph[each][every]:
                    expanded[each][every] = check_graph[each][every]
                    start_cycle = every

    # Inside cycle to outside cycle.
    # This dictionary prevents paths of the same value being attributed as the source of a single edge in the contracted graph.
    visited = defaultdict(lambda:False)
    for each in cycle:
        for every in rdst_candidate[cstar].keys():
            if each in check_graph.keys() and every in check_graph[each].keys():
                # This if statement compares the edge value in the candidate to the original graph to determine what cycle
                # it was attached to.
                if check_graph[each][every]==rdst_candidate[cstar][every] and not visited[every]:
                    expanded[each][every] = check_graph[each][every]
                    visited[every]=True

    # These loops inputs the cycle path:
    for each in range(len(cycle)):
        if each != len(cycle)-1 and cycle[-(each +2)]!= start_cycle:
            expanded[cycle[-(each +1)]][cycle[-(each +2)]] = check_graph[cycle[-(each +1)]][cycle[-(each +2)]]
        if each == len(cycle)-1 and cycle[-1] != start_cycle:
            expanded[cycle[0]][cycle[-1]] = check_graph[cycle[0]][cycle[-1]]

    # This structure prevents replicate paths caused by multiple cycles.
    for each in expanded.keys():
        for every in expanded.keys():
            for all in expanded[each].keys():
                for last in expanded[every].keys():
                    if all == last and each != every:
                        del expanded[every][all]

    return expanded



# copyg1= deepcopy(g1)
# copyg2= deepcopy(g2)
# copycustom= deepcopy(custom)
#
# newg1 = contract_cycle(copyg1,cycg1)
# con_candidateg1 = reverse_digraph_representation(compute_rdst_candidate(reverse_digraph_representation(newg1[0]), 0))
# newg2 = contract_cycle(copyg2,cycg2)
# con_candidateg2 = reverse_digraph_representation(compute_rdst_candidate(reverse_digraph_representation(newg2[0]), 0))
# newcustom = contract_cycle(copycustom,cyccustom)
# con_candidatecustom = reverse_digraph_representation(compute_rdst_candidate(reverse_digraph_representation(newcustom[0]), 0))


# print expand_graph(g1,con_candidateg1,cycg1,newg1[1])
# print expand_graph(g2,con_candidateg2,cycg2,newg2[1])
# print expand_graph(custom,con_candidatecustom,cyccustom,newcustom[1])


def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
    """
    cop_graph = deepcopy(graph)
    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    for each in rdst_candidate.keys():
        for every in rdst_candidate[each].keys():
            rdst_candidate[each][every]=cop_graph[every][each]
    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(graph)


        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        # cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)



        # Step 4(c) of the algorithm
        rdmst = expand_graph(cop_graph, new_rdst_candidate, cycle, cstar)



        return rdmst

def compute_rdmst(graph, root):
    """
    This function checks if:
    (1) root is a node in digraph graph, and
    (2) every node, other than root, is reachable from root
    If both conditions are satisfied, it calls compute_rdmst_helper
    on (graph, root).

    Since compute_rdmst_helper modifies the edge weights as it computes,
    this function reassigns the original weights to the RDMST.
    Arguments:
    graph -- a weighted digraph in standard dictionary representation.
    root -- a node id.
    Returns:
    An RDMST of graph rooted at r and its weight, if one exists;
    otherwise, nothing.
    """

    if root not in graph:
        print "The root node does not exist"
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print "The root does not reach every other node in the graph"
            return

    rdmst = compute_rdmst_helper(graph, root)


    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)




#print compute_rdmst_helper(g0, 0)
#print compute_rdmst_helper(g1, 0)
#print compute_rdmst_helper(g2, 0)
#print compute_rdmst_helper(g3, 1)
#print compute_rdmst_helper(g4, 1)
#print compute_rdmst_helper(custom, 0)
# print compute_rdmst(g0, 0)
# print compute_rdmst(g1, 0)
# print compute_rdmst(g2, 0)
# print compute_rdmst(g3, 1)
# print compute_rdmst(g4, 1)
# print compute_rdmst(custom, 0)



#PROBLEM 3




def infer_transmap(gen_data, epi_data, patient_id):
    """
        Infers a transmission map based on genetic
        and epidemiological data rooted at patient_id

        Arguments:
        gen_data -- filename with genetic data for each patient
        epi_data -- filename with epidemiological data for each patient
        patient_id -- the id of the 'patient 0'

        Returns:
        The most likely transmission map for the given scenario as the RDMST
        of a weighted, directed, complete digraph
        """

    complete_digraph = construct_complete_weighted_digraph(gen_data, epi_data)
    return compute_rdmst(complete_digraph, patient_id)


def read_patient_sequences(filename):
    """
        Turns the bacterial DNA sequences (obtained from patients) into a list containing tuples of
        (patient ID, sequence).

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A list of (patient ID, sequence) tuples.
        """
    sequences = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            if len(line) > 5:
                patient_num, sequence = line.split("\t")
                sequences.append((int(patient_num), ''.join(e for e in sequence if e.isalnum())))
    return sequences


def read_patient_traces(filename):
    """
        Reads the epidemiological data file and computes the pairwise epidemiological distances between patients

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A dictionary of dictionaries where dict[i][j] is the
        epidemiological distance between i and j.
    """
    trace_data = []
    patient_ids = []
    first_line = True
    with open(filename) as f:
        for line in f:
            if first_line:
                patient_ids = line.split()
                patient_ids = map(int, patient_ids)
                first_line = False
            elif len(line) > 5:
                trace_data.append(line.rstrip('\n'))
    return compute_pairwise_epi_distances(trace_data, patient_ids)


def compute_pairwise_gen_distances(sequences, distance_function):
    """
        Computes the pairwise genetic distances between patients (patients' isolate genomes)

        Arguments:
        sequences -- a list of sequences that correspond with patient id's
        distance_function -- the distance function to apply to compute the weight of the
        edges in the returned graph

        Returns:
        A dictionary of dictionaries where gdist[i][j] is the
        genetic distance between i and j.
        """
    gdist = {}
    cultures = {}

    # Count the number of differences of each sequence
    for i in range(len(sequences)):
        patient_id = sequences[i][0]
        seq = sequences[i][1]
        if patient_id in cultures:
            cultures[patient_id].append(seq)
        else:
            cultures[patient_id] = [seq]
            gdist[patient_id] = {}
    # Add the minimum sequence score to the graph
    for pat1 in range(1, max(cultures.keys()) + 1):
        for pat2 in range(pat1 + 1, max(cultures.keys()) + 1):
            min_score = float("inf")
            for seq1 in cultures[pat1]:
                for seq2 in cultures[pat2]:
                    score = distance_function(seq1, seq2)
                    if score < min_score:
                        min_score = score
            gdist[pat1][pat2] = min_score
            gdist[pat2][pat1] = min_score
    return gdist


### HELPER FUNCTIONS. ###

def find_first_positives(trace_data):
    """
        Finds the first positive test date of each patient
        in the trace data.
        Arguments:
        trace_data -- a list of data pertaining to location
        and first positive test date
        Returns:
        A dictionary with patient id's as keys and first positive
        test date as values. The date numbering starts from 0 and
        the patient numbering starts from 1.
        """
    first_pos = {}
    for pat in range(len(trace_data[0])):
        first_pos[pat + 1] = None
        for date in range(len(trace_data)):
            if trace_data[date][pat].endswith(".5"):
                first_pos[pat + 1] = date
                break
    return first_pos


def compute_epi_distance(pid1, pid2, trace_data, first_pos1, first_pos2, patient_ids):
    """
        Computes the epidemiological distance between two patients.

        Arguments:
        pid1 -- the assumed donor's index in trace data
        pid2 -- the assumed recipient's index in trace data
        trace_data -- data for days of overlap and first positive cultures
        first_pos1 -- the first positive test day for pid1
        first_pos2 -- the first positive test day for pid2
        patient_ids -- an ordered list of the patient IDs given in the text file

        Returns:
        Finds the epidemiological distance from patient 1 to
        patient 2.
        """
    first_overlap = -1
    assumed_trans_date = -1
    pid1 = patient_ids.index(pid1)
    pid2 = patient_ids.index(pid2)
    # Find the first overlap of the two patients
    for day in range(len(trace_data)):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            first_overlap = day
            break
    if (first_pos2 < first_overlap) | (first_overlap < 0):
        return len(trace_data) * 2 + 1
    # Find the assumed transmission date from patient 1 to patient 2
    for day in range(first_pos2, -1, -1):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            assumed_trans_date = day
            break
    sc_recip = first_pos2 - assumed_trans_date

    if first_pos1 < assumed_trans_date:
        sc_donor = 0
    else:
        sc_donor = first_pos1 - assumed_trans_date
    return sc_donor + sc_recip


def compute_pairwise_epi_distances(trace_data, patient_ids):
    """
        Turns the patient trace data into a dictionary of pairwise
        epidemiological distances.

        Arguments:
        trace_data -- a list of strings with patient trace data
        patient_ids -- ordered list of patient IDs to expect

        Returns:
        A dictionary of dictionaries where edist[i][j] is the
        epidemiological distance between i and j.
        """
    edist = {}
    proc_data = []
    # Reformat the trace data
    for i in range(len(trace_data)):
        temp = trace_data[i].split()[::-1]
        proc_data.append(temp)
    # Find first positive test days and remove the indication from the data
    first_pos = find_first_positives(proc_data)
    for pid in first_pos:
        day = first_pos[pid]
        proc_data[day][pid - 1] = proc_data[day][pid - 1].replace(".5", "")
    # Find the epidemiological distance between the two patients and add it
    # to the graph
    for pid1 in patient_ids:
        edist[pid1] = {}
        for pid2 in patient_ids:
            if pid1 != pid2:
                epi_dist = compute_epi_distance(pid1, pid2, proc_data,
                                                first_pos[pid1], first_pos[pid2], patient_ids)
                edist[pid1][pid2] = epi_dist
    return edist



def compute_hamming_distance(patient1,patient2):
    """
    Inputs: Two patient sequences in binary form.
    Outputs: The hamming distance of the two sequences in int form.
    This function compares two sequences and provides the count of indices
    for which the two sequences hold different values. This is the hamming number.
    """
    hamming=0
    # iterates through each number in the sequence.
    # assumes same length for each sequence
    for each in range(len(patient1)):
        if patient1[each] != patient2[each]:
            hamming = hamming + 1
    return hamming

def construct_complete_weighted_digraph(geneticname,epidemiologicalname):
    """
    Input: The names of two files in string form.
    Output: A complete weighted digraph in dictionary form based on
    Equation 1 in the Homework 3 Description Document.
    This function takes the data from the files and runs them through the Equaltion 1 formula to create a
    graph that parallels the epidemiological graph in structure, but has weight values according to
    Equation 1.
    """
    # reading in the graphs
    sequences = read_patient_sequences(geneticname)
    traces= read_patient_traces(epidemiologicalname)

    # Preparing a min
    max_e = -float("inf")
    #calling the gab values
    falcon = compute_pairwise_gen_distances(sequences,compute_hamming_distance)

    # Calculating the true max_e
    for each in traces.keys():
        for every in traces[each].keys():
            if traces[each][every] > max_e:
                max_e = traces[each][every]


    newgraph = {}
    for each in traces.keys():
        newgraph[each]={}

    # This for loop inputs the data into the given equation to create the graph.
    for each in traces.keys():
        for every in traces[each].keys():
            numerator = 999 * (float(traces[each][every]) / max_e)
            denominator = 10 ** 5
            falcon_val = falcon[each][every]
            newgraph[each][every]=falcon_val + (numerator/denominator)




    return newgraph

