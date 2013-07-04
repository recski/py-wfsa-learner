import sys

from collections import defaultdict
from math import exp

from pywfsalearner.automaton import Automaton
from entropy_semiring import Vector

# automaton format : a1[src][tgt] = logprob

def get_states(weighted_graph):

    source_states = set(weighted_graph.keys())
    target_states = reduce(lambda x, y:x | y, ([set(weighted_graph[src].keys()) for src in weighted_graph]))
    states = list(source_states | target_states)
    return source_states, target_states, states


def shortest_path(weighted_graph):
    
    states = get_states(weighted_graph)[2]

    distances = defaultdict(int)
    for src in weighted_graph:
        tg_data = weighted_graph[src]
        for tg in tg_data:
            w = tg_data[tg]
            distances[(src, tg)] = exp(w)
    for k in states:
        for i in states:
            for j in states: 
                distances[(i, j)] = distances[(i, j)] + (distances[(i, k)] * distances[(k, j)])  
    return distances 


#def create_cross_entropy(first_automaton, second_automaton):

        


def main():

    first_automaton = Automaton.create_from_dump(sys.argv[1])
    second_automaton = Automaton.create_from_dump(sys.argv[2])
    create_cross_entropy(first_automaton, second_automaton) 

    graph = this_automaton.m
    dist = shortest_path(graph)
    
    print dist[('^', '$')]


if __name__ == "__main__":
    main()

