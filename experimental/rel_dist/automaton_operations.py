import sys
from math import log, exp
import fst

from floyd_warshall import get_states
from pywfsalearner.automaton import Automaton


def set_weight(automaton, function):

    orig_weights = automaton.m
    new_weights = orig_weights
    for src in orig_weights:
        for tg_data in src: 
            tg, w = tg_data.iteritems()[0] 
            new_weights[src][tg] = function(w)
    automaton.m = new_weights         

def phi_0(automaton):
    set_weight(automaton, lambda x:(x, 0.))

def phi_1(automaton):
    set_weight(automaton, lambda x:(1., x))

def logarithm(automaton):
    set_weight(automaton, log)

def exponential(automaton):
    set_weight(automaton, exp)

def moore_to_pyopenfst_mealy(moore):
    print 'moore_to_pyopenfst_mealy' 
    moore_states = get_states(moore.m)[2]
    # print moore_states
    moore_indeces = dict([(value, index) for index, value in enumerate(moore_states)])
    
    mealy = fst.SimpleFst()
    for i in range(len(moore_states)):
        mealy.add_state()
    mealy.start = moore_indeces['^']
    mealy[moore_indeces['$']].final = 1.
    for state in moore_states:
        mealy.add_state()
    for src in moore.m:
        for tg in moore.m[src]:
            w = moore.m[src][tg]
            if tg in moore.emissions:
                emission = "".join(moore.emissions[tg])
            else:
                emission = "EPSILON"
            mealy.add_arc(moore_indeces[src], moore_indeces[tg], emission, emission, w)
    return mealy


def automata_to_fw_input_automata(moore_aut_1):
    
    mealy_aut_1 = moore_to_pyopenfst_mealy(moore_aut_1)
    mealy_aut_2 = moore_to_pyopenfst_mealy(moore_aut_2)

    return input_aut_1, input_aut_2
    
def test_fst(moore_aut_1):
    mealy_aut_1 = moore_to_pyopenfst_mealy(moore_aut_1)
    mealy_aut_1.write('test')
    return ''

def main():
    aut = Automaton.create_from_dump(sys.argv[1])
   # print aut.m
   # print phi_0(aut).m
    test_fst(aut)

if __name__ == "__main__":
    main()
