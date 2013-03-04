import math
import logging

def mdl(automaton, corpus, bits, distfp, n_state, n_alphabet, n_word):
    logging.warning("Not implemented yet, mealy-moore problem.")
    return 0
    distance = automaton.distance_from_corpus(corpus, distfp)
    err = distance * n_word
    bits_per_transition = math.log(n_state ** 2, 2) + math.log(n_alphabet, 2)
    result = transition_n * bits_per_transition + err
    return result

def moore_mdl(automaton, corpus, bits, distfp, n_state, n_alphabet):
    dump = 0.0

    # adding the number of states
    dump += math.log(math.log(len(automaton.m.keys())))

    # adding the emissions per state
    dump += n_state * math.log(n_alphabet)

    # adding the transition probabilities
    dump += 2.0 * math.log(n_state) * bits
    return dump
