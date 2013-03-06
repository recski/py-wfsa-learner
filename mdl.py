import math
import logging
from automaton import Automaton

def mdl(automaton, corpus, bits, distfp, n_state, n_alphabet, n_word):
    logging.warning("Not implemented yet, mealy-moore problem.")
    return 0
    distance = automaton.distance_from_corpus(corpus, distfp)
    err = distance * n_word
    bits_per_transition = math.log(n_state ** 2, 2) + math.log(n_alphabet, 2)
    result = transition_n * bits_per_transition + err
    return result

def moore_mdl(automaton, corpus, bits, n_words=135):
    dump = 0.0

    # adding the number of states
    n_state = len(automaton.m.keys())
    dump += math.log(math.log(n_state))

    # adding the emissions per state
    n_alphabet = len(automaton.m_emittors)
    dump += n_state * math.log(n_alphabet)

    # adding the transition probabilities
    dump += 2.0 * math.log(n_state) * bits

    err = automaton.distance_from_corpus(corpus, Automaton.kullback) * n_words

    return err + dump, dump, err
