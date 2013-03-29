import math
from automaton import Automaton

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
