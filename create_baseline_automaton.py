"""With this script, we can create an automaton that has
a start state, a final state, and one state for every _word_ in the
corpus"""
import math

from automaton import Automaton

def create_word_wfsa(corpus):
    wfsa = Automaton()
    for word in corpus:
        prob = corpus[word]
        state = "".join(word) + "_0"
        wfsa.emissions[state] = word
        wfsa.m_emittors[word].add(state)
        wfsa.m["^"][state] = math.log(prob)
        wfsa.m[state]["$"] = 0 # log(1)
    return wfsa


