import math

from automaton import Automaton

class Encoder(object):
    """Class that encodes automata with given coding"""
    def __init__(self, corpus, entropy):
        self.corpus = corpus
        self.entropy = entropy

    def automaton_bits(self, automaton):
        automaton.round_and_normalize()
        automaton_state_bits = math.log(len(automaton.m), 2)
        automaton_bits = 0.0
        for state in automaton.m:
            if state == "^":
                source_bits = 0.0
            for target, prob in automaton.m:
                # we only need target state and string and probs
                if target == "$":
                    s_bits = 0.0
                    target_bits = 0.0
                else:
                    s_bits = self.entropy * len(automaton.emissions[target])
                    target_bits = automaton_state_bits

                edge_bits = automaton.code.bits
                automaton_bits += (source_bits + s_bits + edge_bits
                                   + target_bits)
        return automaton_bits

    def encode(self, automaton, reverse=False):
        automaton_bits = self.automaton_bits(automaton)
        err_bits = automaton.distance_from_corpus(self.corpus,
                       Automaton.kullback, reverse)
        return automaton_bits, err_bits, automaton_bits + err_bits

