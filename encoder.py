import math
import logging

from automaton import Automaton

class Encoder(object):
    """Class that encodes automata with given coding"""
    def __init__(self, entropy):
        self.entropy = entropy

    def automaton_bits(self, automaton):
        automaton.round_and_normalize()
        automaton_state_bits = math.log(len(automaton.m), 2)
        automaton_bits = 0.0
        edge_bits = automaton.code.bits
        for state in automaton.m:
            logging.debug("State {0}".format(state))

            # bits for emission
            s_len = (len(automaton.emissions[state])
                     if not (state.startswith("EPSILON")or state in ["$", "^"])
                     else 1)
            s_bits = (0.0 if state == "$" else self.entropy * s_len)
            automaton_bits += s_bits
            logging.debug("Emit bits={0}".format(s_bits))

            # bits for transition
            source_bits = (automaton_state_bits if state != "^" else 0.0)
            if len(automaton.m[state].items()) == 1:
                target_bits = (automaton_state_bits
                               if automaton.m[state].items()[0]!= "$"
                               else 0.0)
                automaton_bits += (source_bits + target_bits)
                logging.debug("Only one transition from here, bits={0}".format(
                    source_bits + target_bits))
                continue

            prev = automaton_bits
            for target, prob in automaton.m[state].iteritems():
                # we only need target state and string and probs
                target_bits = (automaton_state_bits if target != "$" else 0.0)

                automaton_bits += (source_bits + edge_bits + target_bits)
            logging.debug("Transition bits={0}".format(
                automaton_bits - prev))
        return automaton_bits

    def encode(self, automaton, corpus, reverse=False):
        automaton_bits = self.automaton_bits(automaton)
        err_bits = automaton.distance_from_corpus(corpus,
                       Automaton.kullback, reverse)
        return automaton_bits, err_bits, automaton_bits + err_bits

