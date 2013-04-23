import math
import logging

from automaton import Automaton

class Encoder(object):
    """Class that encodes automata with given coding"""
    def __init__(self, entropy):
        self.entropy = entropy

    def automaton_bits(self, automaton):
        automaton.round_and_normalize()
        automaton_onestate_bits = math.log(len(automaton.m), 2)
        automaton_emission_bits = 0.0
        automaton_trans_bits = 0.0
        q = automaton.quantizer
        edge_bits = q.bits
        for state in automaton.m:
            logging.debug("State {0}".format(state))

            # bits for emission
            s_len = (len(automaton.emissions[state])
                     if not (state.startswith("EPSILON")or state in ["$", "^"])
                     else 1)
            s_bits = (0.0 if state == "$" else self.entropy * s_len)
            automaton_emission_bits += s_bits
            logging.debug("Emit bits={0}".format(s_bits))

            # bits for transition
            source_bits = (automaton_onestate_bits if state != "^" else 0.0)
            if len(automaton.m[state].items()) == 1:
                target_bits = (automaton_onestate_bits
                               if automaton.m[state].items()[0][0]!= "$"
                               else 0.0)
                automaton_trans_bits += (source_bits + target_bits)
                logging.debug("Only one transition from here, bits={0}".format(
                    source_bits + target_bits))
                continue

            for target, prob in automaton.m[state].iteritems():
                # we only need target state and string and probs
                target_bits = (automaton_onestate_bits if target != "$" else 0.0)

                if q.representer(prob) != q.representer(q.neg_cutoff):
                    automaton_trans_bits += (source_bits + edge_bits + target_bits)
                    logging.debug("transition is encoded in {0} bits({1}-{2}-{3})".format(
                        source_bits + edge_bits + target_bits, source_bits, edge_bits, target_bits))
                else:
                    # we don't wanna encode those transitions at all
                    pass

        return automaton_emission_bits, automaton_trans_bits

    def encode(self, automaton, corpus, reverse=False):
        emit_bits, trans_bits = self.automaton_bits(automaton)
        automaton_bits = emit_bits + trans_bits
        err_bits = automaton.distance_from_corpus(corpus,
                       Automaton.kullback, reverse)
        return automaton_bits, emit_bits, trans_bits, err_bits

