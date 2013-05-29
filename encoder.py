import sys
import math
import logging
from collections import defaultdict

from automaton import Automaton
from corpus import read_corpus, normalize_corpus
from quantizer import LogLinQuantizer

def compute_state_entropy(automaton, which):
    counts = defaultdict(int)
    for src in automaton.m:
        for tgt in automaton.m[src]:
            if which == "src":
                counts[src] += 1
            elif which == "tgt":
                counts[tgt] += 1

    total = float(sum(counts.itervalues()))
    return sum(-c/total * math.log(c/total) for c in counts.itervalues())

class Encoder(object):
    """Class that encodes automata with given coding"""
    def __init__(self, entropy, state_bits="u"):
        self.entropy = entropy
        self.state_bits = state_bits

    def automaton_bits(self, automaton):
        automaton.round_and_normalize()
        automaton_onestate_bits = math.log(len(automaton.m), 2)
        if self.state_bits == "u":
            source_onestate_bits = automaton_onestate_bits
            target_onestate_bits = automaton_onestate_bits
        elif self.state_bits == "e":
            source_onestate_bits = compute_state_entropy(automaton, "src")
            target_onestate_bits = compute_state_entropy(automaton, "tgt")

        automaton_emission_bits = 0.0
        automaton_trans_bits = 0.0
        q = automaton.quantizer
        edge_bits = q.bits
        for state in automaton.m:
            logging.debug("State {0}".format(state))

            # bits for emission
            s_len = (len(automaton.emissions[state])
                     if not (state.startswith("EPSILON") or state == "^")
                     else 0)
            s_bits = self.entropy * s_len
            automaton_emission_bits += s_bits
            logging.debug("Emit bits={0}".format(s_bits))

            # bits for transition
            source_bits = (source_onestate_bits if state != "^" else 0.0)
            if len(automaton.m[state].items()) == 1:
                target_bits = (target_onestate_bits
                               if automaton.m[state].items()[0][0]!= "$"
                               else 0.0)

                # if target is $, and only one transition, we won't encode
                # the source, because we assume that there are no trimmed
                # states
                source_bits = (source_bits
                               if automaton.m[state].items()[0][0]!= "$"
                               else 0.0)

                automaton_trans_bits += (source_bits + target_bits)
                logging.debug("Only one transition from here, bits={0}".format(
                    source_bits + target_bits))
                continue

            for target, prob in automaton.m[state].iteritems():
                # we only need target state and string and probs
                target_bits = (target_onestate_bits if target != "$" else 0.0)

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
        lang = automaton.language()
        gen_lang_entropy = sum([-d["$"] * math.log(d["$"])
                                for d in lang.itervalues() if d["$"] > 0.0])
        return automaton_bits, emit_bits, trans_bits, err_bits, gen_lang_entropy

def main():
    automaton = Automaton.create_from_dump(sys.argv[1])
    corpus = read_corpus(open(sys.argv[2]))
    normalize_corpus(corpus)
    entropy = float(sys.argv[3])
    string_bits = "u"
    if len(sys.argv) > 4:
        string_bits = sys.argv[4]
    q = LogLinQuantizer(10, -20)
    automaton.quantizer = q

    encoder = Encoder(entropy, string_bits)
    print encoder.encode(automaton, corpus)

if __name__ == "__main__":
    main()


