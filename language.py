""" This scripts counts the generated language for a given automaton
first arg, if given, is a probability, and word probabilities are computed
until prob_mass >= 1.0 - first_arg
"""
import sys
import math

from automaton import Automaton

def main():
    wfsa = Automaton.create_from_dump(open(sys.argv[1]))
    if len(sys.argv) > 2:
        remaining = float(sys.argv[2])
        lang = wfsa.language(remaining)
    else:
        lang = wfsa.language()
    for w in lang:
        di = wfsa.state_indices["$"]
        prob = math.exp(lang[w][di])
        print "{0} {1}".format("".join(w), prob)

if __name__ == "__main__":
    main()
