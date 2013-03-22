""" Simple script that excepts a coder dump and a corpus, reads them
and counts the entropy of the probabilities and quantized probs."""

import sys
import math

from code import AbstractCode
from corpus import read_corpus, normalize_corpus
from automaton import Automaton

def compute_entropy(probs, code):
    dist = 0.0
    for prob in probs:
        prob_q = math.exp(code.representer(math.log(prob)))
        if prob_q == 0.0:
            prob_q = Automaton.eps
        dist += Automaton.kullback(prob, prob_q)
    return dist

def main():
    code = AbstractCode.read(open(sys.argv[1]))
    corp = read_corpus(open(sys.argv[2]), separator="#")
    normalize_corpus(corp)
    probs = corp.values()
    dist = compute_entropy(probs, code)
    print dist

if __name__ == "__main__":
    main()

