""" Simple script that excepts a quantizer dump and a corpus, reads them
and counts the entropy of the probabilities and quantized probs."""

import sys
import math

from quantizer import AbstractQuantizer
from corpus import read_corpus, normalize_corpus
from automaton import Automaton

def compute_entropy(probs, quantizer):
    dist = 0.0
    modeled_sum = sum([math.exp(quantizer.representer(math.log(prob))) for prob in probs])
    for prob in probs:
        prob_q = math.exp(quantizer.representer(math.log(prob)))
        if prob_q == 0.0:
            prob_q = Automaton.eps
        prob_q /= modeled_sum
        dist += Automaton.kullback(prob, prob_q)
        #print prob, prob_q, Automaton.kullback(prob, prob_q)
    return dist

def main():
    quantizer = AbstractQuantizer.read(open(sys.argv[1]))
    corp = read_corpus(open(sys.argv[2]), separator="#")
    normalize_corpus(corp)
    probs = corp.values()
    dist = compute_entropy(probs, quantizer)
    print dist

if __name__ == "__main__":
    main()

