"""With this script, we can create an automaton that has
a start state, a final state, and one state for every _word_ in the
corpus"""
import sys
import math

from automaton import Automaton
from corpus import read_corpus, normalize_corpus
from quantizer import LogLinQuantizer

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

def main():
    corpus = read_corpus(open(sys.argv[1]), separator="#")
    normalize_corpus(corpus)
    wfsa = create_word_wfsa(corpus)
    wfsa.finalize()
    if len(sys.argv) == 4:
        wfsa.quantizer = LogLinQuantizer(int(sys.argv[2]), int(sys.argv[3]))
        wfsa.round_and_normalize()
    wfsa.dump(sys.stdout)

if __name__ == "__main__":
    main()


