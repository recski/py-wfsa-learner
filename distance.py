import sys

from automaton import Automaton
from corpus import read_corpus, normalize_corpus

def main():
    # read automaton
    wfsa = Automaton.create_from_dump(open(sys.argv[1]))
    # read corpus
    corpus = read_corpus(open(sys.argv[2]), separator=sys.argv[3], skip=[sys.argv[4]])
    normalize_corpus(corpus)
    # call distance_from_corpus
    distances = {}
    dist = wfsa.distance_from_corpus(corpus, Automaton.kullback, distances=distances)
    # print out result
    for k, v in distances.iteritems():
        print k, v

if __name__ == "__main__":
    main()
