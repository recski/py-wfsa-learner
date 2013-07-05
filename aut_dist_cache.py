import sys
import math
from collections import defaultdict

from misc import closure_and_top_sort
from automaton import Automaton
from corpus import read_corpus

class DistanceCache(object):
    def __init__(self, automaton, corpus):
        self.aut = automaton
        self.corpus = corpus
        self.build_paths()

    def build_paths(self):
        aut = self.aut
        self.paths = defaultdict(set)
        self.paths[()].add(("^",))
        needed = set(self.corpus.keys())
        top_sort_needed = set(closure_and_top_sort(self.corpus.keys()))

        # iterate through all the paths with DFS-like algorithm
        # and prune if needed
        while len(needed) > 0:
            for s, paths in self.paths.items():
                for path in paths.copy():
                    last = path[-1]
                    if last not in aut.m:
                        continue

                    for tgt in aut.m[last]:
                        new_path = path + (tgt,)
                        emission = (aut.emissions[tgt] if tgt in aut.emissions
                                    else ())
                        new_s = s + emission
                        if new_s not in top_sort_needed:
                            continue
                        self.paths[new_s].add(new_path)
                        if s in needed:
                            needed.remove(s)

        needed = set(self.corpus.keys())
        self.paths = dict((s, set([p for p in paths if p[-1] == "$"]))
                      for s, paths in self.paths.iteritems() if s in needed)

    def prob_of_string(self, s):
        p = Automaton.m_inf
        for path in self.paths[s]:
            prob_of_path = 0
            prev = path[0]
            for next_ in path[1:]:
                prob_of_path += self.aut.m[prev][next_]
                prev = next_

            p = max(p, prob_of_path)
        return p
    
    def distance(self, distfp, reverse=False):
        distance = 0.0

        # calculating probabilities for strings
        for item, corpus_p in self.corpus.iteritems():
            if corpus_p > 0.0:
                modeled_p = math.exp(self.prob_of_string(item))
                if modeled_p == 0.0:
                    modeled_p = 1e-50

                dist = (distfp(corpus_p, modeled_p) if not reverse
                             else distfp(modeled_p, corpus_p))
                distance += dist
        return distance

def main():
    automaton = Automaton.create_from_dump(open(sys.argv[1]))
    corpus = read_corpus(open(sys.argv[2]), "#")
    dc = DistanceCache(automaton, corpus)
    dc.build_paths()

if __name__ == "__main__":
    main()
