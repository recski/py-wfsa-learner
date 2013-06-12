"""For the corpus data/lemma_count this script computes the following:
- creates a baseline automaton with create_baseline_automaton.py
- creates a 3-state automaton with create_3state_wfsa.py
- run learner on the latter
- encode both automata with encoder.py based on Quantizer based on different
  entropies e.g. character and morpheme entropy, that are given"""

import sys
import logging
from multiprocessing import Pool

from exp import Exp, generate_options, run_exp

def main(wd):
    corpus_fn = "../data/lemma_count.tab"

    exp = Exp(corpus_fn, wd)

    pool = Pool(processes=1)
    levels = [14] + [2**b for b in [6, 7, 8, 9, 10, 11, 12, 14, 16]]
    levels = [14]
    cutoffs = [-20]
    distances = ["kullback"]
    emissions = ["c", "m"]
    emissions = ["c"]
    type_ = ["l", "3", ["hogy"], "a"]
    type_ = ["l"]
    state_bits = ["u", "e"]
    state_bits = ["u"]
    options = list(generate_options(levels, cutoffs, distances, emissions, type_,
                                    state_bits))
    res = pool.map(run_exp, [(exp,) + o for o in options])
    for r in res:
        r = [("{0:4.5}".format(_) if type(_) == float else str(_)) for _ in r]
        print "\t".join(r)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wd = "."
    if len(sys.argv) > 1:
        wd = sys.argv[1]
    main(wd)
