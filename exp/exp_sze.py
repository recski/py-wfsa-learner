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
    corpus_fn = sys.argv[2]

    exp = Exp(corpus_fn, wd)

    pool = Pool(processes=8)
    #bits = [6]
    #bits = [6, 7, 8, 9, 10, 11, 12, 14, 16]
    bits = [11,12,14,16]
    cutoffs = [-20]
    distances = ["kullback"]
    emissions = ["c"]
    type_ = [sys.argv[3]]
    state_bits = ["u", "e"]
    options = list(generate_options(bits, cutoffs, distances, emissions, type_, state_bits))
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
