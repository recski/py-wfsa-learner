"""For the corpus data/lemma_count this script computes the following:
- creates a baseline automaton with create_baseline_automaton.py
- creates a 3-state automaton with create_3state_wfsa.py
- run learner on the latter
- encode both automata with encoder.py based on Quantizer based on different
  entropies e.g. character and morpheme entropy, that are given"""

import os
import sys
import logging
from multiprocessing import Pool

sys.path.insert(0, sys.path[0].rsplit("/", 1)[0])
from corpus import read_corpus, normalize_corpus
from create_baseline_automaton import create_word_wfsa
from create_3state_wfsa import create_new_three_state_fsa
from quantizer import LogLinQuantizer
from copy import deepcopy as copy
from encoder import Encoder
from learner import Learner
from automaton import Automaton

def generate_quantizers():
    #for bits in [4, 5, 6, 7, 8, 10, 12, 16]:
        #for cutoff in [-4,-5,-6,-7,-8,-9,-10, -12, -14, -16, -20, -24, -28, -32]:
    for bits in [10, 12]:
        for cutoff in [-20, -24, -28, -32]:
            quantizer = LogLinQuantizer(bits, cutoff)
            yield quantizer

def create_corpora(corpus_fn):
    unigram_corpus = read_corpus(open(corpus_fn))
    normalize_corpus(unigram_corpus)

    morpheme_corpus = read_corpus(open(corpus_fn), "#")
    normalize_corpus(morpheme_corpus)
    return unigram_corpus, morpheme_corpus

def create_wfsas(unigram_corpus, morpheme_corpus):
    baseline_unigram_wfsa = create_word_wfsa(unigram_corpus)
    baseline_unigram_wfsa.finalize()
    baseline_morpheme_wfsa = create_word_wfsa(morpheme_corpus)
    baseline_morpheme_wfsa.finalize()
    three_states_wfsa = create_new_three_state_fsa(morpheme_corpus)
    three_states_wfsa.finalize()
    return baseline_unigram_wfsa, baseline_morpheme_wfsa, three_states_wfsa

def learn_wfsa(wfsa, corpus, distfp=None):
    wfsa_ = copy(wfsa)
    wfsa_.round_and_normalize()
    if distfp is not None:
        learner = Learner(wfsa_, corpus, pref_prob=0.0,
            distfp=distfp, turns_for_each=50, factor=0.8,
            start_temp=1e-5, end_temp=1e-7, tempq=0.9)
        learner.main()
    return wfsa_

def encode_wfsa(wfsa, corpus, encoder):
    return encoder.encode(copy(wfsa), corpus)

def run(args):
    (quantizer, wd, unigram_corpus, morpheme_corpus, unigram_wfsa,
     morpheme_wfsa, three_states_wfsa, unigram_encoder,
     morpheme_encoder) = args

    # set quantizers
    unigram_wfsa.quantizer = quantizer
    morpheme_wfsa.quantizer = quantizer
    three_states_wfsa.quantizer = quantizer

    res = []
    logging.info("Running {0} {1}".format(quantizer.bits,
                                          quantizer.neg_cutoff))
    res += [quantizer.bits, quantizer.neg_cutoff]

    #uni_bits_a, uni_bits_e, uni_bits_t, uni_err = encode_wfsa(
        #unigram_wfsa, unigram_corpus, unigram_encoder)
    #res += [uni_bits_a, uni_bits_e, uni_bits_t, uni_err]
#
    #morph_bits_a, morph_bits_e, morph_bits_t, morph_err = encode_wfsa(
        #morpheme_wfsa, morpheme_corpus, morpheme_encoder)
    #res += [morph_bits_a, morph_bits_e, morph_bits_t, morph_err]
#
    for distfp in ["kullback", "l1err", "squarerr"]:
        learnt_wfsa_filename = "{0}/{1}".format(wd,
            "learnt_{0}_{1}_{2}.wfsa".format(quantizer.bits,
                                            quantizer.neg_cutoff,
                                            distfp))
        if os.path.exists(learnt_wfsa_filename):
            learnt_wfsa = Automaton.create_from_dump(learnt_wfsa_filename)
            learnt_wfsa.quantizer = quantizer
            learnt_wfsa.round_and_normalize()
        else:
            learnt_wfsa = learn_wfsa(three_states_wfsa, morpheme_corpus, distfp)
            with open(learnt_wfsa_filename, "w") as of:
                learnt_wfsa.dump(of)
        morph_bits_a, morph_bits_e, morph_bits_t, morph_err = encode_wfsa(
            learnt_wfsa, morpheme_corpus, morpheme_encoder)
        res += [distfp, morph_bits_a, morph_bits_e, morph_bits_t, morph_err]
    return res

def main(wd):
    corpus_fn = "../data/lemma_count.tab"
    unigram_corpus, morpheme_corpus = create_corpora(corpus_fn)
    (baseline_unigram_wfsa, baseline_morpheme_wfsa,
        three_states_wfsa) = create_wfsas(unigram_corpus, morpheme_corpus)

    unigram_encoder = Encoder(4.7872)
    morpheme_encoder = Encoder(3.1196)
    arguments = (unigram_corpus, morpheme_corpus,
        baseline_unigram_wfsa, baseline_morpheme_wfsa, three_states_wfsa,
        unigram_encoder, morpheme_encoder)

    pool = Pool(processes=6)
    quantizers = list(generate_quantizers())
    res = pool.map(run, [(q,wd,) + arguments for q in quantizers])
    for r in res:
        r = [("{0:4.5}".format(_) if type(_) == float else str(_)) for _ in r]
        print "\t".join(r)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wd = "."
    if len(sys.argv) > 1:
        wd = sys.argv[1]
    main(wd)
