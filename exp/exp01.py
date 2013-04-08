"""For the corpus data/lemma_count this script computes the following:
- creates a baseline automaton with create_baseline_automaton.py
- creates a 3-state automaton with create_3state_wfsa.py
- run learner on the latter
- encode both automata with encoder.py based on Quantizer based on different
  entropies e.g. character and morpheme entropy, that are given"""

import sys
import logging
from multiprocessing import Pool

sys.path.insert(0, sys.path[0].rsplit("/", 1)[0])
from corpus import read_corpus, normalize_corpus
from create_baseline_automaton import create_word_wfsa
from create_3state_wfsa import create_new_three_state_fsa
from quantizer import LogLinQuantizer
from copy import copy
from encoder import Encoder
from learner import Learner

def generate_quantizers():
    for bits in [4, 5, 6, 7, 8, 10, 12, 16]:
        for cutoff in [-11, -13, -15, -17, -20, -24, -28, -32]:
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

def encode_learn_wfsa(wfsa, quantizer, corpus, encoder, distfp=None):
    """encode a wfsa based on arguments and run a learning on it if
    a distance parameter is given
    """
    wfsa = copy(wfsa)
    wfsa.quantizer = quantizer
    wfsa.round_and_normalize()
    if distfp is not None:
        learner = Learner(wfsa, corpus, pref_prob=0.0,
            distfp=distfp, turns_for_each=40, factor=0.8,
            start_temp=1e-5, end_temp=1e-7, tempq=0.9)
        learner.main()
    return encoder.encode(wfsa, corpus)

def run(args):
    (quantizer, unigram_corpus, morpheme_corpus, unigram_wfsa, morpheme_wfsa,
       three_states_wfsa, unigram_encoder, morpheme_encoder) = args
    res = []
    logging.info("Running {0} {1}".format(quantizer.bits,
                                          quantizer.neg_cutoff))
    res += [quantizer.bits, quantizer.neg_cutoff]

    uni_bits, uni_err, uni_all = encode_learn_wfsa(
        unigram_wfsa, quantizer, unigram_corpus,
        unigram_encoder)
    res += [uni_bits, uni_err, uni_all]

    morph_bits, morph_err, morph_all = encode_learn_wfsa(
        morpheme_wfsa, quantizer, morpheme_corpus, 
        morpheme_encoder)
    res += [morph_bits, morph_err, morph_all]

    for distfp in ["kullback", "l1err", "squarerr"]:
        morph_bits, morph_err, morph_all = encode_learn_wfsa(
            three_states_wfsa, quantizer, morpheme_corpus, 
            morpheme_encoder, distfp)
        res += [distfp, morph_bits, morph_err, morph_all]
    return res

def main():
    corpus_fn = "../data/lemma_count"
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
    res = pool.map(run, [(q,) + arguments for q in quantizers])
    for r in res:
        print "\t".join((str(_) for _ in r))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
