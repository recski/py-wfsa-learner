"""For the corpus data/lemma_count this script computes the following:
- creates a baseline automaton with create_baseline_automaton.py
- creates a 3-state automaton with create_3state_wfsa.py
- run learner on the latter
- encode both automata with encoder.py based on Code based on different
  entropies e.g. character and morpheme entropy, that are given"""

import sys
import logging

sys.path.insert(0, sys.path[0].rsplit("/", 1)[0])
from corpus import read_corpus, normalize_corpus
from create_baseline_automaton import create_word_wfsa
from create_3state_wfsa import create_new_three_state_fsa
from code import LogLinCode
from copy import copy
from encoder import Encoder
from learner import Learner

def generate_codes():
    for bits in [4, 5, 6, 7, 8, 10, 12, 16]:
    #for bits in [4]:
        for cutoff in [-11, -13, -15, -17, -20, -24, -28, -32]:
        #for cutoff in [-11]:
            code = LogLinCode(bits, cutoff)
            yield code

def main():
    corpus_fn = "../data/lemma_count"
    unigram_corpus = read_corpus(open(corpus_fn))
    unigram_corpus = dict([(k.replace("#", ""), v)
                           for k, v in unigram_corpus.iteritems()])
    normalize_corpus(unigram_corpus)
    morpheme_corpus = read_corpus(open(corpus_fn), "#")
    normalize_corpus(morpheme_corpus)

    baseline_unigram_wfsa = create_word_wfsa(unigram_corpus)
    baseline_unigram_wfsa.finalize()
    baseline_morpheme_wfsa = create_word_wfsa(morpheme_corpus)
    baseline_morpheme_wfsa.finalize()
    three_states_wfsa = create_new_three_state_fsa(morpheme_corpus)
    three_states_wfsa.finalize()
    morpheme_encoder = Encoder(3.1196)
    unigram_encoder = Encoder(4.7872)
    unigram_tuple_corpus = dict([((k, ), v) for k, v in unigram_corpus.iteritems()])
    #morpheme_tuple_corpus = dict([((k, ), v) for k, v in unigram_corpus.iteritems()])

    for code in generate_codes():
        logging.info("Running {0} {1}".format(code.bits, code.neg_cutoff))
        print code.bits, code.neg_cutoff,
        wfsa = copy(baseline_unigram_wfsa)
        wfsa.code = code
        wfsa.round_and_normalize()
        uni_bits, uni_err, uni_all = unigram_encoder.encode(wfsa,
            unigram_tuple_corpus)
        print uni_bits, uni_err, uni_all,

        wfsa = copy(baseline_morpheme_wfsa)
        wfsa.code = code
        wfsa.round_and_normalize()
        morph_bits, morph_err, morph_all = morpheme_encoder.encode(wfsa,
            morpheme_corpus)
        quit()
        print morph_bits, morph_err, morph_all,

        for distfp in ["kullback", "l1err", "squarerr"]:
            wfsa = copy(three_states_wfsa)
            wfsa.code = code
            learner = Learner(wfsa, morpheme_corpus, pref_prob=0.0,
                distfp=distfp, turns_for_each=40, factor=0.8,
                start_temp=1e-5, end_temp=1e-7, tempq=0.9)
            learner.main()
            morph_bits, morph_err, morph_all = morpheme_encoder.encode(wfsa,
                morpheme_corpus)
            print distfp, morph_bits, morph_err, morph_all,
        print


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
