import sys
import os
import logging

sys.path.insert(0, sys.path[0].rsplit("/", 1)[0])
from corpus import read_corpus, normalize_corpus
from create_baseline_automaton import create_word_wfsa
from create_3state_wfsa import create_new_three_state_fsa
from quantizer import LogLinQuantizer
from copy import deepcopy as copy
from encoder import Encoder
from learner import Learner
from automaton import Automaton

def generate_quantizers(bits, cutoffs):
    for bits in bits:
        for cutoff in cutoffs:
            quantizer = LogLinQuantizer(bits, cutoff)
            yield quantizer

def generate_options(bits, cutoffs, distances, emissions, type_):
    for q in generate_quantizers(bits, cutoffs):
        for d in distances:
            for e in emissions:
                for t in type_:
                    yield q, d, e, t

def create_corpora(corpus_fn):
    unigram_corpus = read_corpus(open(corpus_fn), skip=["#"])
    normalize_corpus(unigram_corpus)

    morpheme_corpus = read_corpus(open(corpus_fn), "#")
    normalize_corpus(morpheme_corpus)
    return unigram_corpus, morpheme_corpus

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

class Exp(object):
    def __init__(self, corpus_fn, workdir):
        self.workdir = workdir
        
        self.unigram_corpus, self.morpheme_corpus = create_corpora(corpus_fn)

        self.unigram_encoder = Encoder(4.7872)
        self.morpheme_encoder = Encoder(3.1196)

    def run_list_exp(self, quantizer, emission):
        exp_name = "{0}-{1}-l-{2}".format(
            quantizer.bits,
            abs(quantizer.neg_cutoff),
            emission)

        logging.info("Running {0}".format(exp_name))

        if emission == "m":
            corpus = self.morpheme_corpus
            encoder = self.morpheme_encoder
        elif emission == "c":
            corpus = self.unigram_corpus
            encoder = self.unigram_encoder
        wfsa = create_word_wfsa(corpus)
        wfsa.finalize()
        wfsa.quantizer = quantizer
        bits_a, bits_e, bits_t, err = encode_wfsa(wfsa, corpus, encoder)

        res = [exp_name, bits_a, bits_e, bits_t, err]
        return res

    def run_learn_exp(self, quantizer, distance, harant, emissions):
        exp_name = "{0}-{1}-{2}-{3}-{4}".format(
            quantizer.bits,
            abs(quantizer.neg_cutoff),
            harant,
            emissions,
            distance[0])

        logging.info("Running {0}".format(exp_name))

        learnt_wfsa_filename = "{0}/{1}".format(self.workdir,
            "learnt_{0}.wfsa".format(exp_name))

        # read Automaton or learn it and dump it finally
        if os.path.exists(learnt_wfsa_filename):
            # read already learnt automaton
            learnt_wfsa = Automaton.create_from_dump(learnt_wfsa_filename)
            learnt_wfsa.quantizer = quantizer
            learnt_wfsa.round_and_normalize()
        else:
            # create and learn new automaton
            wfsa = create_new_three_state_fsa(self.morpheme_corpus,
                                              harant, emissions)
            wfsa.finalize()
            wfsa.quantizer = quantizer
            wfsa.round_and_normalize()
            corpus = (self.morpheme_corpus if emissions == "m" else
                      self.unigram_corpus)
            learnt_wfsa = learn_wfsa(wfsa, corpus, distance)

            # dump
            with open(learnt_wfsa_filename, "w") as of:
                learnt_wfsa.dump(of)

        # encode automaton
        encoder = (self.morpheme_encoder if emissions=="m" else
                   self.unigram_encoder)
        bits_a, bits_e, bits_t, err = encode_wfsa(
            learnt_wfsa, self.morpheme_corpus, encoder)

        return [exp_name, bits_a, bits_e, bits_t, err]

def run_exp(args):
    (exp, quantizer, distance, emission, type_) = args

    if type_ == "l":
        return exp.run_list_exp(quantizer, emission)
    elif type_ in ["3", "h", "a"]:
        return exp.run_learn_exp(quantizer, distance, type_, emission)

