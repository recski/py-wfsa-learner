import sys
import os
import math
import logging

sys.path.insert(0, sys.path[0].rsplit("/", 1)[0])
from corpus import read_corpus, normalize_corpus, get_alphabet
from create_baseline_automaton import create_word_wfsa
from create_3state_wfsa import create_new_three_state_fsa
from quantizer import LogLinQuantizer
from copy import deepcopy as copy
from encoder import Encoder
from learner import Learner
from automaton import Automaton

def generate_quantizers(levels, cutoffs):
    for level in levels:
        for cutoff in cutoffs:
            quantizer = LogLinQuantizer(level, cutoff)
            yield quantizer

def generate_options(levels, cutoffs, distances, emissions, type_, state_bits):
    for s in state_bits:
        for q in generate_quantizers(levels, cutoffs):
            for d in distances:
                for e in emissions:
                    for t in type_:
                        yield q, d, e, t, s

def create_corpora(corpus_fn):
    unigram_corpus = read_corpus(open(corpus_fn), skip=["#"])
    normalize_corpus(unigram_corpus)

    morpheme_corpus = read_corpus(open(corpus_fn), "#")
    normalize_corpus(morpheme_corpus)
    return unigram_corpus, morpheme_corpus

def learn_wfsa(wfsa, corpus, distfp=None, checkpoint=None):
    wfsa_ = copy(wfsa)
    wfsa_.round_and_normalize()
    if not checkpoint:
        checkpoint = lambda x: wfsa_.dump(open('{0}.wfsa'.format(x), 'w'))
    if distfp is not None:
        if wfsa_.quantizer is not None:
            bits = int(round(math.log(wfsa_.quantizer.levels, 2)))
            f = [2 ** i for i in xrange(max(0, bits-5), -1, -1)]
            t = [1e-5/2**i for i in xrange(0, max(1, bits-4))]
            learner = Learner(wfsa_, corpus, checkpoint, pref_prob=0.0,
                distfp=distfp, turns_for_each=300, factors=f,
                temperatures=t)
        else:
            # continuous case, not implemented
            pass
        learner.main()
    logging.debug("WFSA learnt")
    return wfsa_

def encode_wfsa(wfsa, corpus, encoder):
    return encoder.encode(copy(wfsa), corpus)

def checkpoint_dump(wfsa, name, *args):
    cp_wfsa_fn = "{0}_{1}.wfsa".format(name,
        "_".join(str(_) for _ in args))
    with open(cp_wfsa_fn, "w") as of:
        wfsa.dump(of)

class Exp(object):
    def __init__(self, corpus_fn, workdir):
        self.workdir = workdir
        
        self.unigram_corpus, self.morpheme_corpus = create_corpora(corpus_fn)

        self.unigram_encoder = Encoder(4.7872)
        self.morpheme_encoder = Encoder(3.1196)

    def run_list_exp(self, quantizer, emission, state_bits):
        aut_name = "{0}-{1}-l-{2}".format(
            quantizer.levels,
            abs(quantizer.neg_cutoff),
            emission)
        exp_name = "{0}-{1}".format(aut_name, state_bits)

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
        encoder.state_bits = state_bits
        bits_a, bits_e, bits_t, err, hq, tc = encode_wfsa(wfsa,
                                                          corpus, encoder)

        res = [exp_name, bits_a, bits_e, bits_t, err, hq, tc]
        return res

    def run_3state_exp(self, quantizer, distance, harant, emissions,
                       state_bits):
        aut_name = "{0}-{1}-{2}-{3}-{4}".format(
            quantizer.levels,
            abs(quantizer.neg_cutoff),
            "_".join(("@".join(h) if type(h) == tuple else h) for h in harant),
            emissions,
            distance[0])
        exp_name = "{0}-{1}".format(aut_name, state_bits)

        logging.info("Running {0}".format(exp_name))

        learnt_wfsa_filename = "{0}/{1}".format(self.workdir,
            "learnt_{0}.wfsa".format(aut_name))

        corpus = (self.morpheme_corpus if emissions == "m" else
                  self.unigram_corpus)

        # read Automaton or learn it and dump it finally
        if os.path.exists(learnt_wfsa_filename):
            # read already learnt automaton
            learnt_wfsa = Automaton.create_from_dump(open(learnt_wfsa_filename))
            learnt_wfsa.quantizer = quantizer
            learnt_wfsa.round_and_normalize()
        else:
            # create and learn new automaton
            wfsa = create_new_three_state_fsa(self.morpheme_corpus,
                                              harant, emissions)
            wfsa.finalize()
            wfsa.quantizer = quantizer
            wfsa.round_and_normalize()
            cp = lambda *x: checkpoint_dump(wfsa, 
                "{0}/cp_{1}".format(self.workdir, aut_name), *x)
            learnt_wfsa = learn_wfsa(wfsa, corpus, distance, cp)

            # dump
            with open(learnt_wfsa_filename, "w") as of:
                learnt_wfsa.dump(of)

        # encode automaton
        encoder = (self.morpheme_encoder if emissions=="m" else
                   self.unigram_encoder)
        encoder.state_bits = state_bits
        bits_a, bits_e, bits_t, err, hq, tc = encode_wfsa(
            learnt_wfsa, corpus, encoder)

        return [exp_name, bits_a, bits_e, bits_t, err, hq, tc]

    def run_uniform_exp(self, quantizer, distance, emissions, state_bits, entropy):
        exp_name = "{0}-{1}-{2}-{3}-{4}".format(
            quantizer.levels,
            abs(quantizer.neg_cutoff),
            'm',
            emissions,
            distance[0])

        logging.info("Running {0}".format(exp_name))
        learnt_wfsa_filename = "{0}/{1}".format(self.workdir,
            "learnt_{0}.wfsa".format(exp_name))

        corpus = (self.morpheme_corpus if emissions == "m" else
                  self.unigram_corpus)

        # read Automaton or learn it and dump it finally
        if os.path.exists(learnt_wfsa_filename):
            # read already learnt automaton
            learnt_wfsa = Automaton.create_from_dump(open(learnt_wfsa_filename))
            learnt_wfsa.quantizer = quantizer
            learnt_wfsa.round_and_normalize()
        else:
            # create and learn new automaton
            alphabet = get_alphabet(corpus)
            numbers_per_letters = dict([(letter, 1)
                                        for letter in alphabet])
            #print numbers_per_letters
            wfsa = Automaton.create_uniform_automaton(numbers_per_letters)
            wfsa.finalize()
            wfsa.quantizer = quantizer
            wfsa.round_and_normalize()
            cp = lambda *x: checkpoint_dump(wfsa, 
                "{0}/cp_{1}".format(self.workdir, exp_name), *x)
            logging.info('learning starts here')
            learnt_wfsa = learn_wfsa(wfsa, corpus, distance, cp)

            # dump
            with open(learnt_wfsa_filename, "w") as of:
                learnt_wfsa.dump(of)

        # encode automaton
        encoder = Encoder(entropy)
        bits_a, bits_e, bits_t, err, hq, tc = encode_wfsa(
            learnt_wfsa, corpus, encoder)
        return [exp_name, bits_a, bits_e, bits_t, err, hq, tc]

    def run_sze_tok_exp(self, quantizer, distance, emissions, state_bits):
        entropy = 0.933201
        return self.run_uniform_exp(quantizer, distance, emissions, state_bits,
                                entropy)

    def run_sze_type_exp(self, quantizer, distance, emissions, state_bits):
        entropy = 1.56655
        return self.run_uniform_exp(quantizer, distance, emissions, state_bits,
                                entropy)

    def run_mnsz_tok_exp(self, quantizer, distance, emissions, state_bits):
        entropy = 2.914877
        return self.run_uniform_exp(quantizer, distance, emissions, state_bits,
                                entropy)
        pass

def run_exp(args):
    (exp, quantizer, distance, emission, type_, state_bits) = args
    
    if type_ == "l":
        return exp.run_list_exp(quantizer, emission, state_bits)
    elif type_ in ["3", "a"] or type(type_) == list:
        return exp.run_3state_exp(quantizer, distance, type_, emission,
                                 state_bits)
    elif type_ == "sze_toks":
        return exp.run_sze_tok_exp(quantizer, distance, emission, state_bits)
    elif type_ == "sze_types":
        return exp.run_sze_type_exp(quantizer, distance, emission, state_bits)
    elif type_ == "mnsz_toks":
        return exp.run_mnsz_tok_exp(quantizer, distance, emission, state_bits)
    
        

