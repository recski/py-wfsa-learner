"""
This module reads a wfsa from a dump, encodes it using the coding specified, compares the result to a corpus and counts the coded wfsa's minimum description length
"""
from automaton import Automaton, Code
from corpus import read_corpus, normalize_corpus
import logging
import math
import sys

def get_weights(automaton, discard_ones_zeros=False):
    weights = {}
    for state1, transitions in automaton.m.iteritems():
        for state2, log_weight in transitions.iteritems():
            if discard_ones_zeros and log_weight in (float("-inf"), 0):
                continue
            w = math.exp(log_weight)
            weights[(state1, state2)] = w

    return weights

#state_no = 3
#morpheme_no = 26
#word_no = 135

def count_mdl(r_automaton, corpus, bit_no):
    kl_err = r_automaton.distance_from_corpus(
             corpus, getattr(Automaton, 'kullback'))
    weights = get_weights(r_automaton, discard_ones_zeros=True)
    #for pair, weight in weights.iteritems():
    #    print pair, weight
    param_no = len(weights)
    err = kl_err*word_no
    #sys.stderr.write(str(kl_err)+' ')
    bit_per_param = math.log(state_no**2)+math.log(morpheme_no)+bit_no
    mdl = (param_no*bit_per_param)+err
    return param_no, bit_per_param, param_no*bit_per_param, kl_err, err, mdl

def main():
    corpus = read_corpus(sys.stdin, '#')
    corpus = normalize_corpus(corpus)
    automaton = Automaton.create_from_dump(sys.argv[1])
    automaton.code = Code.create_from_file(sys.argv[2])
    automaton.round_and_normalize()
    param_no, bit_per_param, bits, kl_err, err, mdl = count_mdl(automaton, corpus, automaton.code.bit_no)
    print 'params:', param_no, 'bit/param:', bit_per_param, 'total bits:', bits
    print 'words:', word_no, 'kl_err:', kl_err, 'total err:', err
    print 'mdl:', mdl

def main_multi():
    corpus = read_corpus(sys.stdin, '#')
    corpus = normalize_corpus(corpus)
    mdls = []
    errs = []
    bits = []
    for i in range(1, 13):
        sys.stderr.write('i='+str(i)+' ')
        mdls.append([])
        errs.append([])
        bits.append([])
        for j in range(-4, -33, -1):
            automaton = Automaton.create_from_dump(sys.argv[1])
            automaton.code = Code.create_from_file("codes/{0}.{1}.code".format(i, j))
            automaton.round_and_normalize()
            param_no, bit_per_param, bit, kl_err, err, mdl = count_mdl(automaton, corpus, automaton.code.bit_no)
            mdls[-1].append(mdl)
            errs[-1].append(err)
            bits[-1].append(bit)
    sys.stderr.write('\n')
    fsa_name = sys.argv[1].split('/')[-1].split('.')[0]
    m = open('stats/{0}.mdl'.format(fsa_name), 'w')
    m.write('\n'.join([' '.join([str(v) for v in line]) for line in mdls]))
    m = open('stats/{0}.err'.format(fsa_name), 'w')
    m.write('\n'.join([' '.join([str(v) for v in line]) for line in errs]))
    m = open('stats/{0}.bit'.format(fsa_name), 'w')
    m.write('\n'.join([' '.join([str(v) for v in line]) for line in bits]))

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    #main_multi()
    main()
