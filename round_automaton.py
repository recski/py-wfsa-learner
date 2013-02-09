from automaton import Automaton
from round_freqs import get_error
from corpus import read_corpus, normalize_corpus
import math
import sys

def my_log(x):
    if x == 0:
        return float('-inf')
    else:
        return math.log(x)

def bit_round(x, n):
    return round(x*((2**n)-1))/((2**n)-1)


def round_32(weight, bit_no):
    if weight in (0, 1): return weight
    to_round = math.log(weight)/-32
    rounded = bit_round(to_round, bit_no)
    rounded_weight = math.exp(rounded*-32)
    print weight, rounded_weight
    return rounded_weight

def round_transitions(transitions, bit_no, state1):
    tr_by_weight = [(math.exp(log_weight), state)
                    for (state, log_weight) in transitions.iteritems()]
    tr_by_weight.sort()
    tr_by_weight.reverse()
    largest_prob, largest_prob_state = tr_by_weight[0]
    rounded_transitions = dict([(state, round_32(weight, bit_no))
                                for (weight, state) in tr_by_weight[1:]])
    #print sum(rounded_transitions.values())
    new_largest_prob = 1-sum(rounded_transitions.values())
    #if new_largest_prob < 1:
    #    print state1, largest_prob_state, largest_prob, new_largest_prob
    #if largest_prob<0:
    #    print largest_prob_state, foo, largest_prob
    #    for state, weight in rounded_transitions.iteritems():
    #        print state, weight
    #    quit()
    rounded_transitions[largest_prob_state] = new_largest_prob
    rounded_transitions = dict([(state, my_log(weight))
                                for state, weight in
                                rounded_transitions.iteritems()])
    return rounded_transitions

def round_automaton(a, bit_no, file_name):
    r_automaton = Automaton.create_from_dump(file_name)
    for state1, transitions in a.m.iteritems():
        rounded_transitions = round_transitions(transitions, bit_no, state1)
        r_automaton.m[state1] = rounded_transitions
    return r_automaton

def get_weights(automaton, discard_ones_zeros=False):
    weights = {}
    for state1, transitions in automaton.m.iteritems():
        for state2, log_weight in transitions.iteritems():
            if discard_ones_zeros and log_weight in (float("-inf"), 0):
                continue
            w = math.exp(log_weight)
            weights[(state1, state2)] = w

    return weights

state_no = 3
morpheme_no = 26
word_no = 135
def count_mdl(automaton, r_automaton, input, file_name, bit_no):
    corpus = read_corpus(input, '#')
    corpus = normalize_corpus(corpus)
    kl_err = r_automaton.distance_from_corpus(
             corpus, getattr(Automaton, 'kullback'))
    weights = get_weights(r_automaton, discard_ones_zeros=True)
    #for pair, weight in weights.iteritems():
    #    print pair, weight
    param_no = len(weights)
    err = kl_err*word_no
    bit_per_param = math.log(state_no**2)+math.log(morpheme_no)+bit_no
    mdl = (param_no*bit_per_param)+err
    print file_name, param_no, bit_no, kl_err, mdl

def count_error(automaton, r_automaton, file_name, bit_no):
    a_weights = get_weights(automaton)
    r_a_weights = get_weights(r_automaton)
    #for p, w in a_weights.iteritems():
    #    print p, w, r_a_weights[p]
    print file_name, bit_no,
    for metric_name in ['l1err', 'squarerr', 'kullback']:
        error = get_error(a_weights, r_a_weights, metric_name)
        print metric_name, error,
    print

def main():
    bit_no = int(sys.argv[2])
    file_name = sys.argv[1]
    automaton = Automaton.create_from_dump(file_name)
    r_automaton = round_automaton(automaton, bit_no, file_name)
    count_mdl(automaton, r_automaton, sys.stdin, file_name, bit_no)
    #count_error(automaton, r_automaton, file_name, bit_no)
    
if __name__=='__main__':
    main()