import math
import sys
from automaton import Automaton

def round(freq, bit_no, log=True):
    if freq in (0, 1): return freq
    if log:
        log_freq = math.log(freq)
        rounded_log_freq = math.floor((2**bit_no)*log_freq)/(2**bit_no)
        rounded_freq = math.exp(rounded_log_freq)
    else:
        rounded_freq = math.floor((2**bit_no)*freq)/(2**bit_no)
    
    return rounded_freq

def normalize_corp(corp):
    total = float(sum(corp.values()))
    return dict([(word, freq/total) for (word, freq) in corp.iteritems()])

def get_error(corp, r_corp, metric_name):
    metric_function = getattr(Automaton, metric_name)
    err = 0
    for word, prob in corp.iteritems():
        r_prob = r_corp[word]
        if metric_name == 'kullback' and r_prob == 0:
            r_prob = 1e-50
        err+=metric_function(prob, r_prob)
    return err

def main():
    corp = {}
    rounded_corp = {}
    bit_no = int(sys.argv[1])
    for line in sys.stdin:
        word, freq = line.strip().split()
        corp[word] = float(freq)
        rounded_freq = round(float(freq), bit_no)
        rounded_corp[word] = rounded_freq
        n_corp = normalize_corp(corp)
        n_r_corp = normalize_corp(rounded_corp)
    for metric_name in ['l1err', 'squarerr', 'kullback']:
        error = get_error(n_corp, n_r_corp, metric_name)
        print metric_name, error

if __name__ == '__main__':
    main()
