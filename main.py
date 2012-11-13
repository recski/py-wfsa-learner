"""The entry point of the program."""

import sys
from optparse import OptionParser
import logging

from learner import Learner
from corpus import get_alphabet, read_corpus, normalize_corpus, read_dict
from automaton import Automaton

def main(options):
    # TODO refactor unreadable if branches
    corpus = read_corpus(sys.stdin, options.separator)
    corpus = normalize_corpus(corpus)

    alphabet = get_alphabet(corpus)

    initial_transitions = None
    if options.initial_transitions:
        initial_transitions = Automaton.read_transitions(options.initial_transitions)
        
    if options.emitfile:
        numbers_per_letters = read_dict(open(options.emitfile))
        if set(numbers_per_letters.keys()) != set(alphabet.keys()):
            raise Exception("File in wrong format describing emitter states")

        automaton = Automaton.create_uniform_automaton(
            numbers_per_letters, initial_transitions=initial_transitions)
    elif options.init_from_corpus:
        automaton = Automaton.create_from_corpus(corpus)
        #logg( "Analytical optimum of KL: %f" % distance(corpus,automaton,kullback) )
        automaton.smooth()
        #logg( "KL after smoothing: %f" % distance(corpus,automaton,kullback) )
    else:
        alphabet_numstate = dict([(letter, options.numstate) for letter in alphabet.keys()])
        if options.num_epsilons > 0:
            # adding states for epsilon emission
            alphabet_numstate["EPSILON"] = options.num_epsilons
        automaton = Automaton.create_uniform_automaton(
            alphabet_numstate, initial_transitions=initial_transitions)

    #automaton.dump()
    learner = Learner.create_from_options(automaton, corpus, options)
    learner.learn()

def optparser():
    parser = OptionParser()
    parser.add_option("-f", "--factor", dest="factor", help="change factor", 
                      metavar="FACTOR", default=0.97, type="float")
    parser.add_option("-t", "--tempq",dest="tempq", default=0.9, type="float",
                      help="temperature quotient", metavar="TEMPQ")
    parser.add_option("-n", "--num_of_states",dest="numstate", type="int", default=1,
                      help="number of states per letter of alphabet", metavar="N")
    parser.add_option("-i", "--iter",dest="iter", type="int",
                      help="number of iterations per temperature", metavar="I")
    parser.add_option("-d", "--distance",dest="distfp", help="distance method",
                      metavar="I", type="choice",
                      choices=["kullback", "l1err", "squarerr"])
    parser.add_option("-e", "--emitfile",dest="emitfile", type="str",
                      help="filename of file having (letter,number) pairs",
                      metavar="FILENAME")
    parser.add_option("-s", "--separator",dest="separator", type="str",
                      help="separator of letters in string (allows using complex letters, ie. labels)",
                      metavar="SEPARATOR", default="")
    parser.add_option("-c", "--from-corpus", dest="init_from_corpus", action="store_true", default=False,
                      help="initialize the automaton from corpus frequencies with smoothing")
    parser.add_option("-D", "--downhill-factor", dest="downhill_factor",
                      metavar="PROBABILITY", default=None, type="float",
                      help="in random parameter selection prefer the one which improved the result in the previous iteration with PROBABILITY")
    parser.add_option("-E", "--num-of-epsilon-states", dest="num_epsilons", type="int",
                      metavar="N", default=0,
                      help="number of (non-initial and non-final) states, that doesn't emit anything") 
    parser.add_option("-I", "--initial-transitions", dest="initial_transitions",
                      metavar="FILE", type="str",
                      help="File with initial transition probabilities. Each transition should be in a separate line, \
    source state, target state and  probability are separated by space. Transitions that are not given share the remaining \
    probability mass equally.")


    (options, args) = parser.parse_args()

    return options


if __name__ == "__main__":
    options = optparser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    main(options)
