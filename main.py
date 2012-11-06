"""The entry point of the program."""

from optparse import OptionParser

from learner import learn

def main(options) :
    learn(options)

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
    parser.add_option("-c", "--from-corpus", dest="init_from_corpus", action="store_true",
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
    main(options)
