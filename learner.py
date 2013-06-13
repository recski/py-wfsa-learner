import logging
import math
import random
import sys
from optparse import OptionParser

from automaton import Automaton
from quantizer import AbstractQuantizer
from corpus import read_corpus, normalize_corpus

class Learner(object):
    def __init__(self, automaton, corpus, checkpoint, pref_prob, distfp, 
                 turns_for_each, factors, temperatures):
        """
        Learner class implements a simulated annealing method and
        is capable of applying downhill simplex as a preference instead
        of random changing if needed.

        @param corpus: needs to be normalized with corpus.normalize_corpus()
        @param automaton: initialized Automaton instance
        @param pref_prob: probability of using downhill preference
        @param distfp: distance function pointer (kullback, l1err, sqerr)
        @param turns_for_each: iterations at one temperature
        @param factors: list of factors, length of list is how many bursts/
                        epochs we want to run, and every element is a float
                        which is the factor of the actual epoch
        @param temperatures: same length list as factors and the elements are
                             the tolerance parameters for the simmulated
                             annealing
        """
        self.turns_for_each = turns_for_each
        self.preference_probability = pref_prob

        self.automaton = automaton
        self.corpus = corpus

        self.distfp = getattr(Automaton, distfp)

        if len(temperatures) != len(factors):
            print temperatures
            print factors
            raise Exception("temperatures has to have the same length as " +
                           "factors when creating a Learner")
        self.factors = factors
        self.temps = temperatures

        self.checkpoint = checkpoint
        #self.preferred_node_pair = None
        #self.preferred_direction = None
        #self.disallowed_node_pair = None

    @staticmethod
    def create_from_options(automaton, corpus, options):
        return Learner(automaton, corpus, checkpoint=None,
                       pref_prob=options.downhill_factor,
                       distfp=options.distfp, turns_for_each=options.iter,
                       factors=options.factors, temperatures=options.temps)

    def change_automaton(self, options=None, revert=False):
        if not revert:
            self.automaton.boost_edge(options["edge"], options["factor"])
        else:
            self.automaton.boost_edge(self.previous_change_options["edge"],
                                      -self.previous_change_options["factor"])

    def randomize_automaton_change(self, factor):
        change_options = {}
        s1 = None
        s2 = None
        states = self.automaton.m.keys()
        while True:
            s1 = random.choice(states)
            if len(self.automaton.m[s1]) < 2:
                continue

            s2 = random.choice(self.automaton.m[s1].keys())
            if not hasattr(self, "previous_change_options"):
                break
            else:
                if not (self.previous_change_options["result"] == False and
                (s1, s2) == self.previous_change_options["edge"]):
                    break
        change_options["edge"] = (s1, s2)
        factor = (factor if random.random() > 0.5 else -factor)
        
        if self.automaton.quantizer is not None:
            factor = self.automaton.quantizer.shift(
                self.automaton.m[s1][s2], factor) - self.automaton.m[s1][s2]

        change_options["factor"] = factor
        return change_options

    def choose_change_options(self, change_options_random, *args):
        if random.random() < self.preference_probability:
            # downhill
            change_options = self.previous_change_options
        else:
            change_options = change_options_random(*args)
        return change_options

    def simulated_annealing(self, compute_energy, change_something,
                            change_back, option_randomizer):

        energy = compute_energy()
        for factor, temperature in zip(self.factors, self.temps):
            for turn_count in xrange(self.turns_for_each):
                if turn_count == 0:
                    logging.info("Running an iteration of Simulated " + 
                                 "Annealing with {0} factor ".format(factor) +
                                 "and at {0} energy level.".format(energy))

                change_options = self.choose_change_options(option_randomizer,
                                                            factor)
                change_something(change_options)
                new_energy = compute_energy()
                energy_change = new_energy - energy
                self.previous_change_options = change_options
                if energy_change < 0:
                    accept = True
                else:
                    still_accepting_probability = random.random()
                    accept = (still_accepting_probability <
                              math.exp(-energy_change/temperature))

                if accept:
                    energy = new_energy
                    self.previous_change_options["result"] = True
                else:
                    self.previous_change_options["result"] = False
                    change_back()

            if self.checkpoint:
                self.checkpoint(factor, temperature)

    def main(self):
        compute_energy = lambda: self.automaton.distance_from_corpus(
                self.corpus, self.distfp)
        change_something = lambda x: self.change_automaton(x, False)
        change_back = lambda: self.change_automaton(None, True)
        option_randomizer = lambda x: self.randomize_automaton_change(x)
        self.simulated_annealing(compute_energy, change_something,
                                 change_back, option_randomizer)


def csl2l_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, [float(_) for _ in value.split(',')])

def optparser():
    parser = OptionParser()
    parser.add_option("-f", "--factors", dest="factors", help="comma " + 
                      "separated list of change factors " + 
                      "[default=%default]", default="0.2,0.4", 
                      type="string", action="callback",
                      callback=csl2l_callback)
    parser.add_option("-t", "--temps", dest="temps", default="1e-5,1e-6",
                      type="string", help="temperatures " + 
                      "[default=%default]", action="callback",
                      callback=csl2l_callback)
    parser.add_option("-i", "--iter",dest="iter", type="int", default=500,
                      help="number of iterations per temperature " +
                      "[default=%default]", metavar="I")
    parser.add_option("-d", "--distance", dest="distfp", type="choice",
                      metavar="I", help="distance method",
                      choices=["kullback", "l1err", "squarerr"])
    parser.add_option("-D", "--downhill-factor", dest="downhill_factor",
                      metavar="PROBABILITY", default=0.0, type="float",
                      help="in random parameter selection prefer the one " +
                      "which improved the result in the previous iteration " +
                      "with PROBABILITY [default=%default]")
    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      type="str", default=None, help="File containing the " +
                      "dump of the learnt automaton [default=stdout]")

    parser.add_option("-a", "--automaton-file", dest="automaton_file",
                      metavar="FILE", type="str", default=None,
                      help="File containing the dump of the input automaton")
    parser.add_option("-q", "--quantizer", dest="quantizer", type="str",
                      default=None, metavar="FILE",
                      help="store parameters using a quantizer")
    parser.add_option("-c", "--corpus", dest="corpus",
                      default=None, type="str", help="optimize automaton " +
                      "on given corpus. Can be used together with -s. " + 
                      "[default=stdin]. See tst/corpus.sample",
                      metavar="FILENAME")
    parser.add_option("-s", "--separator",dest="separator", type="str",
                      help="separator of letters in corpus (allows using " + 
                      " complex letters, ie. labels)", metavar="SEPARATOR",
                      default="")

    (options, args) = parser.parse_args()

    return options

def main(options):
    if not options.automaton_file:
        raise Exception("Automaton \"option\" (-a) is mandatory")
    automaton = Automaton.create_from_dump(options.automaton_file)

    if options.quantizer:
        automaton.quantizer = AbstractQuantizer.read(open(options.quantizer))
        automaton.round_and_normalize()

    input_ = sys.stdin
    if options.corpus:
        input_ = open(options.corpus)
    corpus = read_corpus(input_, options.separator)
    corpus = normalize_corpus(corpus)

    learner = Learner.create_from_options(automaton, corpus, options)
    learner.main()

    output = sys.stdout
    if options.output:
        output = open(options.output, "w")
    learner.automaton.dump(output)

if __name__ == "__main__":
    options = optparser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    main(options)
