import logging
import math
import random
import sys
from optparse import OptionParser

from automaton import Automaton
from code import AbstractCode
from corpus import read_corpus, normalize_corpus
from mdl import mdl

class Learner(object):
    def __init__(self, automaton, corpus, pref_prob, distfp, turns_for_each, factor, start_temp, end_temp, tempq):
        """
        Learner class implements a simulated annealing method and
        is capable of applying downhill simplex as a preference instead
        of random changing if needed.

        @param corpus: needs to be normalized with corpus.normalize_corpus()
        @param automaton: initialized Automaton instance
        @param pref_prob: probability of using downhill preference
        @param distfp: distance function pointer (kullback, l1err, sqerr)
        @param turns_for_each: iterations at one temperature
        @param factor: edge change factor
        @param start_temp: initial temperature
        @param end_temp: final temperature
        @param tempq: temperature reduction quotient
        """
        self.start_temperature = start_temp
        self.end_temperature   = end_temp
        self.temp_quotient = tempq
        self.turns_for_each = turns_for_each
        self.preference_probability = pref_prob

        self.automaton = automaton
        self.corpus = corpus

        self.distfp = getattr(Automaton, distfp)
        self.factor = factor
        #self.preferred_node_pair = None
        #self.preferred_direction = None
        #self.disallowed_node_pair = None

    @staticmethod
    def create_from_options(automaton, corpus, options):
        return Learner(automaton, corpus, pref_prob=options.downhill_factor,
                       distfp=options.distfp, turns_for_each=options.iter,
                       factor=options.factor, start_temp=options.start_temp,
                       end_temp=options.end_temp, tempq=options.tempq)

    def change_automaton(self, options=None, revert=False):
        if not revert:
            self.automaton.boost_edge(options["edge"], options["factor"])
        else:
            self.automaton.boost_edge(self.previous_change_options["edge"],
                                      1.0 / self.previous_change_options["factor"])

    def randomize_automaton_change(self):
        change_options = {}
        s1 = None
        s2 = None
        states = self.automaton.m.keys()
        while True:
            s1 = random.choice(states)
            s2 = random.choice(self.automaton.m[s1].keys())
            if not hasattr(self, "previous_change_options"):
                break
            else:
                if not (self.previous_change_options["result"] == False and
                (s1, s2) == self.previous_change_options["edge"]):
                    break
        change_options["edge"] = (s1, s2)
        change_options["factor"] = (self.factor if random.random() > 0.5
                                    else 2.0 - self.factor)
        return change_options

    def choose_change_options(self, change_options_random):
        if random.random() < self.preference_probability:
            # downhill
            change_options = self.previous_change_options
        else:
            change_options = change_options_random()
        return change_options

    def simulated_annealing(self, compute_energy, change_something,
                            change_back, option_randomizer):
        temperature = self.start_temperature
        end = self.end_temperature
        tempq = self.temp_quotient
        turns_for_each_temp = self.turns_for_each

        #self.automaton.dump(sys.stdout)
        energy = compute_energy()
        #self.automaton.dump(sys.stdout)
        #quit()
        turn_count = 0
        while True:
            if turn_count == 0:
                logging.info("Running an iteration of Simulated Annealing with " +
                             "{0} temperature and at {1} energy level.".format(
                             temperature, energy))

            change_options = self.choose_change_options(option_randomizer)
            change_something(change_options)
            new_energy = compute_energy()
            energy_change = new_energy - energy
            self.previous_change_options = change_options
            if energy_change < 0:
                accept = True
            else:
                still_accepting_probability = random.random()
                accept = (still_accepting_probability < math.exp(-energy_change/temperature))

            if accept:
                energy = new_energy
                self.previous_change_options["result"] = True
            else:
                self.previous_change_options["result"] = False
                change_back()

            turn_count += 1
            if turn_count == turns_for_each_temp:
                turn_count = 0
                temperature *= tempq
                if temperature < end:
                    break

    def main(self):
        compute_energy = lambda: self.automaton.distance_from_corpus(
                self.corpus, self.distfp)
        change_something = lambda x: self.change_automaton(x, False)
        change_back = lambda: self.change_automaton(None, True)
        option_randomizer = lambda: self.randomize_automaton_change()
        self.simulated_annealing(compute_energy, change_something,
                                 change_back, option_randomizer)
        mdl_ = mdl(self.automaton, self.corpus, self.automaton.code.bits,
                   self.distfp, n_state=0, n_alphabet=0, n_word=0)
        logging.info("Learning is finished. MDL is {0}".format(mdl_))

def optparser():
    parser = OptionParser()
    parser.add_option("-f", "--factor", dest="factor", help="change factor " +
                      "[default=%default]", default=0.97, type="float",
                     metavar="FACTOR")
    parser.add_option("-t", "--tempq", dest="tempq", default=0.9, type="float",
                      help="temperature quotient [default=%default]",
                      metavar="TEMPQ")
    parser.add_option("", "--start_temp",dest="start_temp", default=1e-5,
                      type="float", help="start temperature " + 
                      "[default=%default]", metavar="TEMP")
    parser.add_option("", "--end_temp", dest="end_temp", default=1e-7,
                      type="float", help="end temperature " + 
                      "[default=%default]", metavar="TEMP")
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
    parser.add_option("-q", "--code", dest="code", type="str", default=None,
                      metavar="FILE",
                      help="store parameters using a code specified in FILE")
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

    if options.code:
        automaton.code = AbstractCode.read(open(options.code))
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
