import sys
import math
from collections import defaultdict
import logging
from optparse import OptionParser

from misc import closure_and_top_sort
from corpus import read_dict, read_corpus, normalize_corpus, get_alphabet

class Automaton(object):
    """ Classic Moore-automaton class with
    @m: transitions per states
    @emissions: emission per states
    @m_emittors: states per emitted letter"""
    eps = 1e-7
    m_inf = float("-inf")


    def __init__(self) :
        # the transitions
        self.m = defaultdict(lambda: defaultdict(lambda: Automaton.m_inf))

        #self.m = defaultdict(dict)
        # emissions for the states
        self.emissions = {}
        self.m_emittors = defaultdict(set)

        # how edge values can be coded
        self.code = None

    @staticmethod
    def read_transitions(filename):
        tr = {}
        f = open(filename)
        for l in f:
            (state1, state2, probstr) = l.strip().split()
            if state1 not in tr:
                tr[state1] = {}
            prob = float(probstr)
            if not (prob < 0.0):
                raise ValueError("invalid probabilities in {0}, ".format(
                    filename) + "only logprobs are accepted." )

            tr[state1][state2] = prob
        f.close()
        return tr
    
    @staticmethod
    def create_from_dump(file_name):
        """ Reads automaton dump from @file_name"""
        automaton = Automaton()
        # create states and emissions
        for line in open(file_name):
            l = line.strip().split()
            if len(l) != 4: continue
            s1, _, s2, weight = l
            s2 = s2.strip(':')
            weight = float(weight)
            automaton.emissions[s1] = s1[:-2]
            automaton.m_emittors[s1[:-2]].add(s1)
            automaton.emissions[s2] = s2[:-2]
            automaton.m_emittors[s2[:-2]].add(s2)
            if weight >= 0.0:
                raise ValueError("Only logprogs are accepted in dumps")
            automaton.m[s1][s2] = weight

        for node in automaton.m.iterkeys():
            print node
            automaton.check_node_sum(node)

        automaton.finalize()
        return automaton
    
    @staticmethod
    def _create_automaton_from_alphabet(alphabet):
        """ Creates states of the automaton given by @alphabet
        @alphabet is a dict from letters to the number of states that emits
        that letter
        """
        automaton = Automaton()

        # create states and emissions
        is_degenerate = True
        for letter in alphabet:
            for index in xrange(alphabet[letter]):
                state = "".join(letter) + "_" + str(index)
                automaton.emissions[state] = letter
                automaton.m_emittors[letter].add(state)
                if is_degenerate and not Automaton.is_epsilon_state(state):
                    # found at least one emitting state
                    is_degenerate = False

        if is_degenerate:
            raise Exception("Automaton has no emittors")
        
        return automaton

    @staticmethod
    def create_uniform_automaton(alphabet, initial_transitions=None) :
        """Creates an automaton with alphabet and uniform transition probabilities.
        If initial_transitions is given (dict of (state, dict of (state, probability),
        returned by read_transitions) the remaining probability mass will be 
        divided up among the uninitialized transitions in an uniform manner.
        """

        automaton = Automaton._create_automaton_from_alphabet(alphabet)
                    
        states = automaton.emissions.keys()
        states.append('^')
        states.append('$')

        if initial_transitions:
            for s in initial_transitions.keys():
                if s not in states:
                    raise Exception("invalid state name in initial " +
                                   "transitions given by option -I")
                for s2 in initial_transitions[s]:
                    if s2 not in states:
                        raise Exception("invalid state name in initial " +
                                       "transitions given by option -I")

        # calculate uniform transition distributions
        for s1 in states:
            if s1 == '$':
                continue

            init_total = 0.0
            states_initialized = set()
            if initial_transitions and s1 in initial_transitions:
                for s2 in initial_transitions[s1]:
                    if s2 not in states:
                        raise Exception("invalid state name in init: %s" % s2)

                    prob = initial_transitions[s1][s2]
                    automaton.m[s1][s2] = prob
                    init_total += math.exp(prob)
                    states_initialized.add(s2)
                    # TODO refactor this
                    if init_total > 1.0000001:
                        sys.stderr.write("state: {0}, init total: {1}\n".format(s1, init_total))
                        raise Exception("Too much probability for init_total")


            # divide up remaining mass into equal parts
            valid_next_states = set([s2 for s2 in states
                                 if Automaton.is_valid_transition(s1, s2)])
            
            if valid_next_states == states_initialized:
                continue

            u = (1.0 - init_total) / (len(valid_next_states) - len(states_initialized))
            for s2 in valid_next_states - states_initialized:
                try:
                    automaton.m[s1][s2] = math.log(u)
                except ValueError:
                    automaton.m[s1][s2] = Automaton.m_inf
                
        automaton.finalize()
        return automaton
                
    @staticmethod
    def create_from_corpus(corpus):
        """ Creates an automaton from a corpus, where @corpus is a dict from
        items (str or tuple) to counts
        """
        automaton = Automaton()
        alphabet = set()
        total = float(sum(corpus.itervalues()))
        for item, cnt in corpus.iteritems() :
            if type(item) == str:
                item = '^' + item + '$'
            else:
                # items are tuples e.g. letter are multichar strs
                item = ('^',) + item + ('$',)
            for i in range(len(item) - 1) :
                alphabet.add(item[i])
                alphabet.add(item[i+1])
                if item[i+1] in automaton.m[item[i]]:
                    automaton.m[item[i]][item[i+1]] += cnt / total
                else:
                    automaton.m[item[i]][item[i+1]] = cnt / total

        # changing to log probs and normalize
        for node1, outs in automaton.m.iteritems():
            for node2 in outs.iterkeys():
                outs[node2] = math.log(outs[node2])
            automaton.normalize_node(node1)

        for l in alphabet:
            automaton.emissions[l] = l
            automaton.m_emittors[l].add(l)

        automaton.finalize()
        return automaton

    @staticmethod
    def is_epsilon_state(state):
        return state.startswith("EPSILON_")

    @staticmethod
    def is_valid_transition(state1, state2):
        # subsequent non emitting states are not allowed
        # the only exception is '^' -> '$'
        if (state1, state2) == ('^', '$'):
            return True
        return (
            not (Automaton.nonemitting(state1)
                 and Automaton.nonemitting(state2))
            and not state2 == '^'
            and not state1 == "$")

    @staticmethod
    def nonemitting(state) :
        return state=="^" or state=="$" or Automaton.is_epsilon_state(state)

    def finalize(self):
        for state1, transitions in self.m.iteritems():
            self.m[state1] = dict(transitions)
        self.m = dict(self.m)
    
    def emittors(self, letter):
        return self.m_emittors[letter]

    def update_probability_of_string_in_state(self, string, state, memo):
        """The probability of the event that the automaton emits
        'string' and finishes the quest at 'state'.
        'state' did not emit yet:
        It will emit the next symbol following 'string'.
        """
        
        if len(string)==0 :
            if not Automaton.is_epsilon_state(state) and self.m["^"].has_key(state):
                memo[string][state] = self.m["^"][state]
            return

        head = string[:-1]
        tail = string[-1]
        total = Automaton.m_inf
        # compute real emissions
        for previousState in self.emittors(tail):
            soFar = memo[head][previousState]
            try:
                soFar += self.m[previousState][state]
            except KeyError:
                soFar = Automaton.m_inf
            total = max(soFar, total)
            #total = math.log(math.exp(soFar) + math.exp(total))

        # check the case of epsilon emission
        if not Automaton.nonemitting(state):
            for epsilonState in self.emittors("EPSILON"):
                # we already have this value because epsilon states
                # came first
                soFar = memo[string][epsilonState]
                soFar += self.m[epsilonState][state]
                total = max(soFar, total)
                #total = math.log(math.exp(soFar) + math.exp(total))
        memo[string][state] = total

    def update_probability_of_string(self, string, memo) :
        """Probability that the automaton emits this string"""
        states = set(self.m.keys())
        states.add("$")
        states.remove("^")
        logging.debug(string)
        logging.debug(memo)

        # first compute the epsilon states probs because of the
        # memoization dependency
        for state in sorted(states,
                     key=lambda x: not Automaton.is_epsilon_state(x)):
            logging.debug(state)
            self.update_probability_of_string_in_state(string, state, memo)
            logging.debug(memo)

    def probability_of_strings(self, strings) :
        """
        Expects a list of strings.
        Outputs a map from those strings to probabilities.
        This can then be aggregated somewhere else.
        The trick is that it uses memoization
        to reuse results for shorter strings. 
        """ 
        topsorted = closure_and_top_sort(strings)

        # memo[string][state] = probabilityOfState(self,string,state)
        memo = defaultdict(lambda: defaultdict(float))
        output = {}

        for string in topsorted :
            self.update_probability_of_string(string, memo)

            output[string] = memo[string]["$"]
        return output

    @staticmethod
    def kullback(p1, p2):
        if p1 == 0:
            return 0
        return p1 * math.log(p1/p2)

    @staticmethod
    def squarerr(p1, p2):
        return (p1 - p2) ** 2

    @staticmethod
    def l1err(p1, p2):
        return abs(p1 - p2)

    def distance_from_corpus(self, corpus, distfp):
        # TODO rethink prob vs logprob problem
        distance = 0.0
        probs = self.probability_of_strings(list(corpus.keys()))
        for item, prob in corpus.iteritems():
            if prob>0 :
                modeledProb = math.exp(probs[item])
                if modeledProb==0.0 :
                    modeledProb = 1e-50
                    #raise Exception("nem kene ezt kezelni?")
                    #egyelore nem
                distance += distfp(prob, modeledProb)
        return distance

    def round_and_normalize_node(self, node):
        if self.code:
            self.round_transitions(self.m[node]) 
        self.normalize_node(node)
    
    def round_transitions(self, edges):
        for state, weight in edges.iteritems():
            edges[state] = self.code.representer(weight)

    def normalize_node(self, node):
        edges = self.m[node]
        total_log = math.log(sum(math.exp(v) for v in edges.values()))
        for n2 in edges.keys():
            edges[n2] -= total_log
    
    def round_and_normalize(self):
        for state in self.m.iterkeys():
            self.round_and_normalize_node(state)

    def smooth(self):
        """Smooth zero transition probabilities"""
        for state, edges in self.m.iteritems():
            total_log = math.log(sum(math.exp(v) for v in edges.values()))
            added = 0.0
            eps = Automaton.eps
            for other_state in edges:
                old_val = math.exp(edges.get(other_state, Automaton.m_inf))
                if old_val < eps:
                    edges[other_state] = math.log(eps)
                    added += eps - old_val
                if added >= 1.0:
                    raise Exception("Too much probability " +
                                    "added while smoothing")

            # normalize the nodes that haven't been smoothed
            for n in edges:
                if edges[n] > math.log(eps):
                    edges[n] -= total_log - math.log(1 - added)

            #if abs(sum(edges.values()) - 1.0) > eps:
            #    raise Exception("Edges sum up to too much")
	        
    def boost_edge(self, edge, factor):
        """Multiplies the transition probability between n1 and n2 by factor"""
        n1, n2 = edge
        self.m[n1][n2] += math.log(factor)
        self.round_and_normalize_node(n1)
        self.check_node_sum(n1)

    def check_node_sum(self, node):
        edges = self.m[node]
        eps = Automaton.eps
        n_sum = sum([math.exp(log_prob) for log_prob in edges.values()])
        if n_sum < 1 + eps and n_sum > 1 - eps:
            return
        else:
            raise Exception("transitions from state don't sum to 1, " +
                          "but {0}".format(n_sum))

    def dump(self, f):
        nodes = sorted(self.m.keys())
        for n1 in nodes:
            for n2 in nodes + ['$']:
                if n2 in self.m[n1]:
                    f.write("{0} -> {1}: {2}\n".format(
                        n1, n2, self.m[n1][n2]))
        for n1, em in self.emissions.iteritems():
            f.write("{0}: \"{1}\"\n".format(n1, em))
    
def optparser():
    parser = OptionParser()
    parser.add_option("-e", "--emitfile",dest="emitfile", type="str",
                      help="file having (letter,number) pairs, from which " +
                      "a uniform automaton is created. EPSILON can be a " +
                      "letter. Option -I can be used to override some " +
                      "transitions. See tst/emitfile.sample",
                      metavar="FILENAME")

    parser.add_option("-I", "--initial-transitions", dest="initial_transitions",
                      metavar="FILE", type="str", help="File with initial " + 
                      "transition probabilities. Each transition should be " + 
                      "in a separate line, source state, target state and " + 
                      "probability are separated by space. Transitions that " + 
                      "are not given share the remaining probability mass " + 
                      "equally. See tst/init_trans.sample")

    parser.add_option("-c", "--from-corpus", dest="init_from_corpus",
                      default=None, type="str", help="initialize the " +
                      "automaton from corpus frequencies with smoothing. " +
                      "Can be used together with option -s. " +
                      "See tst/corpus.sample", metavar="FILENAME")

    parser.add_option("-s", "--separator",dest="separator", type="str",
                      help="separator of letters in corpus (allows using " + 
                      " complex letters, ie. labels)", metavar="SEPARATOR",
                      default="")

    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      type="str", default=None, help="File containing the " +
                      "dump of the automaton to initialize [default=stdout]")

    parser.add_option("-E", "--num-of-epsilon-states", dest="num_epsilons",
                      type="int", metavar="N", default=0, help="number of " +
                      "(non-initial and non-final) states, that doesn't " + 
                      "emit anything [default=%default]") 

    parser.add_option("-n", "--num_of_states",dest="numstate", type="int",
                      default=1, metavar="N",
                      help="number of states per letter of alphabet " + 
                     "[default=%default]")

    (options, args) = parser.parse_args()
    return options

def create_wfsa(options):
    # open output file or write to stdout
    output = (open(options.output, "w") if options.output else sys.stdout)

    # read initial transitions if given
    it = options.initial_transitions
    initial_transitions = (Automaton.read_transitions(it) if it else {})

    # create uniform automaton with given number of states per letter
    # and the possibility of predefine some transitions
    if options.emitfile:
        numbers_per_letters = read_dict(open(options.emitfile))
        automaton = Automaton.create_uniform_automaton(
            numbers_per_letters, initial_transitions=initial_transitions)
        automaton.dump(output)
        return

    if options.numstate:
        input_ = open(options.init_from_corpus)
        corpus = read_corpus(input_, options.separator)
        alphabet = get_alphabet(corpus)
        numbers_per_letters = dict([(letter, options.numstate)
                              for letter in alphabet])
        if options.num_epsilons:
            numbers_per_letters["EPSILON"] = options.num_epsilons

        automaton = Automaton.create_uniform_automaton(
            numbers_per_letters, initial_transitions)
        automaton.dump(output)
        return

    if options.init_from_corpus:
        if len(initial_transitions) > 0:
            raise Exception("Using initial transitions (-I option) when " +
                           "creating automaton from corpus is not implemented")
        input_ = open(options.init_from_corpus)
        corpus = read_corpus(input_, options.separator)
        corpus = normalize_corpus(corpus)
        automaton = Automaton.create_from_corpus(corpus)
        automaton.smooth()
        automaton.dump(output)
        return


    # fallback
    logging.error("Options are not complete, something is missing to create " +
                  "an Automaton")
    sys.exit(-1)

def main(options):
    create_wfsa(options)

if __name__ == "__main__":
    options = optparser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    main(options)
