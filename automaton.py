# TODO use log prob 
# 

import sys
import copy

import math
from collections import defaultdict

from logger import log
from corpus import Corpus
from learner import SimulatedAnnealing

class Automaton:

    def __init__(self) :
        self.m = defaultdict(lambda: defaultdict(float))
        self.emissions = {}
        # A kibocsatas determinisztikus.
        #
        # By convention, for backward compatibility:
        # ha nincs megmondva az allapothoz a kibocsatas,
        # akkor a sajat nevet bocsatja ki.
        # Kiveve "^" es "$", amik semmit sem bocsatanak ki.
        # Igen, goreny convention, nem vegleges.
        #
        # Az allapotokrol es a kibocsatasokrol is
        # csak annyit tetelezunk fel, hogy hash-elhetok,
        # de a gyakorlatban mindketto string.

    @staticmethod
    def read_emission_states(stream):
        d = {}
        for l in stream:
            le = l.strip().split()
            d[le[0]] = int(le[1])
        return d

    @staticmethod
    def read_transitions(filename):
        tr = {}
        f = open(filename)
        for l in f:
            (state1, state2, probstr) = l.strip().split()
            if state1 not in tr:
                tr[state1] = {}
            prob = float(probstr)
            assert prob >= 0.0 and prob <= 1.0
            tr[state1][state2] = float(prob)
        f.close()
        return tr

    # This only functions correctly if _update_emittors()
    # was called after creating or updating self.emissions
    def emittors(self, letter):
        return self.m_emittors[letter]

    @staticmethod
    def nonemitting(state) :
        return state=="^" or state=="$" or is_epsilon_state(state)

    def _update_emittors(self):
        self.m_emittors = {} # Azert nem defaultdict(set), mert igy biztonsagosabb.
	self.m_emittors["EPSILON"] = set() # avoid key error if there are no epsilon states
        for state, letter in self.emissions.iteritems() :
            if letter not in self.m_emittors :
                self.m_emittors[letter] = set()
            self.m_emittors[letter].add(state)

    def copy(self) :
        a2 = Automaton()
        a2.m = self.m.copy()
        a2.emissions = self.emissions.copy()
        a2.m_emittors = self.m_emittors.copy()
        return a2

    @staticmethod
    def create_from_corpus(corpus) :
        automaton = Automaton()
        alphabet = set()
        for item, cnt in corpus.iteritems() :
            if type(item) == str:
                item = '^' + item + '$'
            else:
                # items are tuples
                item = ('^',) + item + ('$',)
            for i in range(len(item)-1) :
                alphabet.add(item[i])
                alphabet.add(item[i+1])
                automaton.m[item[i]][item[i+1]] += cnt
        for n1, value in automaton.m.iteritems() :
            total = sum(value.values())
            for n2, v in value.iteritems() :
                automaton.m[n1][n2] /= total
        for l in alphabet :
            automaton.emissions[l] = l
        automaton._update_emittors()
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
        return not (nonemitting(state1) and nonemitting(state2)) and \
           not state2 == '^' and not state1 == "$" 

    def probability_of_state(self, string, state):
        """The probability of the event that the automaton emits
        'string' and finishes the quest at 'state'.
        'state' did not emit yet:
        It will emit the next symbol following 'string'.

        It expects that the _update_emittors() method is called
        sometime before, otherwise the output of self.emittors()
        is not up to date.

        """

        if len(string)==0 :
            return self.m["^"][state]
        head = string[:-1]
        tail = string[-1]
        total = 0.0
        for previousState in self.emittors(tail) :
            soFar = self.probability_of_state(self, head, previousState)
            soFar *= self.m[previousState][state]
            total += soFar
        return total

    def probability_of_string(self, string) :
        """Probability that the automaton emits this string with the accepting state"""
        return self.probability_of_state(automaton, string, "$")

    @staticmethod
    def closure_and_top_sort(strings) :
        # Closure: includes all prefixes of the strings.
        # Output topologically sorted according to the
        # partial ordering of "being a prefix". AKA sorted.
        closed = set()
        for string in strings :
            for i in range(len(string)+1) :
                closed.add(string[:i])
        return sorted(list(closed))
 
    def probability_of_strings(self, strings) :
        """Expects a list of strings.
         Outputs a map from those strings to probabilities.
         This can then be aggregated somewhere else.
         The trick is that it uses memoization
         to reuse results for shorter strings. 

         FIXME: probabilityOfState common code
         """ 

        topsorted = closure_and_top_sort(strings)
        states = set(self.m.keys())
        states.add("$")
        states.remove("^")

        # memo[string][state] = probabilityOfState(self,string,state)
        memo = defaultdict(lambda: defaultdict(float))
        output = {}

        for string in topsorted :

            # first compute the epsilon states probs because of the
            # memoization dependency
            for state in [ s for s in states if is_epsilon_state(s) ] + \
                         [ s for s in states if not is_epsilon_state(s) ]:
                if len(string)==0 :
                    memo[string][state] = self.m["^"][state]
                else :
                    total = 0.0

                    head = string[:-1]
                    tail = string[-1]
                    for previousState in self.emittors(tail) :
                        soFar = memo[head][previousState]
                        soFar *= self.m[previousState][state]
                        total += soFar

                    # check the case of epsilon emission
                    for epsilonState in self.emittors("EPSILON"):
                        if not is_valid_transition(epsilonState, state):
                            continue
                        # we already have this value because epsilon states
                        # came first
                        soFar = memo[string][epsilonState]
                        soFar *= self.m[epsilonState][state]
                        total += soFar

                    memo[string][state] = total

            output[string] = memo[string]["$"]
        return output

    @staticmethod
    def kullback(p1, p2):
        return p1 * math.log(p1/p2)

    @staticmethod
    def squarerr(p1, p2):
        return (p1 - p2) ** 2

    @staticmethod
    def l1err(p1, p2):
        return abs(p1 - p2)

    def distance(self, corpus, distfp):
        distance = 0.0
        probs = self.probability_of_strings(list(corpus.keys()))
        for item, prob in corpus.iteritems() :
            if prob>0 :
                modeledProb = probs[item]
                if modeledProb==0.0 :
                    modeledProb = 1e-10
                    raise Exception("nem kene ezt kezelni?")
                distance += distfp(prob, modeledProb)
        return distance

    @staticmethod
    def create_uniform_automaton(alphabet, initial_transitions=None) :
        """Creates an automaton with alphabet and uniform transition probabilities.
        If initial_transitions is given (dict of (state, dict of (state, probability),
        returned by read_transitions) the remaining probability mass will be 
        divided up among the uninitialized transitions in an uniform manner.

        """
        automaton = Automaton()

        # create states and emissions
        is_degenerate = True
        for letter in alphabet:
            for index in xrange(alphabet[letter]):
                state = letter + "_" + str(index)
                automaton.emissions[state] = letter
                if is_degenerate and not is_epsilon_state(state):
                    # found at least one emitting state
                    is_degenerate = False

        assert not is_degenerate
                    
        states = automaton.emissions.keys()
        states.append('^')
        states.append('$')

        if initial_transitions:
            for s in initial_transitions.keys():
                if s not in states:
                    log("invalid state name in init: %s" % s)
                    assert False

        # calculate uniform transition distributions
        for s1 in states:
            if s1 == '$':
                continue
            num_valid_transitions = \
                len([ s2 for s2 in states if is_valid_transition(s1, s2) ])
            assert num_valid_transitions > 0

            init_total = 0.0
            n_states_initialized = 0
            if initial_transitions and s1 in initial_transitions:
                for s2 in initial_transitions[s1]:
                    if s2 not in states:
                        log("invalid state name in init: %s" % s2)
                        # FIXME
                        assert False
                    if not is_valid_transition(s1, s2):
                        log("%s %s in init is not a valid transition" % (s1, s2))
                        # FIXME
                        assert False
                    prob = initial_transitions[s1][s2]
                    automaton.m[s1][s2] = prob
                    init_total += prob
                    n_states_initialized += 1
                    assert init_total <= 1.0
            # divide up remaining mass into equal parts
            u = (1.0 - init_total) / (num_valid_transitions - n_states_initialized)
            for s2 in states:
                if is_valid_transition(s1, s2) and (s1 not in automaton.m or s2 not in automaton.m[s1]): 
                    automaton.m[s1][s2] = u
                
        automaton._update_emittors()
        return automaton
                

    @staticmethod
    def normalize_node(edges) :
        total = sum(edges.values())
        for n2 in edges.keys() :
            edges[n2] /= total

    def smooth(self):
        """Smooth zero transition probabilities"""
        nodes = automaton.m.keys()
        for node, edges in automaton.m.iteritems():
            total = sum(edges.values())
            added = 0
            eps = 1E-4
            for other_node in nodes:
                old_val = edges.get(other_node, 0.0)
                if old_val < eps:
                    edges[other_node] = eps
                    added += eps - old_val
            assert added < 1
            for n in edges:
                if edges[n] > eps:
                    edges[n] /= total / (1 - added)
            assert abs(sum(edges.values()) - 1.0) < eps
            
	        
    def boost_edge(self, n1, n2, factor):
        """Multiplies the transition probability between n1 and n2 by factor"""
        automaton.m[n1][n2] *= factor
        normalize_node(automaton.m[n1])

    def dump(self) :
        nodes = sorted(automaton.m.keys())
        for n1 in nodes :
            for n2 in nodes + ['$'] :
                if n2 in automaton.m[n1] :
                    print n1,n2,automaton.m[n1][n2]
        for n1,em in automaton.emissions.iteritems() :
            print ">",n1,em



def main(options) :
    startTemperature = 1e-5
    endTemperature   = 1e-7
    temperatureQuotient = options.tempq
    turnsForEach = options.iter

    corpus = Corpus.read(sys.stdin, options.separator)
    corpus.normalizeCorpus()

    alphabet = corpus.get_alphabet()

    distfp = options.distfp

    initial_transitions = None
    if options.initial_transitions:
	initial_transitions = read_transitions(options.initial_transitions)
        
    bigram_automaton = Automaton.create_from_corpus(corpus)
    log( "Analytical optimum of the chosen error func: %f" % bigram_automaton.distance(corpus, distfp) )
    if options.emitfile:
        # FIXME: close opened file
        numbers_per_letters = read_emission_states(open(options.emitfile))
        if numbers_per_letters.keys() != alphabet.keys():
            raise Exception("File in wrong format describing emitter states")
        automaton = Automaton.create_uniform_automaton(numbers_per_letters, initial_transitions=initial_transitions)
    elif getattr(options, "init_from_corpus", False):
        automaton = Automaton.create_from_corpus(corpus)
	log( "Analytical optimum of KL: %f" % distance(corpus,automaton,kullback) )
	#dumpAutomaton(automaton)
	automaton.smooth()
	log( "KL after smoothing: %f" % distance(corpus,automaton,kullback) )
	#dumpAutomaton(automaton)
    else:
        alphabet_numstate = dict([(letter, options.numstate) for letter in alphabet.keys()])
	if options.num_epsilons > 0:
	    # adding states for epsilon emission
	    alphabet_numstate["EPSILON"] = options.num_epsilons
        automaton = Automaton.create_uniform_automaton(alphabet_numstate, initial_transitions=initial_transitions)

    log( "memoized KL:\t%f" % automaton.distance(corpus,automaton, kullback) )
    log( "sqerr:\t%f" % distance(corpus,automaton, squarerr) )
    log( "l1err:\t%f" % distance(corpus,automaton, l1err) )


    learner = SimulatedAnnealing(corpus, automaton, distfp)
    optimized_automaton = learner.run()


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
