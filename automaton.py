# TODO use log prob 
# 
import sys
import math
from collections import defaultdict
import logging

from misc import closure_and_top_sort

class Automaton(object):

    def __init__(self) :
        # the transitions
        self.m = defaultdict(lambda: defaultdict(float))

        #self.m = defaultdict(dict)
        # emissions for the states
        self.emissions = {}
        self.m_emittors = defaultdict(set)
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
            if not (prob >= 0.0 and prob <= 1.0):
                raise ValueError("invalid probabilities in {0}".format(
                    filename))

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
            automaton.m[s1][s2] = Automaton.my_log(weight)

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
                state = letter + "_" + str(index)
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
                    raise Exception("invalid state name in init: %s" % s)

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
                    try:                      
                        automaton.m[s1][s2] = math.log(prob)
                    except ValueError:
                        automaton.m[s1][s2] = float("-inf")
                    init_total += prob
                    states_initialized.add(s2)
                    if init_total > 1.0000001:
                        sys.stderr.write("state: {0}, init total: {1}\n".format(s1, init_total))
                        raise Exception("Too much probability for init_total")


            # divide up remaining mass into equal parts
            valid_next_states = set([s2 for s2 in states
                                 if Automaton.is_valid_transition(s1, s2)])
            
            if valid_next_states == states_initialized:
                continue

            #sys.stderr.write('{0} {1}\n'.format(s1, valid_next_states))
            u = (1.0 - init_total) / (len(valid_next_states) - len(states_initialized))
            for s2 in valid_next_states - states_initialized:
                #sys.stderr.write(s2)
                #sys.stderr.write('foo')
                try:
                    automaton.m[s1][s2] = math.log(u)
                except ValueError:
                    automaton.m[s1][s2] = float("-inf")
                
        return automaton
                
    @staticmethod
    def create_from_corpus(corpus) :
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
                # items are tuples
                item = ('^',) + item + ('$',)
            for i in range(len(item)-1) :
                alphabet.add(item[i])
                alphabet.add(item[i+1])
                automaton.m[item[i]][item[i+1]] += cnt / total
        for item1 in automaton.m.iterkeys():
            for item2 in automaton.m[item1].iterkeys():
                automaton.m[item1][item2] = math.log(automaton.m[item1][item2])
        for n1, value in automaton.m.iteritems():
            #TODO ezt ugye ki lehet venni?
            automaton.normalize_node(value)
        for l in alphabet:
            automaton.emissions[l] = l
            automaton.m_emittors[l].add(l)
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

    def change_defaultdict_to_dict(self):
        for state1, transitions in self.m.iteritems():
            self.m[state1] = dict(transitions)
        self.m = dict(self.m)
    
    def emittors(self, letter):
        return self.m_emittors[letter]

#    def copy(self) :
#        a2 = Automaton()
#        a2.m = self.m.copy()
#        a2.emissions = self.emissions.copy()
#        a2.m_emittors = self.m_emittors.copy()
#        return a2

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
        total = float("-inf")
        # compute real emissions
        for previousState in self.emittors(tail):
            soFar = memo[head][previousState]
            try:
                soFar += self.m[previousState][state]
            except KeyError:
                soFar = float("-inf")
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
        distance = 0.0
        probs = self.probability_of_strings(list(corpus.keys()))
        for item, prob in corpus.iteritems():
            if prob>0 :
                modeledProb = math.exp(probs[item])
                if modeledProb==0.0 :
                    #print item, probs[item]
                    modeledProb = 1e-50
                    #raise Exception("nem kene ezt kezelni?")
                    #egyelore nem
                #print prob, modeledProb
                #quit()
                distance += distfp(prob, modeledProb)
                #print distance
        return distance

    def round_and_normalize_node(self, edges):
        if self.code:
            self.round_transitions(edges) 
        self.normalize_node(edges)
    
    def round_transitions(self, edges):
        for state, weight in edges.iteritems():
            edges[state] = self.code.quantize(weight)

    def normalize_node(self, edges):
        #print 'before:', edges.values()
        total_log = math.log(sum(math.exp(v) for v in edges.values()))
        for n2 in edges.keys():
            edges[n2] -= total_log
        #print 'after:', edges.values()
        #print
    
    def round_and_normalize(self):
        for state, edges in self.m.iteritems():
            self.round_and_normalize_node(edges)

    def smooth(self):
        """Smooth zero transition probabilities"""
        for state, edges in self.m.iteritems():
            total_log = math.log(sum(math.exp(v) for v in edges.values()))
            added = 0
            eps = 1E-4
            for other_state in edges:
                old_val = math.exp(edges.get(other_state, float("-inf")))
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
        self.round_and_normalize_node(self.m[n1])
        Automaton.check_node_sum(self.m[n1])

    #obsolete
    @staticmethod
    def check_node_sum(edges):
        epsilon = 1e-5
        n_sum = sum([math.exp(log_prob) for log_prob in edges.values()])
        if n_sum < 1+epsilon and n_sum > 1-epsilon:
            return
        for state, weight in edges.iteritems():
            print state, math.exp(weight)
        logging.info("abort: transitions from state don't sum to 1, but {0}".format(n_sum))
        sys.exit(-1)

    def dump(self, f):
        #raise Exception("Not implemented")
        nodes = sorted(self.m.keys())
        for n1 in nodes:
            for n2 in nodes + ['$']:
                if n2 in self.m[n1]:
                    f.write("{0} -> {1}: {2}\n".format(
                        n1, n2, math.exp(self.m[n1][n2])))
        for n1, em in self.emissions.iteritems():
            f.write("{0}: \"{1}\"\n".format(n1, em))
    
    #obsolete
    @staticmethod
    def dump_round(orig, round):
        for weight, state in orig:
            print state, weight, round[state]

    #obsolete
    @staticmethod
    def my_log(x):
        if x == 0:
            return float('-inf')
        else:
            return math.log(x)

    #obsolete
    @staticmethod
    def bit_round(x, n):
        return round(x*((2**n)-1))/((2**n)-1)

    #obsolete
    @staticmethod
    def round_32(weight, bit_no):
        if weight in (0, 1): return weight
        to_round = math.log(weight)/-32
        rounded = Automaton.bit_round(to_round, bit_no)
        rounded_weight = math.exp(rounded*-32)
        return rounded_weight


class Code():
    
    def __init__(self):
        self.codes_to_rep = {}
        self.intervals_to_rep = {}
        self.intervals = []
        self.bit_no = 0 
    @staticmethod
    def create_from_file(file_name):
        coding = Code()
        for line in file(file_name):
            l = line.strip().split()
            code, interval, rep = l[0], tuple([float(i)
                                               for i in l[1:3]]), float(l[3])
            if interval[1]<interval[0]:
                print interval[1], interval[0]
                logging.critical("malformed interval in coding file: {0}\
                                 ".format(interval))
                sys.exit(-1)
            coding.codes_to_rep[code] = rep
            coding.intervals_to_rep[interval] = rep
            coding.intervals.append(interval)
        coding.bit_no = math.log(len(coding.codes_to_rep) ,2)
        coding.intervals.sort()
        coding.min_value = coding.intervals_to_rep[coding.intervals[0]]
        coding.max_value = coding.intervals_to_rep[coding.intervals[-1]]
        #print coding.intervals_to_rep.values()
        #print [math.exp(l) for l in coding.intervals_to_rep.values()]
        return coding

    def quantize(self, weight):
        try:
            if weight < self.intervals[0][0]: return self.min_value
            elif weight > self.intervals[-1][1]: return self.max_value
            for interval in self.intervals:
                if weight < interval[1]:
                    if not weight >= interval[0]:
                        raise Exception()
                    return self.intervals_to_rep[interval]
            if weight == self.intervals[-1][1]:
                return self.intervals_to_rep[self.intervals[-1]]
            logging.critical("can't round value {0}. Incomplete coding file?\
                             ".format(weight))
            sys.exit(-1)
        except Exception, e:
            logging.critical("can't round value {0}".format(weight))
            raise e

