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
        from encoder import Encoder
        self.encoder = Encoder(3.1196)
        # the transitions
        self.m = defaultdict(lambda: defaultdict(lambda: Automaton.m_inf))

        #self.m = defaultdict(dict)
        # emissions for the states
        self.emissions = {}
        self.m_emittors = defaultdict(set)

        # how edge values can be coded
        self.quantizer = None

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
            if len(l) == 4:
                s1, _, s2, weight = l
                s2 = s2.strip(':')
                weight = float(weight)

                # check this with 1e-10 instead of 0.0 because of floating 
                # point precision error
                if weight > 1e-10:
                    raise ValueError("Only logprogs are accepted in dumps")

                automaton.m[s1][s2] = weight
            elif len(l) == 2:
                state = l[0].rstrip(":")
                emission = eval(l[1])
                automaton.emissions[state] = emission
                automaton.m_emittors[emission].add(state)

        for state in automaton.m.iterkeys():
            automaton.check_state_sum(state)

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
        items (str or tuple) to counts"""
        automaton = Automaton()
        alphabet = set()
        total = float(sum(corpus.itervalues()))
        for item, cnt in corpus.iteritems() :
            item = ('^',) + item + ('$',)
            for i in range(len(item) - 1) :
                alphabet.add(item[i])
                alphabet.add(item[i+1])
                if item[i+1] in automaton.m[item[i]]:
                    automaton.m[item[i]][item[i+1]] += cnt / total
                else:
                    automaton.m[item[i]][item[i+1]] = cnt / total

        # changing to log probs and normalize
        for state1, outs in automaton.m.iteritems():
            for state2 in outs.iterkeys():
                outs[state2] = math.log(outs[state2])
            automaton.normalize_state(state1)

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
        self.m_emittors = dict(self.m_emittors)
    
    def emittors(self, letter):
        return self.m_emittors[letter]

    def update_probability_of_string_in_state(self, string, state, memo):
        """The probability of the event that the automaton emits
        'string' and finishes the quest at 'state'.
        'state' did not emit yet:
        It will emit the next symbol following 'string'.
        """
        total = Automaton.m_inf
        # compute real emissions
        for previousState in self.emissions:
            previousState_i = self.state_indices[previousState]

            # stop if no transitions to state
            if not state in self.m[previousState]:
                continue

            state_emit = self.emissions[previousState]
            state_emit_l = len(state_emit)

            if state_emit == string[-state_emit_l:]:
                head = string[:-state_emit_l]

                soFar = Automaton.m_inf
                if head in memo and memo[head][previousState_i] is not None:
                    soFar = memo[head][previousState_i]
                    soFar += self.m[previousState][state]
                total = max(soFar, total)

        # check the case of epsilon emission
        #if (not Automaton.nonemitting(state)):
        if True:
            for epsilonState in self.m.keys():
                epsilonState_i = self.state_indices[epsilonState]
                if not Automaton.nonemitting(epsilonState):
                    continue

                # we already have this value because epsilon states
                # came first

                # if the automaton is not complete, avoid KeyError:
                if not state in self.m[epsilonState]:
                    continue

                if not string in memo or memo[string][epsilonState_i] is None:
                    continue

                soFar = memo[string][epsilonState_i]
                soFar += self.m[epsilonState][state]
                total = max(soFar, total)

        if string not in memo:
            memo[string] = [None] * len(self.state_indices)

        memo[string][self.state_indices[state]] = total

    def update_probability_of_string(self, string, memo) :
        """Probability that the automaton emits this string"""
        states = set(self.m.keys())
        states.add("$")
        states.remove("^")

        # first compute the epsilon states probs because of the
        # memoization dependency
        for state in sorted(states,
                     key=lambda x: not Automaton.is_epsilon_state(x)):
            self.update_probability_of_string_in_state(string, state, memo)

    def probability_of_strings(self, strings) :
        """
        Expects a list of strings.
        Outputs a map from those strings to probabilities.
        """ 
        topsorted = closure_and_top_sort(strings)
        # remove empty string
        topsorted = topsorted[1:]

        memo = self.init_memo()
        output = {}

        for string in topsorted :
            self.update_probability_of_string(string, memo)
            output[string] = memo[string][self.state_indices["$"]]
        return output

    def init_memo(self):
        
        # to save memory if memo is huge, inner dicts in memo are actually
        # lists with state indices
        states = set(self.m.keys())
        states.add("$")
        self.state_indices = dict([(s, i) for i, s in enumerate(states)])

        memo = {(): [None] * len(states)}
        epsilon_reachables = set(["^"])
        while True:
            targets = set()
            for state in epsilon_reachables:
                state_i = self.state_indices[state]

                for target in self.m[state]:
                    target_i = self.state_indices[target]

                    if target in epsilon_reachables:
                        continue

                    if Automaton.is_epsilon_state(target):
                        targets.add(target)
                        # start is not memoized

                    so_far = Automaton.m_inf
                    if memo[()][target_i] is not None:
                        so_far = memo[()][target_i]
                    
                    prob_this_way = self.m[state][target]
                    if state != "^":
                        prob_this_way += memo[()][state_i]

                    memo[()][target_i] = max(so_far, prob_this_way)
            epsilon_reachables |= targets
            if len(targets) == 0:
                break

        return memo

    @staticmethod
    def kullback(p1, p2):
        if p1 == 0.0:
            return 0.0
        return p1 * math.log(p1/p2)

    @staticmethod
    def squarerr(p1, p2):
        return (p1 - p2) ** 2

    @staticmethod
    def l1err(p1, p2):
        return abs(p1 - p2)

    def distance_from_corpus(self, corpus, distfp, reverse=False,
                             distances={}):
        distance = 0.0
        probs = self.probability_of_strings(list(corpus.keys()))
        for item, corpus_p in corpus.iteritems():
            if corpus_p > 0.0:
                modeled_p = math.exp(probs[item])
                if modeled_p == 0.0:
                    modeled_p = 1e-50

                dist = (distfp(corpus_p, modeled_p) if not reverse
                             else distfp(modeled_p, corpus_p))
                distance += dist
                distances[item] = dist
        return distance

    def round_and_normalize_state(self, state):
        if self.quantizer:
            self.round_transitions(self.m[state]) 
        self.normalize_state(state)
    
    def round_transitions(self, edges):
        for state, weight in edges.iteritems():
            edges[state] = self.quantizer.representer(weight)

    def normalize_state(self, state):
        edges = self.m[state]
        total_log = math.log(sum(math.exp(v) for v in edges.values()))
        for s2 in edges.keys():
            edges[s2] -= total_log
    
    def round_and_normalize(self):
        for state in self.m.iterkeys():
            self.round_and_normalize_state(state)

    def smooth(self):
        """Smooth zero transition probabilities"""
        eps = math.log(Automaton.eps)
        for state, edges in self.m.iteritems():
            for other_state in edges:
                old_val = edges.get(other_state, Automaton.m_inf)
                if old_val < eps:
                    edges[other_state] = eps

            # normalize the transitions
            self.normalize_state(state)
	        
    def boost_edge(self, edge, factor):
        """Adds @factor logprob to @edge"""
        s1, s2 = edge
        self.m[s1][s2] += factor
        self.round_and_normalize_state(s1)
        self.check_state_sum(s1)

    def check_state_sum(self, state):
        edges = self.m[state]
        s_sum = sum([math.exp(log_prob) for log_prob in edges.values()])
        if abs(1.0 - s_sum) < 1e-3:
            return
        else:
            raise Exception("transitions from state {0} ".format(state) + 
                            "don't sum to 1, but {0}".format(s_sum))

    def dump(self, f):
        if self.quantizer is not None:
            emit_bits, trans_bits = self.encoder.automaton_bits(self)
            total_bits = emit_bits + trans_bits
            f.write("total bits: {0} ({1} transition bits, {2} emission bits)\n".format(total_bits, emit_bits, trans_bits))
        states = sorted(self.m.keys())
        for s1 in states:
            for s2 in states + ['$']:
                if s2 in self.m[s1]:
                    f.write("{0} -> {1}: {2}\n".format(
                        s1, s2, self.m[s1][s2]))
        for s1, em in self.emissions.iteritems():
            f.write("{0}: {1}\n".format(s1, repr(em).replace(" ", "")))
    
    def split_state(self, state, new_state, ratio):
        hub_in = 'EPSILON_{0}_{1}_in'.format(state, new_state)
        hub_out = 'EPSILON_{0}_{1}_out'.format(state, new_state)
        self.m[hub_in] = {state+'_0':math.log(1-ratio), new_state+'_0':math.log(ratio)}
        self.m[hub_out] = {}
        self.emissions[new_state+'_0'] = (new_state,)
        self.m_emittors[(new_state,)] = set([new_state+'_0'])
        for s1, trs in self.m.items():
            if s1 in (hub_in, hub_out):
                continue
            for s2, p in trs.items():
                if s2.startswith(state):
                    self.m[s1][hub_in] = p
                    self.m[s1][s2] = float('-inf')
        for s2, p in self.m[state+'_0'].items():
            self.m[hub_out][s2] = p
        self.m[state+'_0'] = {hub_out:0.0}    
        self.m[new_state+'_0'] = {hub_out:0.0}

    def language(self):
        generated_mass = 0.0

        emits = set(self.emissions.itervalues())
        memo = self.init_memo()
        prev_mass = -1.0
        while (abs(generated_mass - prev_mass) >= 1e-4
                   and 1.0 - generated_mass > 0.01):

            prev_mass = generated_mass
            for word in memo.keys():
                for emit in emits:
                    new_word = word + emit
                    self.update_probability_of_string(new_word, memo)

            # filter small probs
            memo = dict([(k, [(None if (lp is None or lp < -100) else lp)
                              for lp in l]) for k, l in memo.iteritems()])

            # filter small prob words
            memo = dict([(k, l) for k, l in memo.iteritems()
                         if sum(filter(lambda x: x is not None, l))>-200])

            # compute generated mass
            generated_mass = sum([math.exp(prob_list[self.state_indices["$"]])
                for s, prob_list in memo.iteritems() if (s != () and
                prob_list[self.state_indices["$"]] is not None )])
            # compute hq - only debug
            # hq = sum([-probs[self.state_indices["$"]] * math.exp(probs[self.state_indices["$"]]) for probs in memo.itervalues()])

        for k in memo.keys():
            if memo[k][self.state_indices["$"]] is None:
                del memo[k]

        return memo

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

    parser.add_option("", "--no-smoothing", dest="smooth", default=True,
                      action="store_false",
                      help="Turn of smoothing")

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
        if not options.smooth:
            automaton.smooth()
        return

    if options.numstate:
        input_ = sys.stdin
        corpus = read_corpus(input_, options.separator)
        alphabet = get_alphabet(corpus)
        numbers_per_letters = dict([(letter, options.numstate)
                              for letter in alphabet])
        if options.num_epsilons:
            numbers_per_letters["EPSILON"] = options.num_epsilons

        automaton = Automaton.create_uniform_automaton(
            numbers_per_letters, initial_transitions)
        if options.smooth:
            automaton.smooth()
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
        if options.smooth:
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
