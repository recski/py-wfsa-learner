# TODO file structure:
# corpus
# automaton
# learner
# main

# TODO use log prob 
# 

import sys
import copy

import logging

import math
import random
from collections import defaultdict

from corpus import read_corpus, read_dict

# Directed graph with edge labels.
# g[a][b] = l

#class Automaton( defaultdict(lambda: defaultdict(float)) ) :
class Automaton :
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

    # This only functions correctly if updateEmittors()
    # was called after creating or updating self.emissions
    def emittors(self,letter) :
        return self.m_emittors[letter]

    def updateEmittors(self) :
        self.m_emittors = {} # Azert nem defaultdict(set), mert igy biztonsagosabb.
	self.m_emittors["EPSILON"] = set() # avoid key error if there are no epsilon states
        for state,letter in self.emissions.iteritems() :
            if letter not in self.m_emittors :
                self.m_emittors[letter] = set()
            self.m_emittors[letter].add(state)

    def copy(self) :
        a2 = Automaton()
        a2.m = self.m.copy()
        a2.emissions = self.emissions.copy()
        a2.m_emittors = self.m_emittors.copy()
        return a2

# REMOVE
def bigramCounts(corpus) :
    bigrams = defaultdict(float)
    for item,cnt in corpus.iteritems() :
        if type(item) == str:
	    item = '^' + item + '$'
	else:
	    # items are tuples
	    item = ('^',) + item + ('$',)
        for i in range(len(item)-1) :
            bigrams[item[i:i+2]] += cnt
#    for item,cnt in bigrams.iteritems() :
#        print item,cnt
    return bigrams

def automatonFromCorpus(corpus) :
    automaton = Automaton()
    alphabet = set()
    for item,cnt in corpus.iteritems() :
        if type(item) == str:
	    item = '^' + item + '$'
	else:
	    # items are tuples
	    item = ('^',) + item + ('$',)
        for i in range(len(item)-1) :
            alphabet.add(item[i])
            alphabet.add(item[i+1])
            automaton.m[item[i]][item[i+1]] += cnt
    for n1,value in automaton.m.iteritems() :
        total = sum(value.values())
        for n2,v in value.iteritems() :
            automaton.m[n1][n2] /= total
    for l in alphabet :
        automaton.emissions[l] = l
    automaton.updateEmittors()
    return automaton

# Ez csak az oldschool automatakra ad helyes eredmenyt,
# ahol mindenki sajat magat bocsatja ki.
# REMOVE
def probability_trivialEmission(automaton,string) :
    p = 1.0
    prev = '^'
    for token in string :
        p *= automaton.m[prev][token]
        prev = token
    p *= automaton.m[prev]['$']
    return p

# Automaton
def nonemitting(state) :
    return state=="^" or state=="$" or is_epsilon_state(state)

# Automaton
def is_epsilon_state(state):
    return state.startswith("EPSILON_")

# Automaton
def is_valid_transition(state1, state2):
    # subsequent non emitting states are not allowed
    # the only exception is '^' -> '$'
    if (state1, state2) == ('^', '$'):
        return True
    return not (nonemitting(state1) and nonemitting(state2)) and \
       not state2 == '^' and not state1 == "$" 


# The probability of the event that 'automaton' emits
# 'string' and finishes the quest at 'state'.
# 'state' did not emit yet:
# It will emit the next symbol following 'string'.
#
# It expects that the updateEmittors() method is called
# sometime before, otherwise the output of automaton.emittors()
# is not up to date.
# Automaton
def probabilityOfState(automaton,string,state) :
    if len(string)==0 :
        return automaton.m["^"][state]
    head = string[:-1]
    tail = string[-1]
    total = 0.0
    for previousState in automaton.emittors(tail) :
        soFar = probabilityOfState(automaton,head,previousState)
        soFar *= automaton.m[previousState][state]
        total += soFar
    return total

# Automaton
# rename: specific name
def probability(automaton,string) :
    return probabilityOfState(automaton,string,"$")

# Closure: includes all prefixes of the strings.
# Output topologically sorted according to the
# partial ordering of "being a prefix". AKA sorted.
def closureAndTopSort(strings) :
    closed = set()
    for string in strings :
        for i in range(len(string)+1) :
            closed.add(string[:i])
    # logg( "Closure increased size from %d to %d." % (len(set(strings)),len(closed)) )
    return sorted(list(closed))

# Expects a list of strings.
# Outputs a map from those strings to probabilities. # TODO switch to log-probs, de nem olyan fontos, mert ugyis csak olyanokra futtatjuk, akik a korpuszunkban elofordulnak.
# This can then be aggregated somewhere else.
# The trick is that it uses memoization
# to reuse results for shorter strings. 

# probabilityOfState common code
# Automaton
def probabilityOfStrings(automaton,strings) :
    topsorted = closureAndTopSort(strings)
    states = set(automaton.m.keys())
    states.add("$")
    states.remove("^")

    # memo[string][state] = probabilityOfState(automaton,string,state)
    memo = defaultdict(lambda: defaultdict(float))
    output = {}

    for string in topsorted :

        # first compute the epsilon states probs because of the
	# memoization dependency
        for state in [ s for s in states if is_epsilon_state(s) ] + \
	             [ s for s in states if not is_epsilon_state(s) ]:
            if len(string)==0 :
                memo[string][state] = automaton.m["^"][state]
            else :
                total = 0.0

		head = string[:-1]
		tail = string[-1]
                for previousState in automaton.emittors(tail) :
                    soFar = memo[head][previousState]
                    soFar *= automaton.m[previousState][state]
                    total += soFar

		# check the case of epsilon emission
		for epsilonState in automaton.emittors("EPSILON"):
		    if not is_valid_transition(epsilonState, state):
		        continue
		    # we already have this value because epsilon states
		    # came first
		    soFar = memo[string][epsilonState]
		    soFar *= automaton.m[epsilonState][state]
		    total += soFar

                memo[string][state] = total

        output[string] = memo[string]["$"]
    return output

# expects a normalized corpus
# This older version does not reuse probability values for prefixes.
# As a sanity check, it should give the same result as kullback().
# REMOVE
def kullbackUnMemoized(corpus,automaton) :
    distance = 0.0
    for item,prob in corpus.iteritems() :
        if prob>0 :
            modeledProb = probability(automaton,item)
            if modeledProb==0.0 :
                modeledProb = 1e-10
                raise Exception("nem kene ezt kezelni?")
            distance += prob * math.log(prob/modeledProb)
    return distance

# Automaton
def kullback(p1,p2):
    return p1 * math.log(p1/p2)

def squarerr(p1,p2):
    return (p1 - p2) ** 2

def l1err(p1,p2):
    return abs(p1 - p2)

def distance(corpus,automaton, distfp):
    distance = 0.0
    probs = probabilityOfStrings(automaton,list(corpus.keys()))
    for item,prob in corpus.iteritems() :
        if prob>0 :
            modeledProb = probs[item]
            if modeledProb==0.0 :
                modeledProb = 1e-10
                raise Exception("nem kene ezt kezelni?")
            distance += distfp(prob, modeledProb)
    return distance

# Automaton
def initialAutomaton(alphabet, initial_transitions) :
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
		logging.warning("invalid state name in init: %s" % s)
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
		    logging.warning("invalid state name in init: %s" % s2)
		    # FIXME
		    assert False
		if not is_valid_transition(s1, s2):
		    logging.warning("%s %s in init is not a valid transition" % (s1, s2))
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
	    
    automaton.updateEmittors()
    return automaton
	    

def normalizeNode(edges) :
    total = sum(edges.values())
    for n2 in edges.keys() :
        edges[n2] /= total

def normalizeAutomaton(automaton) :
    for node,edges in automaton.m.iteritems() :
        normalizeNode(edges)

def smoothAutomaton(automaton):
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
	
	        
# Automaton
def boostEdge(automaton,n1,n2,factor) :
    automaton.m[n1][n2] *= factor
    normalizeNode(automaton.m[n1])
    return automaton

# returns the new automaton, the modified edge and the direction of the modification
# (< 0 decrease, > 0 increase)
# separate class
def changeAutomaton(automaton, factor, preferred_node_pair, preferred_direction,
                    preference_probability, disallowed_node_pair):
    nodes = automaton.m.keys()
    apply_preference = preferred_node_pair and \
      is_valid_transition(preferred_node_pair[0], preferred_node_pair[1]) and \
      random.random() < preference_probability

    # Szandekosan nem pontosan 1 a szorzatuk,
    # igy elvben tud finomhangolni.
    # zseder: 2 az osszeguk inkabb
    if apply_preference and preferred_direction:
        if (factor > 1.0 and preferred_direction < 0.0) or \
	   (factor < 1.0 and preferred_direction > 0.0):
	     factor = 2.0 - factor
	#logg("using preferred factor %f" % factor)
    else:
	factor = factor if random.randrange(2)==0 else 2.0 - factor
    if apply_preference:
        n1, n2 = preferred_node_pair
    else:
        n1 = None
	n2 = None
	while True:
	    # repeat until we find a valid edge.
	    # there exists at least one emitting state so
	    # there must exist at least three valid edges, '^' -> '$',
	    #  '^' -> emitting_state and emitting_state -> '$'
	    n1 = random.choice(nodes)
	    n2 = random.choice(automaton.m[n1].keys())
	    if is_valid_transition(n1, n2) and disallowed_node_pair != (n1, n2):
	        break
	    #logg("ignore disallowed edge: %s %s" % (n1, n2))
        #logg("change edge: %s %s" % (n1, n2))
    automaton = boostEdge(automaton,n1,n2,factor)
    return automaton, (n1, n2), factor - 1.0

# Automaton
def dumpAutomaton(automaton) :
    nodes = sorted(automaton.m.keys())
    for n1 in nodes :
        for n2 in nodes + ['$'] :
            if n2 in automaton.m[n1] :
                print n1,n2,automaton.m[n1][n2]
    for n1,em in automaton.emissions.iteritems() :
        print ">",n1,em

def learn(options) :
    """The learning algorithm -- move to somewhere else."""
    # init
    startTemperature = 1e-5
    endTemperature   = 1e-7
    temperatureQuotient = options.tempq
    turnsForEach = options.iter

    corpus = read_corpus(sys.stdin, options.separator)
    corpus = normalizeCorpus(corpus)

    alphabet = getAlphabet(corpus)

    distfp = globals()[options.distfp]

    initial_transitions = None
    if options.initial_transitions:
	initial_transitions = readTransitions(options.initial_transitions)
        
    logging.info( "Analytical optimum of the chosen error func: %f" % distance(corpus,automatonFromCorpus(corpus),distfp) )
    if options.emitfile:
        numbers_per_letters = read_dict(open(options.emitfile))
        if numbers_per_letters.keys() != alphabet.keys():
            raise Exception("File in wrong format describing emitter states")
        automaton = initialAutomaton(numbers_per_letters, initial_transitions=initial_transitions)
    elif getattr(options, "init_from_corpus", False):
        automaton = automatonFromCorpus(corpus)
	logging.info( "Analytical optimum of KL: %f" % distance(corpus,automaton,kullback) )
	#dumpAutomaton(automaton)
	smoothAutomaton(automaton)
	logging.info( "KL after smoothing: %f" % distance(corpus,automaton,kullback) )
	#dumpAutomaton(automaton)
    else:
        alphabet_numstate = dict([(letter, options.numstate) for letter in alphabet.keys()])
	if options.num_epsilons > 0:
	    # adding states for epsilon emission
	    alphabet_numstate["EPSILON"] = options.num_epsilons
        automaton = initialAutomaton(alphabet_numstate, initial_transitions=initial_transitions)

    #dumpAutomaton(automaton)

    logging.info( "unmemoized KL:\t%f" % kullbackUnMemoized(corpus,automaton) )
    logging.info( "memoized KL:\t%f" % distance(corpus,automaton, kullback) )
    logging.info( "sqerr:\t%f" % distance(corpus,automaton, squarerr) )
    logging.info( "l1err:\t%f" % distance(corpus,automaton, l1err) )

    # FIXME: avoid deepcopy
    automaton2 = copy.deepcopy(automaton) # TODO specialize deepcopy

    # iteration -> learner class
    energy = distance(corpus,automaton, distfp)
    temperature = startTemperature
    turnCount = 0
    last_improving_edge = None
    last_improving_direction = None
    last_worsening_edge = None

    while True :
        preferred_node_pair = None
	preferred_direction = None
	disallowed_node_pair = None
    	if options.downhill_factor:
	    preferred_node_pair = last_improving_edge
	    preferred_direction = last_improving_direction
	    disallowed_node_pair = last_worsening_edge
        automaton, changed_edge, change_direction = \
	    changeAutomaton(automaton, factor=options.factor, preferred_node_pair=preferred_node_pair,
	    	            preferred_direction=preferred_direction,
			    preference_probability=options.downhill_factor,
			    disallowed_node_pair=disallowed_node_pair)
        newEnergy = distance(corpus,automaton, distfp)
        energyChange = newEnergy - energy
        if energyChange<0 :
            accept = True
	    last_improving_edge = changed_edge
	    last_improving_direction = change_direction
	    last_worsening_edge = None
	    #logg("BETTER %f %f" % (newEnergy,energy))
        else :
	    last_improving_edge = None
	    last_improving_direction = None
	    last_worsening_edge = changed_edge
            r = random.random()
            accept = ( r < math.exp(-energyChange/temperature) )
            # if accept : print "WORSE ",newEnergy,energy

        if accept :
            automaton2 = copy.deepcopy(automaton)
            energy = newEnergy
        else :
            automaton = copy.deepcopy(automaton2)

        turnCount += 1
        if turnCount==turnsForEach :
            # print "-----"
	    #print temperature,"\t",energy
	    #sys.stdout.flush()
	    logging.info("%s\t%s" % (temperature,energy))
            # dumpAutomaton(automaton)
            turnCount = 0
            temperature *= temperatureQuotient
            if temperature<endTemperature :
                break

    dumpAutomaton(automaton)

