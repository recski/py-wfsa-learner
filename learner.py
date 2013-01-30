# TODO use log prob 
# 

import copy
import logging
import math
import random
import sys

from automaton import Automaton

# returns the new automaton, the modified edge and the direction of the modification
# (< 0 decrease, > 0 increase)
# separate class
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

#    def change_automaton(self, preferred_node_pair, preferred_direction,
#                        disallowed_node_pair):
#        nodes = self.automaton.m.keys()
#        apply_preference = preferred_node_pair \
#          and random.random() < self.preference_probability \
#          #and is_valid_transition(preferred_node_pair[0], preferred_node_pair[1])
#
#        # Szandekosan nem pontosan 1 a szorzatuk,
#        # igy elvben tud finomhangolni.
#        # zseder: 2 az osszeguk inkabb
#        if apply_preference and preferred_direction:
#            if ((self.factor > 1.0 and preferred_direction < 0.0) or
#                (self.factor < 1.0 and preferred_direction > 0.0)):
#                factor = 2.0 - self.factor
#        else:
#            factor = self.factor if random.randrange(2)==0 else 2.0 - self.factor
#        if apply_preference:
#            n1, n2 = preferred_node_pair
#        else:
#            n1 = None
#            n2 = None
#            while True:
#                # repeat until we find a valid edge.
#                # there exists at least one emitting state so
#                # there must exist at least three valid edges, '^' -> '$',
#                #  '^' -> emitting_state and emitting_state -> '$'
#                n1 = random.choice(nodes)
#                n2 = random.choice(self.automaton.m[n1].keys())
#                if disallowed_node_pair != (n1, n2):
#                    break
#                #logg("ignore disallowed edge: %s %s" % (n1, n2))
#            #logg("change edge: %s %s" % (n1, n2))
#        self.automaton.boost_edge((n1, n2), factor)
#        return (n1, n2), factor - 1.0

    def change_automaton(self, options=None, revert=False):
        if not revert:
            self.automaton.boost_edge(options["edge"], options["factor"])
        else:
            self.automaton.boost_edge(self.previous_change_options["edge"],
                                      1.0 / self.previous_change_options["factor"])

    def randomize_automaton_change(self):
        change_options = {}
        n1 = None
        n2 = None
        nodes = self.automaton.m.keys()
        while True:
            n1 = random.choice(nodes)
            n2 = random.choice(self.automaton.m[n1].keys())
            if not hasattr(self, "previous_change_options"):
                break
            else:
                if not (self.previous_change_options["result"] == False and
                (n1, n2) == self.previous_change_options["edge"]):
                    break
        change_options["edge"] = (n1, n2)
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
        self.automaton.dump(sys.stdout)

#    def learn(self):
#        """
#        The learning algorithm.
#        TODO - separate downhill and simulated annealing
#        """
#        logging.info( "memoized KL: {0}".format(self.automaton.distance_from_corpus(
#            self.corpus, Automaton.kullback)))
#        logging.info( "sqerr: {0}".format(self.automaton.distance_from_corpus(
#            self.corpus, Automaton.squarerr)))
#        logging.info( "l1err: {0}".format(self.automaton.distance_from_corpus(
#            self.corpus, Automaton.l1err)))
#
#        # FIXME: avoid deepcopy
#        automaton2 = copy.deepcopy(self.automaton)
#
#        # iteration -> learner class
#        energy = self.automaton.distance_from_corpus(self.corpus, self.distfp)
#        temperature = self.start_temperature
#        turn_count = 0
#        last_improving_edge = None
#        last_improving_direction = None
#        last_worsening_edge = None
#
#        preferred_node_pair = None
#        preferred_direction = None
#        disallowed_node_pair = None
#        while True:
#            if turn_count == 0:
#                logging.info("Running an iteration of Simulated Annealing with " +
#                             "{0} temperature and at {1} energy level.".format(
#                             temperature, energy))
#            if self.preference_probability:
#                preferred_node_pair = last_improving_edge
#                preferred_direction = last_improving_direction
#                disallowed_node_pair = last_worsening_edge
#
#            changed_edge, change_direction = self.change_automaton(
#                preferred_node_pair=preferred_node_pair,
#                preferred_direction=preferred_direction,
#                disallowed_node_pair=disallowed_node_pair)
#            new_energy = self.automaton.distance_from_corpus(
#                self.corpus, self.distfp)
#            energy_change = new_energy - energy
#            if energy_change < 0:
#                accept = True
#                last_improving_edge = changed_edge
#                last_improving_direction = change_direction
#                last_worsening_edge = None
#                #logg("BETTER %f %f" % (newEnergy,energy))
#            else:
#                last_improving_edge = None
#                last_improving_direction = None
#                last_worsening_edge = changed_edge
#                still_accepting_probability = random.random()
#                accept = (still_accepting_probability < math.exp(-energy_change/temperature))
#
#            if accept:
#                automaton2 = copy.deepcopy(self.automaton)
#                energy = new_energy
#            else:
#                self.automaton = copy.deepcopy(automaton2)
#
#            turn_count += 1
#            if turn_count == self.turns_for_each:
#                # print "-----"
#                #print temperature,"\t",energy
#                #sys.stdout.flush()
#                # dumpAutomaton(self.automaton)
#                turn_count = 0
#                temperature *= self.temp_quotient
#                if temperature < self.end_temperature:
#                    break
#
#        self.automaton.dump()

