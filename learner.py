# TODO file structure:
# corpus
# automaton
# learner
# main

# TODO use log prob 
# 

import copy
import logging
import math
import random

from automaton import Automaton

# returns the new automaton, the modified edge and the direction of the modification
# (< 0 decrease, > 0 increase)
# separate class
class Learner(object):
    def __init__(self, automaton, corpus, pref_prob, turns_for_each, distfp, factor=0.97, start_temp=1e-5, end_temp=1e-7, tempq=0.9):
        """
        @param corpus: needs to be normalized with corpus.normalize_corpus()
        @TODO params
        """
        self.start_temperature = start_temp
        self.end_temperature   = end_temp
        self.temp_quotient = tempq
        self.turns_for_each = turns_for_each
        self.preference_probability = pref_prob

        self.automaton = automaton
        self.corpus = corpus

        self.distfp = Automaton.getattr(distfp)
        self.factor = factor
        #self.preferred_node_pair = None
        #self.preferred_direction = None
        #self.disallowed_node_pair = None

    @staticmethod
    def create_from_options(automaton, corpus, options):
        return Learner(automaton, corpus, pref_prob=options.downhill_factor,
                       turns_for_each=options.iter, distfp=options.distfp,
                       tempq=options.tempq, factor=options.factor)

    def change_automaton(self, preferred_node_pair, preferred_direction,
                        disallowed_node_pair):
        nodes = self.automaton.m.keys()
        apply_preference = preferred_node_pair \
          and random.random() < self.preference_probability \
          #and is_valid_transition(preferred_node_pair[0], preferred_node_pair[1])

        # Szandekosan nem pontosan 1 a szorzatuk,
        # igy elvben tud finomhangolni.
        # zseder: 2 az osszeguk inkabb
        if apply_preference and preferred_direction:
            if ((self.factor > 1.0 and preferred_direction < 0.0) or
                (self.factor < 1.0 and preferred_direction > 0.0)):
                factor = 2.0 - self.factor
        else:
            factor = self.factor if random.randrange(2)==0 else 2.0 - self.factor
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
                n2 = random.choice(self.automaton.m[n1].keys())
                if disallowed_node_pair != (n1, n2):
                    break
                #logg("ignore disallowed edge: %s %s" % (n1, n2))
            #logg("change edge: %s %s" % (n1, n2))
        self.automaton.boost_edge(n1, n2, factor)
        return (n1, n2), factor - 1.0

    def learn(self):
        """
        The learning algorithm.
        TODO - separate downhill and simulated annealing
        """
        logging.info( "memoized KL: {0}".format(self.automaton.distance_from_corpus(
            self.corpus, Automaton.kullback)))
        logging.info( "sqerr: {0}".format(self.automaton.distance_from_corpus(
            self.corpus, Automaton.squarerr)))
        logging.info( "l1err: {0}".format(self.automaton.distance_from_corpus(
            self.corpus, Automaton.l1err)))

        # FIXME: avoid deepcopy
        automaton2 = copy.deepcopy(self.automaton)

        # iteration -> learner class
        energy = self.automaton.distance_from_corpus(self.corpus, self.distfp)
        temperature = self.start_temperature
        turn_count = 0
        last_improving_edge = None
        last_improving_direction = None
        last_worsening_edge = None

        preferred_node_pair = None
        preferred_direction = None
        disallowed_node_pair = None
        while True:
            if self.preference_probability:
                preferred_node_pair = last_improving_edge
                preferred_direction = last_improving_direction
                disallowed_node_pair = last_worsening_edge

            changed_edge, change_direction = self.change_automaton(
                preferred_node_pair=preferred_node_pair,
                preferred_direction=preferred_direction,
                disallowed_node_pair=disallowed_node_pair)
            new_energy = self.automaton.distance_from_corpus(
                self.corpus, self.distfp)
            energy_change = new_energy - energy
            if energy_change < 0:
                accept = True
                last_improving_edge = changed_edge
                last_improving_direction = change_direction
                last_worsening_edge = None
                #logg("BETTER %f %f" % (newEnergy,energy))
            else:
                last_improving_edge = None
                last_improving_direction = None
                last_worsening_edge = changed_edge
                still_accepting_probability = random.random()
                accept = (still_accepting_probability < math.exp(-energy_change/temperature))

            if accept:
                automaton2 = copy.deepcopy(self.automaton)
                energy = new_energy
            else:
                automaton = copy.deepcopy(automaton2)

            turn_count += 1
            if turn_count == self.turns_for_each:
                # print "-----"
                #print temperature,"\t",energy
                #sys.stdout.flush()
                logging.info("%s\t%s" % (temperature,energy))
                # dumpAutomaton(automaton)
                turn_count = 0
                temperature *= self.temp_quotient
                if temperature < self.end_temp:
                    break

        automaton.dump()

