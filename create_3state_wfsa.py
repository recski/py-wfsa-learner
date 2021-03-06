"""
This module creates individual wfsa objects
"""
import sys
import math
from collections import defaultdict
from corpus import read_corpus, normalize_corpus
from automaton import Automaton
import logging

def get_morpheme_frequencies(corpus):
    prefixes = defaultdict(int)
    suffixes = defaultdict(int)
    for morphemes, freq in corpus.iteritems():
        if len(morphemes)==2:
            prefixes[morphemes[0]]+=freq
            suffixes[morphemes[1]]+=freq
        elif len(morphemes)==1:
            suffixes[morphemes[0]]+=freq
        else:
            logging.critical("can't have more than two morphemes \
                              in a '3state' FSA")
            raise Exception()
    return prefixes, suffixes

def create_wfsa(fsa_creator, file_name, corp):
    fsa = fsa_creator(corp)
    fsa_file = open(file_name, 'w')
    fsa.dump(fsa_file)

def create_hogy_fsa(corpus):
    fsa = create_three_state_fsa(corpus)
    fsa = move_word_to_separate_edge('O', 'hogy', fsa)
    return fsa

def create_o_fsa(corpus):
    fsa = create_three_state_fsa(corpus)
    for word in corpus.iterkeys():
        if '#' not in word:
            fsa = move_word_to_separate_edge('O', word, fsa)
    return fsa

def move_word_to_separate_edge(prefix, suffix, fsa):
    #print 'moving word', prefix, suffix
    p_word = math.exp(fsa.m['^'][prefix+'_0']+fsa.m['@_0'][suffix+'_0'])
    #print 'P(word) =', p_word
    p_pref = math.exp(fsa.m['^'][prefix+'_0'])
    #print 'P(prefix) =', p_pref
    p_suff = math.exp(fsa.m['@_0'][suffix+'_0'])
    #print 'P(suffix) =', p_suff
    fsa.m['^'][prefix+'_0'] = math.log(p_pref - p_word)
    fsa.m['@_0'][suffix+'_0'] = math.log(p_suff - p_word)
    if prefix == 'O':
        word = suffix
    else:
        word = prefix+suffix
    fsa.emissions[word+'_0'] = word
    fsa.m_emittors[word].add(word+'_0')
    fsa.m['^'][word+'_0'] = math.log(p_word)
    fsa.m[word+'_0']['$'] = 0.0
    fsa.round_and_normalize()
    return fsa

def create_three_state_fsa(corpus):
    prefixes, suffixes = get_morpheme_frequencies(corpus)    
    fsa = Automaton()
    for morpheme in prefixes.keys()+suffixes.keys()+['@', 'O']:
        state = morpheme+'_0'
        fsa.emissions[state] = morpheme
        fsa.m_emittors[morpheme].add(state)

    total_prefix_freq = sum(prefixes.values())
    total_suffix_freq = sum(suffixes.values())
    
    for prefix, p_freq in prefixes.iteritems(): 
        fsa.m['^'][prefix+'_0'] = math.log(p_freq/total_prefix_freq)
        fsa.m[prefix+'_0']['@_0'] = 0.0
    
    for suffix, s_freq in suffixes.iteritems():
        fsa.m['@_0'][suffix+'_0'] = math.log(s_freq/total_suffix_freq)
        fsa.m[suffix+'_0']['$'] = 0.0
    
    return fsa

def create_new_three_state_fsa(corpus, harant="a", emissions="m"):
    """This fsa will not make use of an "O" state and will have edges from
    the start state to all word-final morphemes
    @harant - transitions to the second level
      - "a": for all suffix morphemes
      - list: morphemes and morpheme igrams that will be harant'ed
      - else: No transitions here
    @emissions - morpheme or character emissions
      - "m": morpheme
      - "c": character
    """
    prefixes, suffixes = get_morpheme_frequencies(corpus)    
    fsa = Automaton()
    morphemes_with_index = zip(len(prefixes) * [0], prefixes.keys())
    morphemes_with_index += zip(len(suffixes) * [2], suffixes.keys())
    for index, morpheme in morphemes_with_index:
        state = "{0}_{1}".format(morpheme, index)
        if emissions == "m":
            fsa.emissions[state] = (morpheme,)
            fsa.m_emittors[morpheme].add(state)
        elif emissions == "c":
            fsa.emissions[state] = tuple(morpheme)
            fsa.m_emittors[tuple(morpheme)].add(state)

    total_suffix_freq = sum(suffixes.values())*0.5
    total_prefix_freq = sum(prefixes.values())+total_suffix_freq
    #half of suffix frequencies should appear on edges from the epsilon state,
    #the other half on edges from the starting state
    
    for prefix, p_freq in prefixes.iteritems(): 
        fsa.m['^'][prefix+'_0'] = math.log(p_freq/total_prefix_freq)
        fsa.m[prefix+'_0']['EPSILON_1'] = 0.0
    fsa.m['^']['EPSILON_0'] = -5.0
    fsa.m['EPSILON_0']['EPSILON_1'] = 0.0
    
    for suffix, s_freq in suffixes.iteritems():
        fsa.m['EPSILON_1'][suffix+'_2'] = math.log(
                                        (0.5*s_freq)/total_suffix_freq)
        #the two 0.5s cancel each other out, which reflects that once
        #we got as far as the epsilon state, it doesn't matter anymore
        #whether we allowed suffixes to follow the start state or not
        if harant == "a":
            fsa.m['^'][suffix+'_2'] = math.log((0.5*s_freq)/total_prefix_freq)
        elif type(harant) == list and suffix in harant:
            fsa.m['^'][suffix+'_2'] = math.log((0.5*s_freq)/total_prefix_freq)
        fsa.m[suffix+'_2']['$'] = 0.0

    for prefix, _ in prefixes.iteritems():
        for suffix, _ in suffixes.iteritems():
            if type(harant) == list and (prefix, suffix) in harant:
                state_name = "".join(prefix) + "".join(suffix) + "_3"
                fsa.m["^"][state_name] = -20.0
                fsa.m[state_name]["$"] = 0.0
                fsa.emissions[state_name] = (prefix,suffix)
                fsa.m_emittors[(prefix,suffix)].add(state_name)

    fsa.round_and_normalize()
    
    return fsa


def main():
    corpus = read_corpus(sys.stdin, separator="#")
    n_corpus = normalize_corpus(corpus)
    file_name = sys.argv[1]
    fsa_type = sys.argv[2]
    if fsa_type == 'plain':
        fsa_creator = lambda corpus: create_three_state_fsa(corpus)
    elif fsa_type == 'hogy':
        fsa_creator = lambda corpus: create_hogy_fsa(corpus)
    elif fsa_type == 'o':
        fsa_creator = lambda corpus: create_o_fsa(corpus)
    elif fsa_type == 'new':
        fsa_creator = lambda corpus: create_new_three_state_fsa(corpus, ["hogy", ("vala", "ki")], "m")
    else:
        logging.critical('unknown fsa type: {0}'.format(fsa_type))
        sys.exit(-1)
    
    create_wfsa(fsa_creator, file_name, n_corpus)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    main()
