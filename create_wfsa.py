import sys
import math
from collections import defaultdict
from corpus import read_corpus, normalize_corpus
from automaton import Automaton

def get_morpheme_frequencies(corpus):
    prefixes = defaultdict(int)
    suffixes = defaultdict(int)
    for word, freq in corpus.iteritems():
        morphemes = word.split('#')
        if len(morphemes)==1:
            prefixes['O']+=freq
            suffixes[morphemes[0]]+=freq
        if len(morphemes)>1:
            prefixes[morphemes[0]]+=freq
            suffixes[morphemes[1]]+=freq
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
    fsa.m['@'][suffix+'_0'] = math.log(p_suff - p_word)
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

def main():
    corpus = read_corpus(sys.stdin)
    n_corpus = normalize_corpus(corpus)
    file_name = sys.argv[1]
    #fsa_creator = lambda corpus: create_three_state_fsa(corpus)
    #fsa_creator = lambda corpus: create_hogy_fsa(corpus)
    fsa_creator = lambda corpus: create_o_fsa(corpus)
    create_wfsa(fsa_creator, file_name, n_corpus)

if __name__ == '__main__':
    main()
