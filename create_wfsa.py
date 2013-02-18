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
        prefixes[morphemes[0]]+=freq
        if len(morphemes)>1:
            suffixes[morphemes[1]]+=freq
    return prefixes, suffixes

def create_wfsa(fsa_creator, file_name, corp):
    fsa = fsa_creator(corp)
    fsa_file = open(file_name, 'w')
    fsa.dump(fsa_file)

def create_three_state_fsa(corpus):
    prefixes, suffixes = get_morpheme_frequencies(corpus)    
    fsa = Automaton()
    for morpheme in prefixes.keys()+suffixes.keys()+['@']:
        state = morpheme+'_0'
        fsa.emissions[state] = morpheme
        fsa.m_emittors[morpheme].add(state)
    
    total_prefix_freq = sum(prefixes.values())
    total_suffix_freq = sum(suffixes.values())
    
    for prefix, p_freq in prefixes.iteritems(): 
        fsa.m['^'][prefix] = math.log(p_freq/total_prefix_freq)
        fsa.m[prefix]['@'] = 0.0
    
    for suffix, s_freq in suffixes.iteritems():
        fsa.m['@'][suffix] = math.log(s_freq/total_suffix_freq)
        fsa.m[suffix]['$'] = 0.0
    
    return fsa

def main():
    corpus = read_corpus(sys.stdin)
    n_corpus = normalize_corpus(corpus)
    file_name = sys.argv[1]
    fsa_creator = lambda corpus: create_three_state_fsa(corpus)
    create_wfsa(fsa_creator, file_name, n_corpus)

if __name__ == '__main__':
    main()
