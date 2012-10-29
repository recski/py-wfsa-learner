"""Contains corpus-related stuff."""

# TODO: docstrings to functions

from collections import defaultdict

def readCorpus(stream, separator) :
    corpus = defaultdict(float)
    for l in stream :
        a = l.strip().split()
        assert len(a) in (1,2)
	if len(separator) > 0:
	    key = tuple(a[0].split(separator))
	else:
	    key = a[0]
        if len(a)==1 :
            corpus[key] += 1
        else :
            corpus[key] += int(a[1])
    return corpus

def readDict(stream):
    d = {}
    for l in stream:
        le = l.strip().split()
        d[le[0]] = int(le[1])
    return d

def readTransitions(filename):
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

def getAlphabet(corpus) :
    alphabet = defaultdict(int)
    for w in corpus.keys() :
        for c in w :
            alphabet[c] += 1
    return alphabet

def normalizeCorpus(corpus) :
    total = sum( corpus.values() )
    for item,cnt in corpus.iteritems() :
        corpus[item] = float(cnt)/total
    return corpus

