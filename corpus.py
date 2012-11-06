"""Contains corpus-related stuff."""

# TODO: docstrings to functions

from collections import defaultdict

class CorpusError(Exception):
    """For corpus-related errors."""
    pass

# TODO: put to a class
# Number of fields allowed in readCorpus
__FIELD_RANGE = (1, 2)

def readCorpus(stream, separator=None, silent=False):
    """
    Reads the corpus from @p stream and returns it. The corpus is a
    {word: frequency} (???) map.
    @param separator the key separator (?)
    @param silent if @c True, invalid lines are silently dropped; otherwise,
                  a CorpusException is thrown.
    """
    # TEST
    corpus = defaultdict(float)
    line_no = -1
    for l in stream:
        line_no += 1
        a = l.strip().split()
        if len(a) == 0:
            continue
        elif len(a) not in __FIELD_RANGE:
            if not silent:
                raise CorpusError("Line {0}: number of fields ({1}) " +
                                  "out of range {2}".format(
                                  line_no, len(a), __FIELD_RANGE))
            else:
                continue
        if separator is not None and len(separator) > 0:
            key = tuple(a[0].split(separator))
        else:
            key = a[0]
            if len(a)==1 :
                corpus[key] += 1
            else :
                corpus[key] += int(a[1])
    return corpus

def readDict(stream, silent=False):
    """
    Reads a dictionary from a stream.
    @param silent if @c True, invalid lines are silently dropped; otherwise,
                  a CorpusException is thrown.
    """
    d = {}
    line_no = -1
    for l in stream:
        line_no += 1
        le = l.strip().split()
        if len(le) == 0:
            continue
        elif len(le) != 2:
            if not silent:
                raise CorpusError("Line {0}: number of fields ({1}) != 2".format(
                                  line_no, len(le)))
            else:
                continue
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

