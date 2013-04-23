"""Contains corpus-related stuff."""

from collections import defaultdict

class CorpusError(Exception):
    """For corpus-related errors."""
    pass

# TODO: put to a class?
# Number of fields allowed in readCorpus
__FIELD_RANGE = (1, 2)

def read_corpus(stream, separator=None, silent=False, skip=None):
    """
    Reads the corpus from stream and returns it. The corpus is a
    {word: frequency} (???) map.
    @param stream input
    @param separator the letter separator
    @param silent if @c True, invalid lines are silently dropped; otherwise,
                  a CorpusException is thrown.
    @param skip: skip characters/strings that will be omitted
    """
    # TEST

    if skip is None:
        skip = []
    skip = set(skip)
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
        key = a[0]
        if separator is not None and len(separator) > 0:
            key = key.split(separator)
        key = tuple((symbol if len(symbol) > 0 else "EPSILON")
                    for symbol in key if symbol not in skip)
        if len(a)==1:
            corpus[key] += 1
        else :
            corpus[key] += int(a[1])
    return corpus

def read_dict(stream, silent=False):
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

def get_alphabet(corpus):
    alphabet = defaultdict(int)
    for w in corpus.keys():
        for c in w:
            alphabet[c] += 1
    return alphabet

def normalize_corpus(corpus):
    total = sum(corpus.values())
    for item,cnt in corpus.iteritems():
        corpus[item] = float(cnt) / total
    return corpus

