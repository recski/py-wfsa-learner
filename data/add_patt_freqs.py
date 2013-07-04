import sys
from collections import defaultdict
counter = defaultdict(int)
for line in sys.stdin:
    patt, c = line.strip().split()
    counter[patt]+=int(c)
for patt, c in counter.iteritems():
    print patt+'\t'+str(c)
