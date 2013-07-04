import sys
from collections import defaultdict
counter = defaultdict(int)
for line in sys.stdin:
    patt, c = line.strip().split()
    for char in patt:
        counter[char]+=int(c)
for char, c in counter.iteritems():
    print char+'\t'+str(c)
