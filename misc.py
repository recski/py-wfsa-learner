"""Misc functions."""

# Closure: includes all prefixes of the strings.
# Output topologically sorted according to the
# partial ordering of "being a prefix". AKA sorted.
def closure_and_top_sort(strings) :
    closed = set()
    for string in strings :
        for i in range(len(string)+1) :
            closed.add(string[:i])
    # logg( "Closure increased size from %d to %d." % (len(set(strings)),len(closed)) )
    return sorted(list(closed))
