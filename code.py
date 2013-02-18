
def my_round(number, lowest, highest, bits, interval_len=None):
    if interval_len is None:
        interval_len = (highest - lowest) / bits

    # change interval for easier rounding because we want representers
    # to be not at highest/lowest values. highest/lowest values will be
    # points where round() change value

    lowest -= inverval_len
    highest += interval_len

    val = 

class AbstractCode(object):
    def __init__(self):
        #self.codes_to_rep = {}
        self.rep_to_codes = {}
        #self.intervals_to_rep = {}

    def representer(self, number):
        """returns a representer element for @number as a float"""
        raise NotImplementedError()

    def code(self, representer):
        """returns a binary code representation of @representer"""
        raise NotImplementedError()
    
class LogLinCode(AbstractCode):
    """ Class to realize linear quantizing on log space values"""

    def __init__(self, bits, neg_cutoff=-30, pos_cutoff=0):
        """ @bits: on how many bits we want to store information
                   (number of representers)
            @neg_cutoff: below this, everything is represented at cutoff,
                         but above only with epsilon is represented as the
                         next representer
            @pos_cutoff: above this, everything is represented at cutoff,
                         but below only with epsilon is represented as the
                         previous representer
        """
        AbstractCode.__init__(self)
        self.bits = bits
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff

        useful_codes = 2 ** bits

        self.rep_to_codes[self.neg_cutoff] = bin(0)
        useful_codes -= 1

        if self.pos_cutoff != 0:
            self.rep_to_codes[self.pos_cutoff] = bin(2 ** bits - 1)
            useful_codes -= 1

        interval_len = (self.pos_cutoff - self.neg_cutoff) / useful_codes
        self.interval_len = interval_len
        for useful_code_i in xrange(useful_codes):
            code = bin(useful_code_i + 1)
            representer = ((useful_code_i + 1) * interval_len +
                           useful_code_i * interval_len) / 2.0
            self.rep_to_codes[representer] = code

    def representer(self, number):
        """ Because this is a linear coder, easy to locate representers,
        we have cutoffs, though, that we have to watch"""

        # if less than anything, return lowest representer
        if number < self.neg_cutoff:
            return self.neg_cutoff
        # if more than anything, return highest representer
        elif self.pos_cutoff != 0 and number > self.pos_cutoff:
            return self.pos_cutoff
        else:
            val = my_round(number, self.neg_cutoff, self.pos_cutoff,
                           self.bits, self.interval_len)

