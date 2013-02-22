import sys
import math
import logging
from optparse import OptionParser

def my_round(number, lowest, highest, bits, interval_len=None):
    """rounding with transformation of ranges"""
    if interval_len is None:
        interval_len = (highest - lowest) / bits

    # change interval for easier rounding because we want representers
    # to be not at highest/lowest values. highest/lowest values will be
    # points where round() change value

    lowest -= interval_len / 2.0
    highest += interval_len / 2.0
    new_total_length = highest - lowest + interval_len
    
    normalized_num = number / new_total_length
    transformed_into_new_range = normalized_num * (2 ** bits)
    rounded_in_new_range = round(transformed_into_new_range)
    normalized_after_round = rounded_in_new_range / (2 ** bits)
    transformed_back_into_old_range = normalized_after_round * new_total_length
    return transformed_back_into_old_range

class AbstractCode(object):
    def __init__(self):
        self.rep_to_code = {}

    def representer(self, number):
        """returns a representer element for @number as a float"""
        raise NotImplementedError()

    def code(self, representer):
        """returns a binary code representation of @representer"""
        if representer in self.rep_to_code:
            return self.rep_to_code[representer]

        # handling floating point inaccuracy
        for known_rep in self.rep_to_code.iterkeys():
            if abs(known_rep - representer) < 1e-7:
                return self.rep_to_code[known_rep]

        raise Exception("There is no code for this representer")
    
    def _dump_header(self, ostream):
        ostream.write("#{0}".format(self.__class__.__name__))

    def dump(self, ostream):
        self._dump_header(ostream)
        # after that, comes the real dump, if called

    @staticmethod
    def read(istream):
        """ reads a Code class from a file, and depending on first line,
        it will create one of the sublasses, with calling its read()"""
        l = istream.readline().strip()
        le = l.split("\t")

        class_name = le[0][1:] # because it starts with '#'
        const_args = le[1:]
        if class_name == "LinearCode":
            bits = int(const_args[0])
            coder = LinearCode(bits)
        elif class_name == "LogLinCode":
            bits = int(const_args[0])
            neg_cutoff = float(const_args[1])
            pos_cutoff = float(const_args[2])
            coder = LogLinCode(bits, neg_cutoff, pos_cutoff)
        else:
            raise Exception("Unknown Code class in dump")
        coder.read(istream)
        return coder

class LinearCode(AbstractCode):
    def __init__(self, bits):
        AbstractCode.__init__(self)
        self.bits = bits
        self.interval_to_rep = {}

    @staticmethod
    def create(bits, neg, pos):
        lc = LinearCode(bits)

        # keep to codes for -inf and +inf
        int_len = (pos - neg) / (2 ** bits - 2)
        
        # left most interval
        lc.interval_to_rep[(float("-inf"), neg)] = neg - int_len / 2.0
        lc.rep_to_code[neg - int_len / 2.0] = 0

        # right most interval
        lc.interval_to_rep[(pos, float("inf"))] = pos + int_len / 2.0
        lc.rep_to_code[pos + int_len / 2.0] = 2 ** bits - 1

        for code_i in xrange(1, 2 ** bits - 1):
            interval = (neg + (code_i - 1) * int_len, neg + (code_i) * int_len)
            rep = (interval[1] + interval[0]) / 2.0
            lc.interval_to_rep[interval] = rep
            lc.rep_to_code[rep] = code_i

        return lc

    def representer(self, number):
        for interval, rep in self.interval_to_rep.iteritems():
            if number > interval[0] and number <= interval[1]:
                return rep

    def read(self, istream):
        # read intervals
        for l in istream:
            le = l.strip().split("\t")
            if len(le) != 4:
                raise Exception("LinearCode dump cannot be read, it has " +
                                "a line with not 4 columns")
            code, left, right, rep = le
            code = int(code, 2)
            interval = (float(left), float(right))
            rep = float(rep)
            self.rep_to_code[rep] = code
            self.interval_to_rep[interval] = rep

    def _dump_header(self, ostream):
        AbstractCode._dump_header(self, ostream)
        ostream.write("\t{0}\n".format(self.bits))

    def dump(self, ostream):
        AbstractCode.dump(self, ostream)

        l = [(self.rep_to_code[rep], interval, rep)
              for interval, rep in self.interval_to_rep.iteritems()]
        l.sort(key=lambda x: x[0])
        max_bits = int(math.ceil(math.log(len(l), 2)))
        for code, interval, rep in l:
            # length-adjusted code string
            adjusted_code = bin(code)[2:].rjust(max_bits, "0")

            ostream.write("{0}\t{1}\t{2}\t{3}\n".format(
                adjusted_code, interval[0], interval[1], rep))
    
class LogLinCode(LinearCode):
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
        LinearCode.__init__(self, bits)
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff

        useful_codes = 2 ** bits

        self.rep_to_code[self.neg_cutoff] = 0
        self.interval_to_rep[(float("-inf"),self.neg_cutoff)] = self.neg_cutoff
        useful_codes -= 1

        if self.pos_cutoff != 0:
            self.rep_to_code[self.pos_cutoff] = 2 ** bits - 1
            self.interval_to_rep[(self.pos_cutoff, 0.0)] = self.pos_cutoff
            useful_codes -= 1

        interval_len = float(self.pos_cutoff - self.neg_cutoff) / useful_codes
        self.interval_len = interval_len
        for useful_code_i in xrange(useful_codes):
            code = useful_code_i + 1
            representer = (self.neg_cutoff
                           + ((useful_code_i + 1) * interval_len
                              + useful_code_i * interval_len) / 2.0)
            self.interval_to_rep[(representer - interval_len / 2.0,
                representer + interval_len / 2.0)] = representer
            self.rep_to_code[representer] = code

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
            return val

    def read(self, istream):
        logging.warning("while using LogLinCode class for coding, only " +
                        "header information is used from dump, intervals " +
                        "are counted from them, other lines are discarded")

    def _dump_header(self, ostream):
        AbstractCode._dump_header(self, ostream)
        ostream.write("\t{0}\t{1}\t{2}\n".format(
            self.bits, self.neg_cutoff, self.pos_cutoff))

def optparser():
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="typ", type="choice",
                      choices=["linear", "loglinear_cutoff"],
                      help="what kind of quantizer/coder to create. " + 
                      "Choices are: linear an loglinear with cutoff")

    parser.add_option("-b", "--bits", dest="bits", type="int", default=8,
                      help="how many bits to use. [default=%default]")

    parser.add_option("", "--min", dest="low", type="float", default=-30.,
                      help="lowest value to encode when using linear coder," +
                     "negative cutoff when using loglinear")

    parser.add_option("", "--max", dest="high", type="float", default=0.,
                      help="highest value to encode when using linear coder," +
                     "positive cutoff when using loglinear")

    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      type="str", default=None, help="File containing the " +
                      "dump of the coder object [default=stdout]")

    (options, args) = parser.parse_args()
    return options

def create_coder(options):
    if not options.typ:
        raise Exception("coder type is mandatory")

    output = (open(options.output, "w") if options.output else sys.stdout)

    if options.typ == "linear":
        lc = LinearCode.create(options.bits, options.low, options.high)
        lc.dump(output)
        return
    elif options.typ == "loglinear_cutoff":
        llc = LogLinCode(options.bits, options.low, options.high)
        llc.dump(output)
        return


def main():
    options = optparser()
    create_coder(options)

if __name__ == '__main__':
    main()
