import sys
import math
import logging
from optparse import OptionParser

def my_round(number, lowest, highest, levels, interval_len):
    """rounding with transformation of ranges"""

    # change interval for easier rounding because we want representers
    # to be not at highest/lowest values. highest/lowest values will be
    # points where round() change value

    total_length = highest - lowest 
    normalized_num = number / total_length
    transformed_into_new_range = normalized_num * levels
    rounded_in_new_range = round(transformed_into_new_range)
    normalized_after_round = rounded_in_new_range / levels
    transformed_back_into_old_range = normalized_after_round * total_length
    # shifting a half interval because highest is not a representer
    return transformed_back_into_old_range - interval_len / 2.0

class AbstractQuantizer(object):
    def __init__(self):
        self.rep_to_code = {}
        self.code_to_rep = {}
        self._first = None
        self._last = None

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

    def first(self):
        if self._first is not None:
            return self._first
        s = sorted(self.code_to_rep.keys())
        self._first = s[0]
        self._last = s[-1]
        return self._first

    def last(self):
        if self._last is not None:
            return self._last
        s = sorted(self.code_to_rep.keys())
        self._first = s[0]
        self._last = s[-1]
        return self._last

    def shift(self, element, shift):
        """returns a representer that is @shift levels away (+-)
        from @element"""
        new_element_code = self.rep_to_code[self.representer(element)] + shift
        if new_element_code < self.first():
            new_element_code = self.first()
        if new_element_code > self.last():
            new_element_code = self.last()

        return self.code_to_rep[new_element_code]

    def _dump_header(self, ostream):
        ostream.write("#{0}".format(self.__class__.__name__))

    def dump(self, ostream):
        self._dump_header(ostream)
        # after that, comes the real dump, if called

    @staticmethod
    def read(istream):
        """ reads a Quantizer class from a file, and depending on first line,
        it will create one of the sublasses, with calling its read()"""
        l = istream.readline().strip()
        le = l.split("\t")

        class_name = le[0][1:] # because it starts with '#'
        constr_args = le[1:]
        if class_name == "LinearQuantizer":
            levels = int(constr_args[0])
            quantizer = LinearQuantizer(levels)
        elif class_name == "LogLinQuantizer":
            levels = int(constr_args[0])
            neg_cutoff = float(constr_args[1])
            pos_cutoff = float(constr_args[2])
            quantizer = LogLinQuantizer(levels, neg_cutoff, pos_cutoff)
        else:
            raise Exception("Unknown Quantizer class in dump")
        quantizer.read(istream)
        return quantizer

class LinearQuantizer(AbstractQuantizer):
    def __init__(self, levels):
        AbstractQuantizer.__init__(self)
        self.levels = levels
        self.interval_to_rep = {}

    @staticmethod
    def create(levels, neg, pos):
        lc = LinearQuantizer(levels)

        # keep to codes for -inf and +inf
        int_len = (pos - neg) / (levels - 2)
        lc.interval_len = int_len
        
        # left most interval
        lc.interval_to_rep[(float("-inf"), neg)] = neg - int_len / 2.0
        lc.rep_to_code[neg - int_len / 2.0] = 0

        # right most interval
        lc.interval_to_rep[(pos, float("inf"))] = pos + int_len / 2.0
        lc.rep_to_code[pos + int_len / 2.0] = levels - 1

        for code_i in xrange(1, levels - 1):
            interval = (neg + (code_i - 1) * int_len, neg + (code_i) * int_len)
            rep = (interval[1] + interval[0]) / 2.0
            lc.interval_to_rep[interval] = rep
            lc.rep_to_code[rep] = code_i

        lc.code_to_rep = dict([(v, k) for k, v in lc.rep_to_code.iteritems()])

        return lc

    def representer(self, number):
        for interval, rep in self.interval_to_rep.iteritems():
            if number >= interval[0] and (number <= interval[1]
                    or rep == self.code_to_rep[self.last()]):
                return rep

    def read(self, istream):
        # read intervals
        for l in istream:
            le = l.strip().split("\t")
            if len(le) != 4:
                raise Exception("LinearQuantizer dump cannot be read, it has " +
                                "a line with not 4 columns")
            code, left, right, rep = le
            code = int(code, 2)
            interval = (float(left), float(right))
            rep = float(rep)
            self.rep_to_code[rep] = code
            self.code_to_rep[code] = rep
            self.interval_to_rep[interval] = rep

    def _dump_header(self, ostream):
        AbstractQuantizer._dump_header(self, ostream)
        ostream.write("\t{0}\n".format(self.levels))

    def dump(self, ostream):
        AbstractQuantizer.dump(self, ostream)

        l = [(self.rep_to_code[rep], interval, rep)
              for interval, rep in self.interval_to_rep.iteritems()]
        l.sort(key=lambda x: x[0])
        max_bits = int(math.ceil(math.log(len(l), 2)))
        for code, interval, rep in l:
            # length-adjusted code string
            adjusted_code = bin(code)[2:].rjust(max_bits, "0")

            ostream.write("{0}\t{1}\t{2}\t{3}\n".format(
                adjusted_code, interval[0], interval[1], rep))
    
class LogLinQuantizer(LinearQuantizer):
    """ Class to realize linear quantizing on log space values"""

    def __init__(self, levels, neg_cutoff=-30, pos_cutoff=0):
        """ @levels: how many levels we want to use
            @neg_cutoff: below this, everything is represented at cutoff,
                         but above only with epsilon is represented as the
                         next representer
            @pos_cutoff: above this, everything is represented at cutoff,
                         but below only with epsilon is represented as the
                         previous representer
        """
        LinearQuantizer.__init__(self, levels)
        self.neg_cutoff = neg_cutoff
        self.pos_cutoff = pos_cutoff

        useful_codes = levels

        last_rep = 2 * neg_cutoff
        self.rep_to_code[last_rep] = 0
        self.interval_to_rep[(float("-inf"),self.neg_cutoff)] = last_rep
        useful_codes -= 1

        if abs(self.pos_cutoff) > 1e-10:
            self.rep_to_code[self.pos_cutoff] = levels - 1
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
        self.code_to_rep = dict([(v, k) for k, v in 
                                 self.rep_to_code.iteritems()])

    def read(self, istream):
        logging.warning("while using LogLinQuantizer class for coding, only " +
                        "header information is used from dump, intervals " +
                        "are counted from them, other lines are discarded")

    def _dump_header(self, ostream):
        AbstractQuantizer._dump_header(self, ostream)
        ostream.write("\t{0}\t{1}\t{2}\n".format(
            self.levels, self.neg_cutoff, self.pos_cutoff))

def optparser():
    parser = OptionParser()
    parser.add_option("-t", "--type", dest="typ", type="choice",
                      choices=["linear", "loglinear_cutoff"],
                      help="what kind of quantizer to create. " + 
                      "Choices are: linear an loglinear with cutoff")

    parser.add_option("-n", "--levels", dest="levels", type="int", default=256,
                      help="how many levels to use. [default=%default]")

    parser.add_option("", "--min", dest="low", type="float", default=-30.,
                      help="lowest value to encode when using linear" +
                      "quantizer, negative cutoff when using loglinear")

    parser.add_option("", "--max", dest="high", type="float", default=0.,
                      help="highest value to encode when using linear" +
                      "quantizer, negative cutoff when using loglinear")

    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      type="str", default=None, help="File containing the " +
                      "dump of the quantizer object [default=stdout]")

    (options, args) = parser.parse_args()
    return options

def create_quantizer(options):
    if not options.typ:
        raise Exception("quantizer type is mandatory")

    output = (open(options.output, "w") if options.output else sys.stdout)

    if options.typ == "linear":
        lc = LinearQuantizer.create(options.levels, options.low, options.high)
        lc.dump(output)
        return
    elif options.typ == "loglinear_cutoff":
        llc = LogLinQuantizer(options.levels, options.low, options.high)
        llc.dump(output)
        return


def main():
    options = optparser()
    create_quantizer(options)

if __name__ == '__main__':
    main()
