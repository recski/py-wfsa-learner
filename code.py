import sys
import math

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

    def dump(self, ostream):
        ostream.write("#{0}\n".format(self.__class__.__name__))
        # after that, comes the real dump, if called

    @staticmethod
    def read(istream):
        """ reads a Code class from a file, and depending on first line,
        it will create one of the sublasses, with calling its read()"""

class LinearCode(AbstractCode):
    def __init__(self, bits):
        self.bits = bits
        self.interval_to_rep = {}

    def representer(self, number):
        pass

    def read(self, istream):
        # to count bits later!
        num_of_lines = 0

        # read intervals
        for l in istream:
            le = l.strip().split("\t")
            if len(le) != 4:
                raise Exception("LinearCode dump cannot be read, it has " +
                                "a line with not 4 columns")
            code, left, right, rep = le
            num_of_lines += 1
            code = bin(code)
            interval = (float(left), float(right))
            rep = float(rep)
            self.rep_to_code[rep] = code
            self.interval_to_rep[interval] = rep

        # counting bits
        if abs(math.trunc(math.log(num_of_lines, 2)) - 0.) > 1e-7:
            raise Exception("LinearCode dump contains not power of 2 " +
                            "number of lines as coded intervals")
        self.bits = round(math.log(num_of_lines, 2))

    def dump(self, ostream):
        AbstractCode.dump(self, ostream)
        l = [(self.rep_to_code[rep], interval, rep)
              for interval, rep in self.interval_to_rep.iteritems()]
        l.sort(key=lambda x: x[0])
        max_bits = math.ceil(math.log(len(l), 2))
        for code, interval, rep in l:
            # length-adjusted code string
            adjusted_code = "0b" + str(code)[2:].rjust(max_bits, "0")

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
            return val

    def read(self, istream):
        pass

    def dump(self, ostream):
        pass

def main():
    bits = int(sys.argv[1])
    cutoff = float(sys.argv[2])
    llc = LogLinCode(bits, cutoff)
    llc.dump(sys.stdout)

if __name__ == '__main__':
    main()
