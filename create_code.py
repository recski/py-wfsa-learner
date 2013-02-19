import sys 

def get_linear_interval_boundaries(num_of_intervals, interval):
    """Creates a list of floats with linear splitting @interval
    with @num_of_intervals number of intervals
    WARNING: actually, there are num_of_intervals + 1 intervals, see WARNING
             in create_code()
    """
    min_, max_ = interval
    return sorted([min_ + max_ * i / ((2 ** num_of_intervals) - 1)
                   for i in range(2 ** num_of_intervals)])

def create_code(num_of_intervals, max_value, min_value=0):
    """Creates coding with linear intervals
    WARNING: the representer elements are at linear split points,
             and the first and last interval is half-long as the others
    TODO: create Code class instead of printing"""
    interval_boundaries = get_linear_interval_boundaries(
        num_of_intervals, max_value)

    code = bin(0)
    for c, v in enumerate(interval_boundaries):
        if c == 0:
            int_start = interval_boundaries[0]
        else:
            int_start = (v+interval_boundaries[c-1])/2
        if c == len(interval_boundaries)-1:
            int_end = interval_boundaries[-1]
        else:
            int_end = (v+interval_boundaries[c+1])/2
        print code, int_start, int_end, v
        code = bin(int(code, 2)+1)
        
