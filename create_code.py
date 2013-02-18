import sys 

def get_values(b, c):
    return sorted([c*i/((2**b)-1) for i in range(2**b)])
    

def create_code(b, c):
    values = get_values(b, c)
    print bin(0), '-inf', values[0], '-inf'
    code = bin(1)
    for c, v in enumerate(values):
        if c == 0:
            int_start = values[0]
        else:
            int_start = (v+values[c-1])/2
        if c == len(values)-1:
            int_end = values[-1]
        else:
            int_end = (v+values[c+1])/2
        print code, int_start, int_end, v
        code = bin(int(code, 2)+1)
        
def main():
    b = int(sys.argv[1])
    c = float(sys.argv[2])
    create_code(b, c)

if __name__ == '__main__':
    main()
