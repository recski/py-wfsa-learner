import sys

class Vector(object):

        def __init__(self, a, b):
            self.x = a
            self.y = b

        def __add__(a, b):
            return Vector(a.x + b.x, a.y + b.y)

        def __mul__(a, b):
            return Vector(a.x * b.x, a.x * b.y + a.y * b.x)

def main():

    a = [int(sys.argv[1]), int(sys.argv[2])]
    b = [int(sys.argv[3]), int(sys.argv[4])]   
    a_ring_el = Vector(*a)
    b_ring_el = Vector(*b)
    print (a_ring_el *  b_ring_el).__dict__
         

if __name__ == "__main__" :
         main()
