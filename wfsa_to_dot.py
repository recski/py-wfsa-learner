from collections import defaultdict
import sys

header = 'digraph finite_state_machine {\n\trankdir=LR;\n\tdpi=100;\n\tranksep="5.0 equally";'#\n\tordering=out;'

g_prefixes = set(['^', 'a_0', 'akAr_0', 'bAr_0', 'egy_0', 'mAs_0', 'se_0', 'vala_0'])
g_suffixes = set(['$', 'vala_0', 'Ert_0', 'hogy_0', 'hol_0', 'honnEt_0', 'honnan_0', 'hovA_0', 'hova_0', 'ki_0', 'kor_0', 'meddig_0', 'mely_0', 'melyik_0', 'mennyi_0', 'miErt_0', 'mi_0', 'mikor_0', 'milyen_0'])

g_prefixes = set([s.split('_')[0] for s in g_prefixes])
g_suffixes = set([s.split('_')[0] for s in g_suffixes])
def main(threshold, prefixes=g_prefixes, suffixes=g_suffixes):
    print header
    transitions = defaultdict(list)
    #state_names = {'start':'start', 'end':'end'}
    states = set()
    for line in sys.stdin:
        l = line.strip().split()
        if '->' in l:
            state1, _, state2, weight = l
            weight = float(weight)
            if weight < threshold:
                continue
            state2 = state2[:-1]
            #if not ((state1 in prefixes and state2 in suffixes) or
            #    (state1 == '^' and state2 in prefixes) or
            #    (state1 in suffixes and state2 == '$')):
            #    continue
            if state1 == '^':
                state1 = 'start'
            if state2 == '$':
                state2 = 'end'
            transitions[state1].append((weight, state2))
            states.add(state1)
            states.add(state2)
        else:
            pass
            #state, state_name = l
            #state = state[:-1]
            #state_names[state] = state_name.strip('"')
    
    #print '\tnode [shape = point]; start;'
    #print '\tnode [shape = doublecircle]; end;'
    for state in states:
    #for state, state_name in state_names.iteritems():
        print '\tnode [shape = circle]; {0};'.format(state)
    for state1, transitions in transitions.iteritems():
        for weight, state2 in transitions:
            if weight < threshold:
                print '\t{0} -> {1} [ style=invis ];'.format(
                      state1, state2)
            else:
                print '\t{0} -> {1} [ label = "{2:.4f}" ];'.format(
                      state1, state2, weight)
    print '}'

if __name__ == "__main__":
    if len(sys.argv)==3:
        main(float(sys.argv[1]), [sys.argv[2]])
    elif len(sys.argv)==2:
        main(float(sys.argv[1]))
    else:
        sys.stderr.write('invalid no. of arguments')
        sys.exit(-1)

"""
    node [shape = doublecircle]; S;
    node [shape = point ]; qi
 
    node [shape = circle];
    qi -> S;
    S  -> q1 [ label = "a" ];
    S  -> S  [ label = "a" ];
    q1 -> S  [ label = "a" ];
    q1 -> q2 [ label = "b" ];
    q2 -> q1 [ label = "b" ];
    q2 -> q2 [ label = "b" ];
}
"""
