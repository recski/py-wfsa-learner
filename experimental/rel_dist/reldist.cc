/**
 * This is a small openFst based program that will compute a relative
 * entropy of two automata based on an article
 * On the Computation of the Relative Entropy of Probabilistic Automata
 * by Cortes et al., 2007 
 * The three input fsts are created by the other scripts from the two
 * original automata whose reldist we are interested in
 * input format(tab seperated):
 *   first line: who is the acceptor
 *   other lines:
 *       src_id tgt_id label_on_edge w1 w2
 *       (two weights because of entropy semiring)
 * output is relative entropy
 * */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cmath>

#include <fst/fstlib.h>
#include <fst/vector-fst.h>
#include <fst/arc.h>
#include <fst/expectation-weight.h>

using namespace fst;
using namespace std;
//typedef ExpectationWeight<TropicalWeight, TropicalWeight> ExpectWeight;
//typedef ExpectationArc<StdArc,TropicalWeight> ExpectArc;
typedef ExpectationWeight<RealWeight, RealWeight> ExpectWeight;
typedef ExpectationArc<StdArc,RealWeight> ExpectArc;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void parse_file(ifstream& myfile, vector<int>& srcs, vector<ExpectArc>& arcs, int& final, int& num_of_states)
{
    string line;
    int src, tgt, label;
    float w1, w2;
    if (myfile.is_open())
    {
        // which is final
        getline (myfile,line);
        vector<string> elems = split(line, ' ');
        final = atoi(elems[0].c_str());
        while (myfile.good())
        {
            getline (myfile,line);
            elems = split(line, '\t');
            if (elems.size() == 5)
            {
                src = atoi(elems[0].c_str());
                tgt = atoi(elems[1].c_str());
                num_of_states = num_of_states >= src ? num_of_states : src;
                num_of_states = num_of_states >= tgt ? num_of_states : tgt;
                label = atoi(elems[2].c_str());
                w1 = log(atof(elems[3].c_str()));
                w2 = log(atof(elems[4].c_str()));
                srcs.push_back(src);
                arcs.push_back(ExpectArc(label, label, ExpectWeight(w1,w2), tgt)); 
            }
        }
        myfile.close();
    }
    num_of_states++;
}

void create_fst(const char* myfilename, VectorFst<ExpectArc>& myfst)
{
    ifstream myfile (myfilename);
    vector<int> srcs;
    vector<ExpectArc> arcs;
    int final, num_of_states = 0;

    parse_file(myfile, srcs, arcs, final, num_of_states);

    for(int i=0; i < num_of_states; i++)
        myfst.AddState();
    myfst.SetStart(0);
    myfst.SetFinal(final, ExpectWeight::One());  // 1st arg is state ID, 2nd arg weight 

    vector<int>::iterator src_it;
    vector<ExpectArc>::iterator arc_it = arcs.begin();
    src_it = srcs.begin();

    for (; src_it != srcs.end(); ++src_it, ++arc_it)
        myfst.AddArc(*src_it, *arc_it); 
}

double shortest(VectorFst<ExpectArc>& fst)
{
    vector<ExpectWeight> distances;
    ShortestDistance(fst, &distances);
    ExpectWeight dist(ExpectWeight::Zero());
    for (StateIterator<VectorFst<ExpectArc> > siter(fst); !siter.Done(); siter.Next())
        if ( fst.Final(siter.Value()) != ExpectWeight::Zero())
            dist = Plus(dist, distances[siter.Value()]);
    return dist.Value2().Value();
}

int main(int argc, char* argv[])
{
    VectorFst<ExpectArc> fst1, fst2, fst3, inter12, inter13;
    create_fst(argv[1], fst1);
    RmEpsilon(&fst1);
    ArcSort(&fst1, OLabelCompare<ExpectArc>());
    create_fst(argv[2], fst2);
    RmEpsilon(&fst2);
    ArcSort(&fst2, ILabelCompare<ExpectArc>());
    create_fst(argv[3], fst3);
    RmEpsilon(&fst3);
    ArcSort(&fst3, ILabelCompare<ExpectArc>());
    Intersect(fst1, fst2, &inter12);
    Intersect(fst1, fst3, &inter13);
    double s12 = shortest(inter12);
    double s13 = shortest(inter13);
    cout << s12 << " " << s13 << " " << s12 - s13 << "\n";
    cout << exp(s12) << " " << exp(s13) << " " << exp(s12) - exp(s13) << "\n";

    return 0;
}
