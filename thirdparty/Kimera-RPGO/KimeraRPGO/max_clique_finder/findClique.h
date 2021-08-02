/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * */
/*   Description:  a library for finding the maximum clique of a graph



/*   Authors: Bharath Pattabiraman and Md. Mostofa Ali Patwary */
/*            EECS Department, Northwestern University */
/*            email: {bpa342,mpatwary}@eecs.northwestern.edu */

/*   Copyright, 2014, Northwestern University */
/*   See COPYRIGHT notice in top-level directory. */

/*   Please site the following publication if you use this package: */
/*   Bharath Pattabiraman, Md. Mostofa Ali Patwary, Assefaw H. Gebremedhin2,

/*   Wei-keng Liao, and Alok Choudhary. */
/*   "Fast Algorithms for the Maximum Clique Problem on Massive Graphs with */
/*   Applications to Overlapping Community Detection"

/*   http://arxiv.org/abs/1411.7460 */

#ifndef KIMERARPGO_MAX_CLIQUE_FINDER_FINDCLIQUE_H_
#define KIMERARPGO_MAX_CLIQUE_FINDER_FINDCLIQUE_H_

#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstddef>
#include <iostream>
#include <vector>

#include "KimeraRPGO/max_clique_finder/graphIO.h"

using namespace std;

#ifdef _DEBUG
int DEBUG = 1;
#endif

namespace FMC {

// Function Definitions
bool fexists(const char* filename);
double wtime();
void usage(char* argv0);
int getDegree(vector<int>* ptrVtx, int idx);
void print_max_clique(vector<int>& max_clique_data);

int maxClique(CGraphIO* gio, int l_bound, vector<int>* max_clique_data);
void maxCliqueHelper(CGraphIO* gio,
                     vector<int>* U,
                     int sizeOfClique,
                     int* maxClq,
                     vector<int>* max_clique_data_inter);

int maxCliqueHeu(CGraphIO* gio, vector<int>* max_clique_data);
void maxCliqueHelperHeu(CGraphIO* gio,
                        vector<int>* U,
                        int sizeOfClique,
                        int* maxClq,
                        vector<int>* max_clique_data_inter);

}  // namespace FMC
#endif  // KIMERARPGO_MAX_CLIQUE_FINDER_FINDCLIQUE_H_
