#ifndef NETIO_H
#define NETIO_H

#include <bits/stdc++.h>
using namespace std;

void read_3d(vector<vector<vector<float>>> &f) {
    for (auto &x: f) {
        for (auto &y: x) {
            for (auto &z: y) {
                cin>>z;
            }
        }
    }
}

// vector<vector<vector<float>>> trans_3d(vector<vector<vector<float>>> &f) {
//     vector<vector<vector<float>>> res(f.size(), vector<vector<float>>(f[0][0].size(), vector<float>(f[0].size())));
    
//     for (int i = 0; i < res.size(); i++) {
//         for (int j = 0; j < res[0].size(); j++) {
//             for (int k = 0; k < res)
//         }
//     }
// }

void print_3d(vector<vector<vector<float>>> &f) {
    for (auto &x: f) {
        for (auto &y: x) {
            for (auto &z: y) {
                printf("%.3f\t", z);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void read_2d(vector<vector<float>> &l) {
    for (auto &y: l) {
        for (auto &z: y) {
            cin>>z;
        }
    }
}

void print_2d(vector<vector<float>> &l) {
    for (auto &y: l) {
        for (auto &z: y) {
            printf("%.3f\t", z);
        }
        printf("\n");
    }
}

void read_1d(vector<float> &b) {
    for (auto &i: b) cin>>i;
}


void print_1d(vector<float> &b) {
    for (auto &i: b) printf("%.3f\t", i);
    printf("\n");
}

void print_progmem_3d(vector<vector<vector<float>>> &f) {
    cout<<"{";
    for (int i = 0; i < f.size(); i++) {
        cout<<"\n\t{";
        // cout<<"\n\t";
        for (int j = 0; j < f[0].size(); j++) {
            cout<<"{";
            for (int k = 0; k < f[0][0].size(); k++) {
                printf("%.9f",f[i][j][k]);
                if (k != f[0][0].size() - 1) printf(",");
            }
            cout<<"}";
            if (j != f[0].size() -1) cout<<",";
        }
        cout<<"}";
        if (i != f.size() - 1) cout<<",";
    }
    cout<<"\n};\n\n";
}

void print_progmem_2d(vector<vector<float>> &f) {
    cout<<"{";
    for (int i = 0; i < f.size(); i++) {
        cout<<"\n\t{";
        for (int j = 0; j < f[0].size(); j++) {
            printf("%.9f",f[i][j]);
            if (j != f[0].size() - 1) printf(",");
        }
        cout<<"}";
        if (i != f.size() - 1) cout<<",";
    }
    cout<<"\n};\n\n";
}

void print_progmem_1d(vector<float> &f) {
    cout<<"{";
    for (int i = 0; i < f.size(); i ++) {
        printf("%.9f",f[i]);
        if (i != f.size() - 1) printf(",");
    }
    cout<<"};\n\n";
}

#endif