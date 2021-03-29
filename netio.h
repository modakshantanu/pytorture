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


#endif