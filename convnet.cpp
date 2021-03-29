#include <bits/stdc++.h>
#include "netio.h"
using namespace std;

vector<vector<vector<float>>> filter1, filter2;
vector<float> bias1, bias2;
vector<vector<float>> linear;
vector<pair<vector<vector<float>>, int>> dataset;



void append_to_dataset(vector<pair<vector<vector<float>>, int>> &target, string path, int label) {
    ifstream file;
    file.open(path);

    int num_samples;
    file>>num_samples;

    while(num_s)

}

int main() {
    filter1 = vector<vector<vector<float>>>(8, vector<vector<float>>(5, vector<float>(5)));
    filter2 = vector<vector<vector<float>>>(16, vector<vector<float>>(8, vector<float>(5)));

    bias1 = vector<float>(8);
    bias2 = vector<float>(16);

    linear = vector<vector<float>>(11, vector<float>(160));

    read_3d(filter1);
    read_1d(bias1);

    read_3d(filter2);
    read_1d(bias2);

    read_2d(linear);

    append_to_dataset(dataset, "data/capstone/24Mar/gun_combined.csv", 1);





    

}
