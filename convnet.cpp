#include <bits/stdc++.h>
#include "netio.h"
#include "dataset.h"
using namespace std;

vector<vector<vector<float>>> filter1, filter2;
vector<float> bias1, bias2;
vector<vector<float>> linear;
vector<pair<vector<vector<float>>, int>> dataset;


void apply_filter(vector<vector<float>> &input, vector<vector<vector<float>>> &filters, int padding = 2) {
    
    if (input.size() != filters[0].size()) {
        cout<<"DIMENSION ERROR";
    }

    vector<vector<float>> padded(input.size(), vector<float>(input[0].size() + padding*2, 0));
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[0].size(); j++) {
            padded[i][j+padding] = input[i][j];
        }
    }

    

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

    cout<<dataset.size();




    

}
