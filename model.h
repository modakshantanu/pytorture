#ifndef MODEL_H
#define MODEL_H

#include <bits/stdc++.h>
using namespace std;

float mx = -1e9, mn = 1e9;

vector<vector<float>> apply_filter(vector<vector<float>> &input, vector<vector<vector<float>>> &filters, vector<float>& bias, int padding = 2) {
    
    if (input.size() != filters[0].size() || filters.size() != bias.size()) {
        cout<<"DIMENSION ERROR";
    }

    vector<vector<float>> padded(input.size(), vector<float>(input[0].size() + padding*2, 0));
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[0].size(); j++) {
            padded[i][j+padding] = input[i][j];
        }
    }

    vector<vector<float>> output(filters.size(), vector<float>(input[0].size() , 0));

    for (int filter = 0; filter < filters.size(); filter++) {
        for (int t = 0; t < input[0].size(); t++) {
            for (int i = 0; i < input.size(); i++) {
                for (int j = 0; j < filters[0][0].size(); j++) {
                    // printf("Multiply %f %f\n", padded[i][t+j] ,filters[filter][i][j]);
                    output[filter][t] += padded[i][t+j] * filters[filter][i][j];
                }
            }
            output[filter][t] += bias[filter];
            if (output[filter][t] < 0) output[filter][t] = 0; // ReLU
        }
    }
    return output; 
}

vector<vector<float>> max_pool(vector<vector<float>> &input) {
    vector<vector<float>> output(input.size());
    for (int i  =0; i < input.size(); i++) {
        for (int j = 0; j < input[0].size(); j += 2) {
            output[i].push_back(max(input[i][j], input[i][j+1]));
        }
    }
    return output;
}

vector<float> linear_net(vector<float> &input, vector<vector<float>> &w, vector<float> &b) {
    vector<float> output = b;
    for (int i = 0; i < b.size(); i++) {
        for (int j = 0; j < input.size(); j++) {
            output[i] += input[j] * w[i][j];
        }
    }
    return output;
}

vector<float> flatten(vector<vector<float>> &input) {
    vector<float> output;

    // for (int i = 0; i < input[0].size(); i++) {
    //     for (int j = 0; j < input.size(); j++) {
    //         output.push_back(input[j][i]);
    //     }
    // }

    for (auto &i: input) {
        for (auto &j: i) {
            output.push_back(j);
        }
    }
    return output;
}

vector<float> relu(vector<float>& input) {
    vector<float> output;
 
    for (int i = 0; input.size(); i++) {
        
        float ans = input[i] > 0 ? input[i] : 0;

        // cout<<ans;

        output.push_back(ans);
    }
    return output;
}

vector<vector<float>> inner_transpose(vector<vector<float>> &input, int r, int c) {

    vector<vector<float>> out(input.size());
    for (int j = 0 ; j < input.size(); j++) {
        vector<float> res(r*c);
        for (int i = 0; i < r*c; i++) {
        res[i] = input[j][10 * (i % 16) + i/16];
        }
        out[j] = res;
    }
    return out;
    
}



#endif