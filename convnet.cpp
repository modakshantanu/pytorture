#include <bits/stdc++.h>
#include "netio.h"
#include "dataset.h"
#include "model.h"
#include "microcnn.h"
using namespace std;

vector<vector<vector<float>>> filter1, filter2;
vector<float> bias1, bias2, bias3;
vector<vector<float>> linear;
vector<pair<vector<vector<float>>, int>> dataset;



vector<float> feedforward(vector<vector<float>> &input) {
    auto res1 = apply_filter(input, filter1, bias1);
    auto res2 = max_pool(res1);
    auto res3 = apply_filter(res2, filter2, bias2);
    auto res4 = max_pool(res3);
    
    //print_2d(res4); 
    // [16][10] => [10][10][10] .... [10]
    auto res5 = flatten(res4);
    auto res6 = linear_net(res5, linear, bias3);
    // print_1d(res6);
    return res6;
}

void pre_process(vector<vector<float>> &input) {
    for (int i = 0; i < 3; i++) {
        float avg = 0;
        for (int j = 0; j < 40; j++) {
            avg += input[i][j];
        }
        avg /= 40;
        for (int j = 0; j < 40; j++) {
            input[i][j] = (input[i][j] - avg)/2048.0;
        }
    }

    for (int i = 3; i < 5; i++) {
        for (int j = 0; j < 40; j++) {
            input[i][j] = (((int)input[i][j]) / 10) /90.0;
        }
    }

    // print_2d(input);
}


void test_accuracy() {

    int correct = 0;
    int total = 0;
    int idx = 0;
    auto linear_t = inner_transpose(linear, 16, 10);
    auto start = std::chrono::system_clock::now();
    for (auto& it: dataset) {
        auto &data = it.first;
        auto &label = it.second;
        for (int t = 0; t < 40; t++) {
            add_pkt(data[0][t], data[1][t], data[2][t], data[3][t], data[4][t]);
        }

        // pre_process(data);

        // auto res = feedforward(data);


        pre_process();
        conv1(filter1, bias1);
        conv2(filter2,bias2);
        fc_layer(linear_t, bias3);
        
        

        bool not_highest = false;
        // for (int i  = 0; i < 11; i++) {
        //     if (res[i] > res[label]) not_highest = true;
        // }
        for (int i = 0; i < 11; i++) {
            if (nb[i + 96] > nb[label + 96]) not_highest = true;
        }

        if (!not_highest) correct++;
        total++;    
        // break;    
        ++idx;
    }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        printf("Elapsed time = %.9f\n", elapsed_seconds.count());

        printf("Accuracy = %.5f\n", (100.0 * correct) / total);
        printf("%d / %d\n", correct, total);

        // Initial measure: 8 seconds 

}


int main() {
    filter1 = vector<vector<vector<float>>>(8, vector<vector<float>>(5, vector<float>(5)));
    filter2 = vector<vector<vector<float>>>(16, vector<vector<float>>(8, vector<float>(5)));

    bias1 = vector<float>(8);
    bias2 = vector<float>(16);
    bias3 = vector<float>(11);
    linear = vector<vector<float>>(11, vector<float>(160));

    read_3d(filter1);
    read_1d(bias1);

    read_3d(filter2);
    read_1d(bias2);

    read_2d(linear);
    read_1d(bias3);

    append_to_dataset(dataset, "data/capstone/24Mar/gun_combined.csv", 5);
    append_to_dataset(dataset, "data/capstone/24Mar/sidepump_combined.csv", 9);
    append_to_dataset(dataset, "data/capstone/24Mar/elbowkick_combined.csv", 6);
    append_to_dataset(dataset, "data/capstone/24Mar/listen_combined.csv", 7);
    append_to_dataset(dataset, "data/capstone/24Mar/pointhigh_combined.csv", 8);
    append_to_dataset(dataset, "data/capstone/24Mar/wipetable_combined.csv", 10);
    append_to_dataset(dataset, "data/capstone/24Mar/dab_combined.csv", 3);
    append_to_dataset(dataset, "data/capstone/24Mar/hair_combined.csv", 4);


    test_accuracy();






    

}
