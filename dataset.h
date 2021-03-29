#ifndef DATASET_H
#define DATASET_H
#include <bits/stdc++.h>
using namespace std;

void append_to_dataset(vector<pair<vector<vector<float>>, int>> &dataset, string path, int label) {
    ifstream file;
    file.open(path);

    int num_samples;
    file>>num_samples;

    while(num_samples--) {
        int time_samples;
        file>>time_samples;

        vector<vector<float>> raw_samples(5, vector<float>(time_samples));
        vector<vector<float>> subset(5, vector<float>(40));


        for (int i = 0; i < time_samples; i++) {
            string csvline;
            file>>csvline;
            stringstream str(csvline);
            // 10 11 12 16 17
            for (int j = 0; j < 18; j++) {
                int tmp;
                string ts;
                getline(str, ts, ',');

                // cout<<ts<<" ";
                if (j != 0 && j != 9)
                    tmp = stoi(ts);


                if (j == 10) raw_samples[0][i] = tmp;
                if (j == 11) raw_samples[1][i] = tmp;
                if (j == 12) raw_samples[2][i] = tmp;
                if (j == 16) raw_samples[3][i] = tmp;
                if (j == 17) raw_samples[4][i] = tmp;
            }
            // cout<<endl;

        }
        for (int i = 0; i < time_samples - 40 + 1; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 40; k++) {
                    subset[j][k] = raw_samples[j][i+k];
                }
            }
            dataset.push_back({subset, label});
        }

    }
}

#endif