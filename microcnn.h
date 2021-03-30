#ifndef MICROCNN_H
#define MICROCNN_H
#include <bits/stdc++.h>
#include "model.h"


int16_t inbuff[3][40];
float nb[192];
int inbuff_idx = 0;
float ch_avg[3];


#define INBUFF(ch,t) (\
    t < 0 || t > 39? 0:\
    ch<3? (inbuff[ch][t] / 2048.0 - ch_avg[ch]):\
    ch==3? ((90.0 - acos((inbuff[1][t]/sqrt(inbuff[0][t] * inbuff[0][t] + inbuff[1][t] * inbuff[1][t] + inbuff[2][t] * inbuff[2][t]))) * 180.0 / 3.14159) / 90.0 - 0.5):\
    ((-90.0 + acos((inbuff[0][t]/sqrt(inbuff[0][t] * inbuff[0][t] + inbuff[1][t] * inbuff[1][t] + inbuff[2][t] * inbuff[2][t]))) * 180.0 / 3.14159) / 90.0 - 0.5 )\
)

#define NB_WRITE(ch,t,r,v,o) (nb[(o + ch + ((t) * r)) % 192] = v)
#define NB_READ(ch,t,r,o) (nb[(o + ch + ((t) * r)) % 192])

void add_pkt(int x, int y,int z) {
    inbuff[0][inbuff_idx] = x;
    inbuff[1][inbuff_idx] = y;
    inbuff[2][inbuff_idx] = z;
    inbuff_idx = (inbuff_idx + 1) % 40;
}

void pre_process() {
    for (int ch = 0; ch < 3; ch++) {
        ch_avg[ch] = 0;
        for (int i = 0; i < 40; i++) {
            ch_avg[ch] += inbuff[ch][i] / 81920.0; 
        }
    }
}

void conv1(vector<vector<vector<float>>> &filters, vector<float> &bias) {
    
    for (int t = -2; t < 38; t++) {
        for (int f = 0; f < 8; f++) {
            float sum = 0;
            for (int ch = 0; ch < 5; ch++) {
                for (int col = 0; col < 5; col++) {
                    sum += INBUFF(ch, t + col) * filters[f][ch][col];
                }
            }
            sum += bias[f];

            if (sum < 0) sum = 0;

            

            if (t % 2 == 0) {
                NB_WRITE(f, t/2 + 1, 8, sum, 0);
            } else if (NB_READ(f, t/2 , 8, 0) < sum) {
                NB_WRITE(f, t/2 + 1, 8, sum, 0);
            }
        }
    }
}

void conv2(vector<vector<vector<float>>> &filters, vector<float> &bias) {
    // nb filled from index 0 to 159
    // Start next layer from 160

    for (int t = -2; t < 18; t++) {
        for (int f = 0; f < 16; f++) {
            float sum = 0;
            for (int ch = 0; ch < 8; ch++) {
                for (int col = 0; col < 5; col++) {
                    // Padding of 2
                    if (t + col < 0 || t + col >= 20) continue;

                    // printf("Multipliy %f %f = %f\n", NB_READ(ch, t + col, 20, 0) , filters[f][ch][col], NB_READ(ch, t + col, 20, 0) * filters[f][ch][col]);
                    // printf("Reading from %d\n", (0 + ch + ((t + col) * 8)) % 192);

                    sum += NB_READ(ch, t + col, 8, 0) * filters[f][ch][col];
                }
            }
            sum += bias[f];
            if (sum < 0) sum = 0;
            
            if (t % 2 == 0) {
                // printf("t = %d, f = %d, writing to %d\n", t, f, (160 + f + (t/2 + 1) * 16) % 192);
                NB_WRITE( f, t/2+1, 16, sum, 160);
            } else if (NB_READ(f, t/2 + 1, 16, 160) < sum) {
                // printf("t = %d, f = %d, overwriting %d\n", t, f, (160 + t/2 + 1 + ((f) * 16)) % 192);

                NB_WRITE(f, t/2+1, 16, sum, 160);
            }

        }
    }

}


#endif