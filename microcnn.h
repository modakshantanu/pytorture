#ifndef MICROCNN_H
#define MICROCNN_H
#include <bits/stdc++.h>
#include "model.h"


int16_t inbuff[4][40];
float nb[216];
int inbuff_idx = 0;
float ch_avg[3];



#define NB_WRITE(ch,t,r,v,o) (nb[(o + ch + ((t) * (r))) % 216] = (v))
#define NB_READ(ch,t,r,o) (nb[(o + ch + ((t) * (r))) % 216])

#define NB_ADDR(ch,t,r,o) ((o + ch + ((t) * (r))) % 216)


void add_pkt(int x, int y,int z, int p, int r) {
    inbuff[0][inbuff_idx] = x;
    inbuff[1][inbuff_idx] = y;
    inbuff[2][inbuff_idx] = z;
    p /= 10;
    p += 90;
    r /= 10;
    r += 90;
    inbuff[3][inbuff_idx] = ((p << 8) & 0xff00) | (r & 0x00ff);
    inbuff_idx = (inbuff_idx + 1) % 40;

}

// 3 stages
void pre_process() {
    for (int ch = 0; ch < 3; ch++) {
        ch_avg[ch] = 0;
        for (int i = 0; i < 40; i++) {
            ch_avg[ch] += inbuff[ch][i] / 81920.0; 
        }
    }


    int write_addr  =0;
    for (int i = 0; i < 40; i++) {
        nb[write_addr++] = inbuff[0][i] / 2048.0 - ch_avg[0];
        nb[write_addr++] = inbuff[1][i] / 2048.0 - ch_avg[1];
        nb[write_addr++] = inbuff[2][i] / 2048.0 - ch_avg[2];
        nb[write_addr++] = (((uint8_t)(inbuff[3][i] >> 8)) / 90.0) - 1.0;
        nb[write_addr++] = (((uint8_t)(inbuff[3][i] & 0xff)) / 90.0) - 1.0;
    }
}

// 52 stages
void conv1(vector<vector<vector<float>>> &filters, vector<float> &bias) {

    int write_addr = 200;
    int read_addr = -10;
    for (int t = -2; t < 38; t++) {
        for (int f = 0; f < 8; f++) {
            float sum = 0;
            int read_addr_2 = read_addr - 1;
            for (int col = 0; col < 5; col++) {
                bool skip = (t + col < 0 || t+ col >= 40);
                for (int ch = 0; ch < 5; ch++) {
                    read_addr_2++;
                    if (skip) continue;
                    sum += nb[read_addr_2] * filters[f][ch][col];

                }
            }
            sum += bias[f];

            if (sum < 0) sum = 0;

            if (!((unsigned)t & 1)|| nb[write_addr + f] < sum) {
                nb[write_addr + f] = sum;
            }
        }
        if ((unsigned)t & 1) write_addr += 8;
        if (write_addr == 216) write_addr = 0;
        read_addr += 5;
    }
}

// 48 stages
void conv2(vector<vector<vector<float>>> &filters, vector<float> &bias) {


    int write_addr = 144;
    int read_addr = 200 - 2*8;

    for (int t = -2; t < 18; t++) {
        for (int f = 0; f < 16; f++) {
            float sum = 0;
            int read_addr_2 = read_addr - 1;
            for (int col = 0; col < 5; col++) {
                for (int ch = 0; ch < 8; ch++) {
                    read_addr_2++;
                    if (t + col < 0 || t + col >= 20) continue;

                    sum += nb[read_addr_2] * filters[f][ch][col];
                }
                if (read_addr_2 == 215) read_addr_2 = -1;
            }
            sum += bias[f];
            if (sum < 0) sum = 0;

            
            if (!((unsigned)t & 1)|| nb[write_addr + f] < sum) {
                nb[write_addr + f] = sum;
            
            }

        }

        if ((unsigned)t & 1) write_addr += 16;
        if (write_addr == 208) write_addr = 0;
        read_addr += 8;
        if (read_addr == 216) read_addr = 0;
    }
}

void fc_layer(vector<vector<float>> &w, vector<float> &b, vector<vector<float>> &w2, vector<float> &b2) {

    int read_addr = 144;
    for (int i = 0; i < 12; i++) {
        float sum = 0;
        read_addr = 144;
        for (int j = 0; j < 160; j++) {
            sum += nb[read_addr++] * w[i][j];
            if (read_addr == 208) read_addr = 0;
        }
        sum += b[i];
        if (sum < 0) sum = 0;
        nb[i + 96] = sum;
    }

    

    for (int i = 0; i < 12; i++) {
        float sum = 0;
        read_addr = 96;
        for (int j = 0; j < 12; j++) {
            sum += nb[read_addr++] * w2[i][j];
        }
        sum += b2[i];
        nb[108 + i] = sum;

        printf("%.2f " , sum);
    }
    printf("\n");
}

void debug_nb(int ch, int cols,  int offset) {
    for (int i = 0; i < ch; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f\t", NB_READ(i, j, ch, offset));
        }
        printf("\n");
    }
}

#endif