#include "Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector_types.h>


void randomizeBodies(float* pos, float* vel, unsigned bodies) {
    unsigned p = 0, v = 0;

    float x, y, z;

    for (int i = 0; i < bodies; i++) {
        x = (float)rand() / RAND_MAX * 2 - 1;
        y = (float)rand() / RAND_MAX * 2 - 1;
        z = (float)rand() / RAND_MAX * 2 - 1;

        float3 point = { x, y, z };

        pos[p++] = point.x;
        pos[p++] = point.y;
        pos[p++] = point.z;
        pos[p++] = 1;

        x = (float)rand() / RAND_MAX * 2 - 1;
        y = (float)rand() / RAND_MAX * 2 - 1;
        z = (float)rand() / RAND_MAX * 2 - 1;

        point = { x, y, z };

        vel[v++] = x;
        vel[v++] = y;
        vel[v++] = z;
    }
}

void readFromFile(float* pos, float* vel, unsigned bodies, const char* path) {
    unsigned p = 0, v = 0;

    float x, y, z, w;

    std::ifstream file;
    file.open(path, std::ios::binary | std::ios::in);

    for (int i = 0; i < bodies; i++) {
        file >> x >> y >> z >> w;

        pos[p++] = x;
        pos[p++] = y;
        pos[p++] = z;
        pos[p++] = w;

        file >> x >> y >> z;
        
        vel[v++] = x;
        vel[v++] = y;
        vel[v++] = z;
    }

    file.close();
}