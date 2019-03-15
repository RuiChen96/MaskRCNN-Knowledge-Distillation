// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include "gpu_nms.hpp"
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
    /* Code block avoids redefinition of cudaError_t error */ \
    do { \
        cudaError_t error = condition; \
        if (error != cudaSuccess) { \
            std::cout << cudaGetErrorString(error) << std::endl; \
        } \
    } while (0)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
    float left = max();
}