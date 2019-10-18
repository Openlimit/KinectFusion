#ifndef KINECTFUSION_GN_SOLVER_H
#define KINECTFUSION_GN_SOLVER_H

#include "../common.h"
#include "cuda_common.h"
#include <math_constants.h>

#define THREADS_PER_BLOCK_DENSE_DEPTH 128
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64
#define THREADS_PER_BLOCK_DENSE_OVERLAP 512

#define THREADS_PER_BLOCK 512 // keep consistent with the CPU
#define WARP_SIZE 32

#define THREADS_PER_BLOCK_JT_DENSE 128

#define FLOAT_EPSILON 0.000001f

#define ONE_TWENTIETH 0.05f
#define ONE_SIXTH 0.16666667f

template<bool useColor>
struct CUDACachedFrame {
    void alloc(unsigned int width, unsigned int height, unsigned int org_width, unsigned int org_height) {
        CUDA_SAFE_CALL(cudaMalloc(&d_depthMap, sizeof(float) * org_width * org_height));
        CUDA_SAFE_CALL(cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height));
        CUDA_SAFE_CALL(cudaMalloc(&d_cameraposDownsampled, sizeof(Vec4f) * width * height));
        CUDA_SAFE_CALL(cudaMalloc(&d_normalsDownsampled, sizeof(Vec4f) * width * height));

        if (useColor) {
            CUDA_SAFE_CALL(cudaMalloc(&d_intensityDownsampled, sizeof(float) * width * height));
            CUDA_SAFE_CALL(cudaMalloc(&d_intensityDerivsDownsampled, sizeof(Vec2f) * width * height));
        }
    }

    void free() {
        CUDA_SAFE_FREE(d_depthMap);
        CUDA_SAFE_FREE(d_depthDownsampled);
        CUDA_SAFE_FREE(d_cameraposDownsampled);
        CUDA_SAFE_FREE(d_normalsDownsampled);

        if (useColor) {
            CUDA_SAFE_FREE(d_intensityDownsampled);
            CUDA_SAFE_FREE(d_intensityDerivsDownsampled);
        }
    }

    float *d_depthMap;
    float *d_depthDownsampled;
    Vec4f *d_cameraposDownsampled;
    Vec4f *d_normalsDownsampled;

    //for dense color term
    float *d_intensityDownsampled;
    Vec2f *d_intensityDerivsDownsampled;
};

typedef CUDACachedFrame<true> CUDADataFrame;

struct SolverInput {
    unsigned int numberOfImages;

    CUDADataFrame *d_cacheFrames;
    unsigned int denseDepthWidth;
    unsigned int denseDepthHeight;
    Vec4f intrinsics;

    float *weightsDenseDepth;
    float *weightsDenseColor;
};

// State of the GN Solver
struct SolverState {
    float *d_sumResidualDEBUG;
    int *d_numCorrDEBUG;
    float *d_sumResidualColorDEBUG;
    int *d_numCorrColorDEBUG;
    float *d_J;

    Vec3f *d_deltaRot;                    // Current linear update to be computed
    Vec3f *d_deltaTrans;                // Current linear update to be computed

    Vec3f *d_xRot;                        // Current state
    Vec3f *d_xTrans;                    // Current state

    Vec3f *d_rRot;                        // Residuum // jtf
    Vec3f *d_rTrans;                    // Residuum // jtf

    Vec3f *d_zRot;                        // Preconditioned residuum
    Vec3f *d_zTrans;                    // Preconditioned residuum

    Vec3f *d_pRot;                        // Decent direction
    Vec3f *d_pTrans;                    // Decent direction

    Vec3f *d_Ap_XRot;                    // Cache values for next kernel call after A = J^T x J x p
    Vec3f *d_Ap_XTrans;                // Cache values for next kernel call after A = J^T x J x p

    float *d_scanAlpha;                // Tmp memory for alpha scan

    float *d_rDotzOld;                    // Old nominator (denominator) of alpha (beta)

    Vec3f *d_precondionerRot;            // Preconditioner for linear system
    Vec3f *d_precondionerTrans;        // Preconditioner for linear system

    // for dense depth term
    float *d_denseJtJ;
    float *d_denseJtr;
    float *d_denseCorrCounts;

    Mat4f *d_xTransforms;
    Mat4f *d_xTransformInverses;

    uint2 *d_denseOverlappingImages;
    int *d_numDenseOverlappingImages;
};

struct SolverParameters {
    unsigned int nNonLinearIterations;        // Steps of the non-linear solver
    unsigned int nLinIterations;            // Steps of the linear solver

    // bounding box
    Vec3f boundingMax;
    Vec3f boundingMin;

    // dense depth corr
    float denseDistThresh;
    float denseNormalThresh;
    float denseColorThresh;
    float denseColorGradientMin;
    float denseDepthMin;
    float denseDepthMax;

    unsigned int denseOverlapCheckSubsampleFactor;

    float weightDenseDepth;
    float weightDenseColor;

    //Filter
    int minNumOverlapCorr;
    int minNumDenseCorr;
};

void convertMatricesToLiePoses(const Mat4f *d_transforms, unsigned int numTransforms, Vec3f *d_rot, Vec3f *d_trans);

void copyVec3ToVec4(Vec4f *dst, Vec3f *src, int num, float w);

#endif //KINECTFUSION_GN_SOLVER_H
