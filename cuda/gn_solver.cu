#include "gn_solver.h"

__inline__ __device__ float warpReduce(float val) {
    int offset = 32 >> 1;
    while (offset > 0) {
        val = val + __shfl_down_sync(FULL_MASK, val, offset, 32);
//        val = val + __shfl_down(val, offset, 32);
        offset = offset >> 1;
    }
    return val;
}

//! compute a rotation exponential using the Rodrigues Formula.
//		rotation axis w (theta = |w|); A = sin(theta) / theta; B = (1 - cos(theta)) / theta^2
__inline__ __device__ void rodrigues_so3_exp(const Vec3f &w, float A, float B, Mat3f &R) {
    {
        const float wx2 = w(0) * w(0);
        const float wy2 = w(1) * w(1);
        const float wz2 = w(2) * w(2);
        R(0, 0) = 1.0f - B * (wy2 + wz2);
        R(1, 1) = 1.0f - B * (wx2 + wz2);
        R(2, 2) = 1.0f - B * (wx2 + wy2);
    }
    {
        const float a = A * w(2);
        const float b = B * (w(0) * w(1));
        R(0, 1) = b - a;
        R(1, 0) = b + a;
    }
    {
        const float a = A * w(1);
        const float b = B * (w(0) * w(2));
        R(0, 2) = b + a;
        R(2, 0) = b - a;
    }
    {
        const float a = A * w(0);
        const float b = B * (w(1) * w(2));
        R(1, 2) = b - a;
        R(2, 1) = b + a;
    }
}

__inline__ __device__ void poseToMatrix(const Vec3f &rot, const Vec3f &trans, Mat4f &matrix) {
    matrix.setIdentity();

    Vec3f translation;
    Mat3f rotation;

    const float theta_sq = rot.dot(rot);
    const float theta = std::sqrt(theta_sq);
    float A, B;

    Vec3f cr = rot.cross(trans);

    if (theta_sq < 1e-8) {
        A = 1.0f - ONE_SIXTH * theta_sq;
        B = 0.5f;
        translation = trans + 0.5f * cr;
    } else {
        float C;
        if (theta_sq < 1e-6) {
            C = ONE_SIXTH * (1.0f - ONE_TWENTIETH * theta_sq);
            A = 1.0f - theta_sq * C;
            B = 0.5f - 0.25f * ONE_SIXTH * theta_sq;
        } else {
            const float inv_theta = 1.0f / theta;
            A = sinf(theta) * inv_theta;
            B = (1 - cosf(theta)) * (inv_theta * inv_theta);
            C = (1 - A) * (inv_theta * inv_theta);
        }

        Vec3f w_cross = rot.cross(cr);
        translation = trans + B * cr + C * w_cross;
    }

    // 3x3 rotation part:
    rodrigues_so3_exp(rot, A, B, rotation);

    //set rotation
    matrix.topLeftCorner(3, 3) = rotation;
    //set translation
    matrix.topRightCorner(3, 1) = translation;
}

__inline__ __device__ Mat4f poseToMatrix(const Vec3f &rot, const Vec3f &trans) {
    Mat4f res;
    poseToMatrix(rot, trans, res);
    return res;
}

//! exponentiate a vector in the Lie algebra to generate a new SO3(a 3x3 rotation matrix).
__inline__ __device__ Mat3f exp_rotation(const Vec3f &w) {
    const float theta_sq = w.dot(w);
    const float theta = std::sqrt(theta_sq);
    float A, B;
    //Use a Taylor series expansion near zero. This is required for
    //accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
    if (theta_sq < 1e-8) {
        A = 1.0f - ONE_SIXTH * theta_sq;
        B = 0.5f;
    } else {
        if (theta_sq < 1e-6) {
            B = 0.5f - 0.25f * ONE_SIXTH * theta_sq;
            A = 1.0f - theta_sq * ONE_SIXTH * (1.0f - ONE_TWENTIETH * theta_sq);
        } else {
            const float inv_theta = 1.0f / theta;
            A = sinf(theta) * inv_theta;
            B = (1 - cosf(theta)) * (inv_theta * inv_theta);
        }
    }

    Mat3f result;
    rodrigues_so3_exp(w, A, B, result);
    return result;
}

//! logarithm of the 3x3 rotation matrix, generating the corresponding vector in the Lie Algebra
__inline__ __device__ Vec3f ln_rotation(const Mat3f &rotation) {
    Vec3f result; // skew symm matrix = (R - R^T) * angle / (2 * sin(angle))

    const float cos_angle = (rotation.trace() - 1.0f) * 0.5f;
    //(R - R^T) / 2
    result(0) = (rotation(2, 1) - rotation(1, 2)) * 0.5f;
    result(1) = (rotation(0, 2) - rotation(2, 0)) * 0.5f;
    result(2) = (rotation(1, 0) - rotation(0, 1)) * 0.5f;

    float sin_angle_abs = result.norm(); //sqrt(result*result);
    if (cos_angle > (float) 0.70710678118654752440) { // [0 - Pi/4[ use asin
        if (sin_angle_abs > 0) {
            result *= asinf(sin_angle_abs) / sin_angle_abs;
        }
    } else if (cos_angle > -(float) 0.70710678118654752440) { // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
        float angle = acosf(cos_angle);
        result *= angle / sin_angle_abs;
    } else {  // rest use symmetric part
        // antisymmetric part vanishes, but still large rotation, need information from symmetric part
        const float angle = CUDART_PI_F - asinf(sin_angle_abs);
        const float
                d0 = rotation(0, 0) - cos_angle,
                d1 = rotation(1, 1) - cos_angle,
                d2 = rotation(2, 2) - cos_angle;
        Vec3f r2;
        if (fabsf(d0) > fabsf(d1) && fabsf(d0) > fabsf(d2)) { // first is largest, fill with first column
            r2(0) = d0;
            r2(1) = (rotation(1, 0) + rotation(0, 1)) * 0.5f;
            r2(2) = (rotation(0, 2) + rotation(2, 0)) * 0.5f;
        } else if (fabsf(d1) > fabsf(d2)) {                // second is largest, fill with second column
            r2(0) = (rotation(1, 0) + rotation(0, 1)) * 0.5f;
            r2(1) = d1;
            r2(2) = (rotation(2, 1) + rotation(1, 2)) * 0.5f;
        } else {                                // third is largest, fill with third column
            r2(0) = (rotation(0, 2) + rotation(2, 0)) * 0.5f;
            r2(1) = (rotation(2, 1) + rotation(1, 2)) * 0.5f;
            r2(2) = d2;
        }
        // flip, if we point in the wrong direction!
        if (r2.dot(result) < 0)
            r2 *= -1;
        result = r2;
        result *= (angle / r2.norm());
    }
    return result;
}

__inline__ __device__ void matrixToPose(const Mat4f &matrix, Vec3f &rot, Vec3f &trans) {
    const Mat3f R = matrix.topLeftCorner(3, 3);
    const Vec3f t = matrix.topRightCorner(3, 1);
    rot = ln_rotation(R);
    const float theta = rot.norm();

    float shtot = 0.5f;
    if (theta > 0.00001f)
        shtot = sinf(theta * 0.5f) / theta;

    // now do the rotation
    Vec3f rot_half = rot;
    rot_half *= -0.5f;
    const Mat3f halfrotator = exp_rotation(rot_half);

    trans = halfrotator * t;

    if (theta > 0.001f)
        trans -= rot * (t.dot(rot) * (1 - 2 * shtot) / rot.dot(rot));
    else
        trans -= rot * (t.dot(rot) / 24);
    trans *= 1.0f / (2 * shtot);
}

__inline__ __device__ void
evalMinusJTFDevice(unsigned int variableIdx, SolverInput &input, SolverState &state, SolverParameters &parameters,
                   Vec3f &resRot, Vec3f &resTrans) {
    // Reset linearized update vector
    state.d_deltaRot[variableIdx].setZero();
    state.d_deltaTrans[variableIdx].setZero();

    //// trans在前，rot在后
    uint3 transIndices = make_uint3(variableIdx * 6 + 0, variableIdx * 6 + 1, variableIdx * 6 + 2);
    uint3 rotIndices = make_uint3(variableIdx * 6 + 3, variableIdx * 6 + 4, variableIdx * 6 + 5);
    resRot = -Vec3f(state.d_denseJtr[rotIndices.x], state.d_denseJtr[rotIndices.y],
                    state.d_denseJtr[rotIndices.z]); //minus since -Jtf, weight already built in
    resTrans = -Vec3f(state.d_denseJtr[transIndices.x], state.d_denseJtr[transIndices.y],
                      state.d_denseJtr[transIndices.z]); //minus since -Jtf, weight already built in
    //// preconditioner
    Vec3f pRot(state.d_denseJtJ[rotIndices.x * input.numberOfImages * 6 + rotIndices.x],
               state.d_denseJtJ[rotIndices.y * input.numberOfImages * 6 + rotIndices.y],
               state.d_denseJtJ[rotIndices.z * input.numberOfImages * 6 + rotIndices.z]);
    Vec3f pTrans(state.d_denseJtJ[transIndices.x * input.numberOfImages * 6 + transIndices.x],
                 state.d_denseJtJ[transIndices.y * input.numberOfImages * 6 + transIndices.y],
                 state.d_denseJtJ[transIndices.z * input.numberOfImages * 6 + transIndices.z]);

    // Preconditioner depends on last solution P(input.d_x)
    if (pRot(0) > FLOAT_EPSILON) state.d_precondionerRot[variableIdx](0) = 1.0f / pRot(0);
    else state.d_precondionerRot[variableIdx](0) = 1.0f;

    if (pRot(1) > FLOAT_EPSILON) state.d_precondionerRot[variableIdx](1) = 1.0f / pRot(1);
    else state.d_precondionerRot[variableIdx](1) = 1.0f;

    if (pRot(2) > FLOAT_EPSILON) state.d_precondionerRot[variableIdx](2) = 1.0f / pRot(2);
    else state.d_precondionerRot[variableIdx](2) = 1.0f;

    if (pTrans(0) > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx](0) = 1.0f / pTrans(0);
    else state.d_precondionerTrans[variableIdx](0) = 1.0f;

    if (pTrans(1) > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx](1) = 1.0f / pTrans(1);
    else state.d_precondionerTrans[variableIdx](1) = 1.0f;

    if (pTrans(2) > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx](2) = 1.0f / pTrans(2);
    else state.d_precondionerTrans[variableIdx](2) = 1.0f;
}

__inline__ __device__ void
applyJTJDenseDevice(unsigned int variableIdx, SolverState &state, float *d_JtJ, unsigned int N, Vec3f &outRot,
                    Vec3f &outTrans, unsigned int threadIdx) {
    // Compute J^T*d_Jp here
    outRot.setZero();
    outTrans.setZero();

    const unsigned int dim = 6 * N;

    unsigned int baseVarIdx = variableIdx * 6;
    unsigned int i = (threadIdx > 0) ? threadIdx : THREADS_PER_BLOCK_JT_DENSE;
    for (; i < N; i += THREADS_PER_BLOCK_JT_DENSE) // iterate through (6) row(s) of JtJ
    {
        // (row, col) = vars, i
        unsigned int baseIdx = 6 * i;

        Mat3f block00, block01, block10, block11;
        block00 << d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 2],
                d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 2],
                d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 2];
        block01 << d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 0) * dim + baseIdx + 5],
                d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 1) * dim + baseIdx + 5],
                d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 2) * dim + baseIdx + 5];
        block10 << d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 2],
                d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 2],
                d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 0], d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 1],
                d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 2];
        block11 << d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 3) * dim + baseIdx + 5],
                d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 4) * dim + baseIdx + 5],
                d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 3], d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 4],
                d_JtJ[(baseVarIdx + 5) * dim + baseIdx + 5];

        //// trans在前，rot在后
        outTrans += (block00 * state.d_pTrans[i] + block01 * state.d_pRot[i]);
        outRot += (block10 * state.d_pTrans[i] + block11 * state.d_pRot[i]);
    }

    outRot(0) = warpReduce(outRot(0));
    outRot(1) = warpReduce(outRot(1));
    outRot(2) = warpReduce(outRot(2));
    outTrans(0) = warpReduce(outTrans(0));
    outTrans(1) = warpReduce(outTrans(1));
    outTrans(2) = warpReduce(outTrans(2));
}

__inline__ __device__ void
computeLieUpdate(const Vec3f &updateW, const Vec3f &updateT, const Vec3f &curW, const Vec3f &curT,
                 Vec3f &newW, Vec3f &newT) {
    Mat4f update = poseToMatrix(updateW, updateT);
    Mat4f cur = poseToMatrix(curW, curT);
    Mat4f n = update * cur;
    matrixToPose(n, newW, newT);
}

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters) {
    const unsigned int N = input.numberOfImages;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    float d = 0.0f;
    if (x > 0 && x < N) {
        Vec3f resRot, resTrans;
        evalMinusJTFDevice(x, input, state, parameters, resRot,
                           resTrans);  // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0

        state.d_rRot[x] = resRot;                                            // store for next iteration
        state.d_rTrans[x] = resTrans;                                        // store for next iteration

        const Vec3f pRot = state.d_precondionerRot[x].cwiseProduct(resRot);            // apply preconditioner M^-1
        state.d_pRot[x] = pRot;

        const Vec3f pTrans = state.d_precondionerTrans[x].cwiseProduct(resTrans);        // apply preconditioner M^-1
        state.d_pTrans[x] = pTrans;

        d = resRot.dot(pRot) + resTrans.dot(
                pTrans);                        // x-th term of nomimator for computing alpha and denominator for computing beta

        state.d_Ap_XRot[x].setZero();
        state.d_Ap_XTrans[x].setZero();
    }

    d = warpReduce(d);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];                // store result for next kernel call
}

__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters) {
    const unsigned int N = input.numberOfImages;                            // Number of block variables
    const unsigned int x = blockIdx.x;
    const unsigned int lane = threadIdx.x % WARP_SIZE;

    if (x > 0 && x < N) {
        Vec3f rot, trans;
        applyJTJDenseDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans,
                            threadIdx.x);            // A x p_k  => J^T x J x p_k

        if (lane == 0) {
            atomicAdd(&state.d_Ap_XRot[x].data()[0], rot(0));//TODO 待检验
            atomicAdd(&state.d_Ap_XRot[x].data()[1], rot(1));
            atomicAdd(&state.d_Ap_XRot[x].data()[2], rot(2));

            atomicAdd(&state.d_Ap_XTrans[x].data()[0], trans(0));
            atomicAdd(&state.d_Ap_XTrans[x].data()[1], trans(1));
            atomicAdd(&state.d_Ap_XTrans[x].data()[2], trans(2));
        }
    }
}

__global__ void PCGStep_Kernel1b(SolverInput input, SolverState state, SolverParameters parameters) {
    const unsigned int N = input.numberOfImages;                                // Number of block variables
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    float d = 0.0f;
    if (x > 0 && x < N) {
        d = state.d_pRot[x].dot(state.d_Ap_XRot[x]) +
            state.d_pTrans[x].dot(state.d_Ap_XTrans[x]);        // x-th term of denominator of alpha
    }

    d = warpReduce(d);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state) {
    const unsigned int N = input.numberOfImages;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    const float dotProduct = state.d_scanAlpha[0];

    float b = 0.0f;
    if (x > 0 && x < N) {
        float alpha = 0.0f;
        if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;        // update step size alpha

        state.d_deltaRot[x] = state.d_deltaRot[x] + alpha * state.d_pRot[x];            // do a decent step
        state.d_deltaTrans[x] = state.d_deltaTrans[x] + alpha * state.d_pTrans[x];    // do a decent step

        Vec3f rRot = state.d_rRot[x] - alpha * state.d_Ap_XRot[x];                    // update residuum
        state.d_rRot[x] = rRot;                                                        // store for next kernel call

        Vec3f rTrans = state.d_rTrans[x] - alpha * state.d_Ap_XTrans[x];                // update residuum
        state.d_rTrans[x] = rTrans;                                                    // store for next kernel call

        Vec3f zRot = state.d_precondionerRot[x].cwiseProduct(rRot);                    // apply preconditioner M^-1
        state.d_zRot[x] = zRot;                                                        // save for next kernel call

        Vec3f zTrans = state.d_precondionerTrans[x].cwiseProduct(rTrans);             // apply preconditioner M^-1
        state.d_zTrans[x] = zTrans;                                                    // save for next kernel call

        b = zRot.dot(rRot) + zTrans.dot(rTrans);              // compute x-th term of the nominator of beta
    }
    b = warpReduce(b);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&state.d_scanAlpha[1], b);
    }
}

template<bool lastIteration>
__global__ void PCGStep_Kernel3(SolverInput input, SolverState state) {
    const unsigned int N = input.numberOfImages;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > 0 && x < N) {
        const float rDotzNew = state.d_scanAlpha[1];                                // get new nominator
        const float rDotzOld = state.d_rDotzOld[x];                                // get old denominator

        float beta = 0.0f;
        if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;                // update step size beta

        state.d_rDotzOld[x] = rDotzNew;                                            // save new rDotz for next iteration
        state.d_pRot[x] = state.d_zRot[x] + beta * state.d_pRot[x];        // update decent direction
        state.d_pTrans[x] = state.d_zTrans[x] + beta * state.d_pTrans[x];        // update decent direction


        state.d_Ap_XRot[x].setZero();
        state.d_Ap_XTrans[x].setZero();

        if (lastIteration) {
            Vec3f rot, trans;
            computeLieUpdate(state.d_deltaRot[x], state.d_deltaTrans[x], state.d_xRot[x], state.d_xTrans[x],
                             rot, trans);
            state.d_xRot[x] = rot;
            state.d_xTrans[x] = trans;
        }
    }
}

void Initialization(SolverInput &input, SolverState &state, SolverParameters &parameters) {
    const unsigned int N = input.numberOfImages;

    const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (blocksPerGrid > THREADS_PER_BLOCK) {
        std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: "
                  << THREADS_PER_BLOCK * THREADS_PER_BLOCK << std::endl;
        while (1);
    }

    CUDA_SAFE_CALL(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));

    PCGInit_Kernel1 << < blocksPerGrid, THREADS_PER_BLOCK >> > (input, state, parameters);

    PCGInit_Kernel2 << < blocksPerGrid, THREADS_PER_BLOCK >> > (N, state);
}

bool PCGIteration(SolverInput &input, SolverState &state, SolverParameters &parameters, bool lastIteration) {
    const unsigned int N = input.numberOfImages;    // Number of block variables

    // Do PCG step
    const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (blocksPerGrid > THREADS_PER_BLOCK) {
        std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: "
                  << THREADS_PER_BLOCK * THREADS_PER_BLOCK << std::endl;
        while (1);
    }

    CUDA_SAFE_CALL(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

    PCGStep_Kernel_Dense << < N, THREADS_PER_BLOCK_JT_DENSE >> > (input, state, parameters);

    PCGStep_Kernel1b << < blocksPerGrid, THREADS_PER_BLOCK >> > (input, state, parameters);

    PCGStep_Kernel2 << < blocksPerGrid, THREADS_PER_BLOCK >> > (input, state);

    float scanAlpha;
    CUDA_SAFE_CALL(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
    if (fabs(scanAlpha) < 5e-7) {
        lastIteration = true;
    }  //todo check this part

    if (lastIteration)
        PCGStep_Kernel3<true> << < blocksPerGrid, THREADS_PER_BLOCK >> > (input, state);
    else
        PCGStep_Kernel3<false> << < blocksPerGrid, THREADS_PER_BLOCK >> > (input, state);

    return lastIteration;
}

inline __device__ float
bilinearInterpolation(float x, float y, const float *d_input, unsigned int imageWidth, unsigned int imageHeight) {
    const Vec2i p00(floorf(x), floorf(y));
    const Vec2i p01 = p00 + Vec2i(0.0f, 1.0f);
    const Vec2i p10 = p00 + Vec2i(1.0f, 0.0f);
    const Vec2i p11 = p00 + Vec2i(1.0f, 1.0f);

    const float alpha = x - p00(0);
    const float beta = y - p00(1);

    float s0 = 0.0f;
    float w0 = 0.0f;
    if (p00(0) < imageWidth && p00(1) < imageHeight) {
        float v00 = d_input[p00(1) * imageWidth + p00(0)];
        if (v00 != NAN) {
            s0 += (1.0f - alpha) * v00;
            w0 += (1.0f - alpha);
        }
    }
    if (p10(0) < imageWidth && p10(1) < imageHeight) {
        float v10 = d_input[p10(1) * imageWidth + p10(0)];
        if (v10 != NAN) {
            s0 += alpha * v10;
            w0 += alpha;
        }
    }

    float s1 = 0.0f;
    float w1 = 0.0f;
    if (p01(0) < imageWidth && p01(1) < imageHeight) {
        float v01 = d_input[p01(1) * imageWidth + p01(0)];
        if (v01 != NAN) {
            s1 += (1.0f - alpha) * v01;
            w1 += (1.0f - alpha);
        }
    }
    if (p11(0) < imageWidth && p11(1) < imageHeight) {
        float v11 = d_input[p11(1) * imageWidth + p11(0)];
        if (v11 != NAN) {
            s1 += alpha * v11;
            w1 += alpha;
        }
    }

    const float p0 = s0 / w0;
    const float p1 = s1 / w1;

    float ss = 0.0f;
    float ww = 0.0f;
    if (w0 > 0.0f) {
        ss += (1.0f - beta) * p0;
        ww += (1.0f - beta);
    }
    if (w1 > 0.0f) {
        ss += beta * p1;
        ww += beta;
    }

    if (ww > 0.0f) return ss / ww;
    else return NAN;
}

inline __device__ Vec2f
bilinearInterpolation(float x, float y, const Vec2f *d_input, unsigned int imageWidth, unsigned int imageHeight) {
    const Vec2i p00(floorf(x), floorf(y));
    const Vec2i p01 = p00 + Vec2i(0.0f, 1.0f);
    const Vec2i p10 = p00 + Vec2i(1.0f, 0.0f);
    const Vec2i p11 = p00 + Vec2i(1.0f, 1.0f);

    const float alpha = x - p00(0);
    const float beta = y - p00(1);

    Vec2f s0(0.0f, 0.0f);
    float w0 = 0.0f;
    if (p00(0) < imageWidth && p00(1) < imageHeight) {
        Vec2f v00 = d_input[p00(1) * imageWidth + p00(0)];
        if (v00(0) != NAN) {
            s0 += (1.0f - alpha) * v00;
            w0 += (1.0f - alpha);
        }
    }
    if (p10(0) < imageWidth && p10(1) < imageHeight) {
        Vec2f v10 = d_input[p10(1) * imageWidth + p10(0)];
        if (v10(0) != NAN) {
            s0 += alpha * v10;
            w0 += alpha;
        }
    }

    Vec2f s1(0.0f, 0.0f);
    float w1 = 0.0f;
    if (p01(0) < imageWidth && p01(1) < imageHeight) {
        Vec2f v01 = d_input[p01(1) * imageWidth + p01(0)];
        if (v01(0) != NAN) {
            s1 += (1.0f - alpha) * v01;
            w1 += (1.0f - alpha);
        }
    }
    if (p11(0) < imageWidth && p11(1) < imageHeight) {
        Vec2f v11 = d_input[p11(1) * imageWidth + p11(0)];
        if (v11(0) != NAN) {
            s1 += alpha * v11;
            w1 += alpha;
        }
    }

    const Vec2f p0 = s0 / w0;
    const Vec2f p1 = s1 / w1;

    Vec2f ss(0.0f, 0.0f);
    float ww = 0.0f;
    if (w0 > 0.0f) {
        ss += (1.0f - beta) * p0;
        ww += (1.0f - beta);
    }
    if (w1 > 0.0f) {
        ss += beta * p1;
        ww += beta;
    }

    if (ww > 0.0f) return ss / ww;
    else return Vec2f(NAN, NAN);
}

inline __device__ Vec4f
bilinearInterpolation(float x, float y, const Vec4f *d_input, unsigned int imageWidth, unsigned int imageHeight) {
    const Vec2i p00(floorf(x), floorf(y));
    const Vec2i p01 = p00 + Vec2i(0.0f, 1.0f);
    const Vec2i p10 = p00 + Vec2i(1.0f, 0.0f);
    const Vec2i p11 = p00 + Vec2i(1.0f, 1.0f);

    const float alpha = x - p00(0);
    const float beta = y - p00(1);

    Vec4f s0(0.f, 0.f, 0.f, 0.f);
    float w0 = 0.0f;
    if (p00(0) < imageWidth && p00(1) < imageHeight) {
        Vec4f v00 = d_input[p00(1) * imageWidth + p00(0)];
        if (v00(0) != NAN) {
            s0 += (1.0f - alpha) * v00;
            w0 += (1.0f - alpha);
        }
    }
    if (p10(0) < imageWidth && p10(1) < imageHeight) {
        Vec4f v10 = d_input[p10(1) * imageWidth + p10(0)];
        if (v10(0) != NAN) {
            s0 += alpha * v10;
            w0 += alpha;
        }
    }

    Vec4f s1(0.f, 0.f, 0.f, 0.f);
    float w1 = 0.0f;
    if (p01(0) < imageWidth && p01(1) < imageHeight) {
        Vec4f v01 = d_input[p01(1) * imageWidth + p01(0)];
        if (v01(0) != NAN) {
            s1 += (1.0f - alpha) * v01;
            w1 += (1.0f - alpha);
        }
    }
    if (p11(0) < imageWidth && p11(1) < imageHeight) {
        Vec4f v11 = d_input[p11(1) * imageWidth + p11(0)];
        if (v11(0) != NAN) {
            s1 += alpha * v11;
            w1 += alpha;
        }
    }

    const Vec4f p0 = s0 / w0;
    const Vec4f p1 = s1 / w1;

    Vec4f ss(0.f, 0.f, 0.f, 0.f);
    float ww = 0.0f;
    if (w0 > 0.0f) {
        ss += (1.0f - beta) * p0;
        ww += (1.0f - beta);
    }
    if (w1 > 0.0f) {
        ss += beta * p1;
        ww += beta;
    }

    if (ww > 0.0f) return ss / ww;
    else return Vec4f(NAN, NAN, NAN, NAN);
}


__inline__ __device__ Vec2f cameraToDepth(float fx, float fy, float cx, float cy, const Vec4f &pos) {
    return Vec2f(pos(0) * fx / pos(2) + cx, pos(1) * fy / pos(2) + cy);
}

__inline__ __device__ Vec4f depthToCamera(float fx, float fy, float cx, float cy, const Vec2i &loc, float depth) {
    const float x = ((float) loc(0) - cx) / fx;
    const float y = ((float) loc(1) - cy) / fy;
    return Vec4f(depth * x, depth * y, depth, 1.0f);
}

__global__ void convertLiePosesToMatrices_Kernel(const Vec3f *d_rot, const Vec3f *d_trans, unsigned int numTransforms,
                                                 Mat4f *d_transforms, Mat4f *d_transformInvs) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTransforms) {
        poseToMatrix(d_rot[idx], d_trans[idx], d_transforms[idx]);
        d_transformInvs[idx] = d_transforms[idx].inverse();
    }
}

void
convertLiePosesToMatrices(const Vec3f *d_rot, const Vec3f *d_trans, unsigned int numTransforms, Mat4f *d_transforms,
                          Mat4f *d_transformInvs) {
    convertLiePosesToMatrices_Kernel << < (numTransforms + 8 - 1) / 8, 8 >> > (d_rot, d_trans, numTransforms,
            d_transforms, d_transformInvs);

}

__global__ void convertMatricesToLiePoses_Kernel(const Mat4f *d_transforms, unsigned int numTransforms,
                                                 Vec3f *d_rot, Vec3f *d_trans) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTransforms) {
        matrixToPose(d_transforms[idx], d_rot[idx], d_trans[idx]);
    }
}

void convertMatricesToLiePoses(const Mat4f *d_transforms, unsigned int numTransforms, Vec3f *d_rot, Vec3f *d_trans) {
    convertMatricesToLiePoses_Kernel << < (numTransforms + 8 - 1) / 8, 8 >> > (d_transforms, numTransforms,
            d_rot, d_trans);
}

__inline__ __device__ bool computeAngleDiff(const Mat4f &transform, float angleThresh) {
    Vec3f x(1.0f, 1.0f, 1.0f);
    x.normalize();
    Vec3f v = transform.topLeftCorner(3, 3) * x;
    float angle = acosf(fmaxf(fminf(x.dot(v), 1.0f), -1.0f));
    return fabsf(angle) < angleThresh;
}

__inline__ __device__ bool isInBoundingBox(const Vec4f cpos, const Mat4f &c2g_transform,
                                           const Vec3f &boundingMin, const Vec3f &boundingMax) {
    Vec4f gpos = c2g_transform * cpos;
    for (int i = 0; i < 3; ++i) {
        if (gpos(i) > boundingMax(i) || gpos(i) < boundingMin(i))
            return false;
    }
    return true;
}

//for pre-filter, no need for normal threshold
__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
                                         float distThresh, const Mat4f &transformi_inv, const Mat4f &transformj,
                                         const Vec4f &intrinsics,
                                         const float *tgtDepth, const float *srcDepth,
                                         float depthMin, float depthMax,
                                         Vec3f &boundingMin, Vec3f &boundingMax) {
    unsigned int x = idx % imageWidth;
    unsigned int y = idx / imageWidth;
    Vec2i loc(x, y);
    const Vec4f cposj = depthToCamera(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3), loc, srcDepth[idx]);
    if (cposj(2) > depthMin && cposj(2) < depthMax && isInBoundingBox(cposj, transformj, boundingMin, boundingMax)) {
        Vec4f camPosSrcToTgt = transformi_inv * transformj * cposj;
        Vec2f tgtScreenPosf = cameraToDepth(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3),
                                            camPosSrcToTgt);
        Vec2i tgtScreenPos((int) roundf(tgtScreenPosf(0)), (int) roundf(tgtScreenPosf(1)));
        if (tgtScreenPos(0) >= 0 && tgtScreenPos(1) >= 0 && tgtScreenPos(0) < (int) imageWidth &&
            tgtScreenPos(1) < (int) imageHeight) {
            Vec4f camPosTgt = depthToCamera(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3),
                                            tgtScreenPos,
                                            tgtDepth[tgtScreenPos(1) * imageWidth + tgtScreenPos(0)]);
            if (camPosTgt(2) > depthMin && camPosTgt(2) < depthMax) {
                Vec4f diff = camPosSrcToTgt - camPosTgt;
                if (diff.norm() <= distThresh) {
                    return true;
                }
            }
        } // valid projection
    } // valid src camera position
    return false;
}

__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
                                         float distThresh, float normalThresh, const Mat4f &transformi_inv,
                                         const Mat4f &transformj, const Vec4f &intrinsics,
                                         const float *tgtDepth, const Vec4f *tgtNormals,
                                         const float *srcDepth, const Vec4f *srcNormals,
                                         float depthMin, float depthMax,
                                         Vec3f &boundingMin, Vec3f &boundingMax) {
    unsigned int x = idx % imageWidth;
    unsigned int y = idx / imageWidth;
    Vec2i loc(x, y);
    const Vec4f cposj = depthToCamera(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3), loc, srcDepth[idx]);
    if (cposj(2) > depthMin && cposj(2) < depthMax && isInBoundingBox(cposj, transformj, boundingMin, boundingMax)) {
        Vec4f nrmj = srcNormals[idx];
        if (nrmj(0) != NAN) {
            nrmj = transformi_inv * transformj * nrmj;
            Vec4f camPosSrcToTgt = transformi_inv * transformj * cposj;
            Vec2f tgtScreenPosf = cameraToDepth(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3),
                                                camPosSrcToTgt);
            Vec2i tgtScreenPos((int) roundf(tgtScreenPosf(0)), (int) roundf(tgtScreenPosf(1)));
            if (tgtScreenPos(0) >= 0 && tgtScreenPos(1) >= 0 && tgtScreenPos(0) < (int) imageWidth &&
                tgtScreenPos(1) < (int) imageHeight) {
                Vec4f camPosTgt = depthToCamera(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3),
                                                tgtScreenPos,
                                                tgtDepth[tgtScreenPos(1) * imageWidth + tgtScreenPos(0)]);
                if (camPosTgt(2) > depthMin && camPosTgt(2) < depthMax) {
                    Vec4f normalTgt = tgtNormals[tgtScreenPos(1) * imageWidth + tgtScreenPos(0)];
                    if (normalTgt(0) != NAN) {
                        Vec4f diff = camPosSrcToTgt - camPosTgt;
                        float dist = diff.norm();
                        float dNormal = nrmj.dot(normalTgt);
                        if (dNormal >= normalThresh && dist <= distThresh) {
                            return true;
                        }
                    }
                }
            } // valid projection
        } // valid src normal
    } // valid src camera position
    return false;
}

//using camera positions
__device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
                              float distThresh, float normalThresh, const Mat4f &transformi_inv,
                              const Mat4f &transformj, const Vec4f &intrinsics,
                              const Vec4f *tgtCamPos, const Vec4f *tgtNormals,
                              const Vec4f *srcCamPos, const Vec4f *srcNormals,
                              float depthMin, float depthMax,
                              Vec3f &boundingMin, Vec3f &boundingMax,
                              Vec4f &camPosSrc, Vec4f &camPosSrcToTgt,
                              Vec2f &tgtScreenPosf, Vec4f &camPosTgt, Vec4f &normalTgt) {
    const Vec4f cposj = srcCamPos[idx];
    if (cposj(2) > depthMin && cposj(2) < depthMax && isInBoundingBox(cposj, transformj, boundingMin, boundingMax)) {
        camPosSrc = cposj;
        Vec4f nrmj = srcNormals[idx];
        if (nrmj(0) != NAN) {
            nrmj = transformi_inv * transformj * nrmj;
            camPosSrcToTgt = transformi_inv * transformj * camPosSrc;
            tgtScreenPosf = cameraToDepth(intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3), camPosSrcToTgt);
            Vec2i tgtScreenPos((int) roundf(tgtScreenPosf(0)), (int) roundf(tgtScreenPosf(1)));
            if (tgtScreenPos(0) >= 0 && tgtScreenPos(1) >= 0 && tgtScreenPos(0) < (int) imageWidth &&
                tgtScreenPos(1) < (int) imageHeight) {
                Vec4f cposi = bilinearInterpolation(tgtScreenPosf(0), tgtScreenPosf(1), tgtCamPos,
                                                    imageWidth, imageHeight);
                if (cposi(2) > depthMin && cposi(2) < depthMax) {
                    camPosTgt = cposi;
                    Vec4f nrmi = bilinearInterpolation(tgtScreenPosf(0), tgtScreenPosf(1), tgtNormals,
                                                       imageWidth, imageHeight);
                    if (nrmi(0) != NAN) {
                        normalTgt = nrmi;
                        Vec4f diff = camPosSrcToTgt - camPosTgt;
                        float dist = diff.norm();
                        float dNormal = nrmj.dot(nrmi);
                        if (dNormal >= normalThresh && dist <= distThresh) {
                            return true;
                        }
                    }
                }
            } // valid projection
        } // valid src normal
    } // valid src camera position
    return false;
}

__inline__ __device__ Mat3f VectorToSkewSymmetricMatrix(const Vec3f &v) {
    Mat3f res;
    res.setZero();
    res(1, 0) = v(2);
    res(2, 0) = -v(1);
    res(2, 1) = v(0);
    res(0, 1) = -v(2);
    res(0, 2) = v(1);
    res(1, 2) = -v(0);
    return res;
}

inline __device__ Mat23f dCameraToScreen(const Vec3f &p, float fx, float fy) {
    Mat23f res;
    res.setZero();
    const float wSquared = p(2) * p(2);

    res(0, 0) = fx / p(2);
    res(1, 1) = fy / p(2);
    res(0, 2) = -fx * p(0) / wSquared;
    res(1, 2) = -fy * p(1) / wSquared;

    return res;
}

/////////////////////////////////////////////////////////////////////////
// deriv for Ti: (A * e^e * D)^{-1} * p; A = Tj^{-1}; D = Ti
/////////////////////////////////////////////////////////////////////////
__inline__ __device__ Mat36f evalLie_derivI(const Mat4f &A, const Mat4f &D, const Vec3f &p) {
    Mat312f j0;
    Mat126f j1;
    const Mat4f transform = A * D;
    Vec3f pt = p - transform.topRightCorner(3, 1);
    j0.setZero();
    j1.setZero();
    j0(0, 0) = pt(0);
    j0(0, 1) = pt(1);
    j0(0, 2) = pt(2);
    j0(1, 3) = pt(0);
    j0(1, 4) = pt(1);
    j0(1, 5) = pt(2);
    j0(2, 6) = pt(0);
    j0(2, 7) = pt(1);
    j0(2, 8) = pt(2);
    for (unsigned int r = 0; r < 3; r++) {
        for (unsigned int c = 0; c < 3; c++) {
            j0(r, c + 9) = -transform(c, r); //-R(AD)^T
            j1(r + 9, c) = A(r, c);     // R(A)
        }
    }

    Mat3f RA = A.topLeftCorner(3, 3);
    for (unsigned int k = 0; k < 4; k++) {
        Vec3f v(D(0, k), D(1, k), D(2, k));
        Mat3f ss = VectorToSkewSymmetricMatrix(v);
        Mat3f m = RA * ss * -1.0f; //RA * col k of D
        for (unsigned int r = 0; r < 3; r++) {
            for (unsigned int c = 0; c < 3; c++)
                j1(3 * k + r, 3 + c) = m(r, c);
        }
    }

    return j0 * j1;
}


/////////////////////////////////////////////////////////////////////////
// deriv for Tj: (A * e^e * D) * p; A = Ti^{-1}; D = Tj
/////////////////////////////////////////////////////////////////////////
__inline__ __device__ Mat36f evalLie_derivJ(const Mat4f &A, const Mat4f &D, const Vec3f &p) {
    Vec3f dr1(D(0, 0), D(0, 1), D(0, 2));    //rows of D (rotation part)
    Vec3f dr2(D(1, 0), D(1, 1), D(1, 2));
    Vec3f dr3(D(2, 0), D(2, 1), D(2, 2));
    float dtx = D(0, 3);    //translation of D
    float dty = D(1, 3);
    float dtz = D(2, 3);
    Mat36f jac;
    jac(0, 0) = 1.0f;
    jac(0, 1) = 0.0f;
    jac(0, 2) = 0.0f;
    jac(1, 0) = 0.0f;
    jac(1, 1) = 1.0f;
    jac(1, 2) = 0.0f;
    jac(2, 0) = 0.0f;
    jac(2, 1) = 0.0f;
    jac(2, 2) = 1.0f;
    jac(0, 3) = 0.0f;
    jac(0, 4) = p.dot(dr3) + dtz;
    jac(0, 5) = -(p.dot(dr2) + dty);
    jac(1, 3) = -(p.dot(dr3) + dtz);
    jac(1, 4) = 0.0f;
    jac(1, 5) = p.dot(dr1) + dtx;
    jac(2, 3) = p.dot(dr2) + dty;
    jac(2, 4) = -(p.dot(dr1) + dtx);
    jac(2, 5) = 0.0f;

    jac = A.topLeftCorner(3, 3) * jac;
    return jac;
}

__inline__ __device__ void computeJacobianBlockRow_i(Vec6f &jacBlockRow, const Mat4f &transform_i,
                                                     const Mat4f &invTransform_j, const Vec4f &camPosSrc,
                                                     const Vec4f &normalTgt) {
    Vec3f camPosSrc_ = camPosSrc.head(3);
    Vec3f normalTgt_ = normalTgt.head(3);

    Mat36f jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc_);
    for (unsigned int i = 0; i < 6; i++) {
        Vec3f v(jac(0, i), jac(1, i), jac(2, i));
        jacBlockRow(i) = -v.dot(normalTgt_); //rot
    }
}

__inline__ __device__ void computeJacobianBlockRow_j(Vec6f &jacBlockRow, const Mat4f &invTransform_i,
                                                     const Mat4f &transform_j, const Vec4f &camPosSrc,
                                                     const Vec4f &normalTgt) {
    Vec3f camPosSrc_ = camPosSrc.head(3);
    Vec3f normalTgt_ = normalTgt.head(3);

    Mat36f jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc_);
    for (unsigned int i = 0; i < 6; i++) {
        Vec3f v(jac(0, i), jac(1, i), jac(2, i));
        jacBlockRow(i) = -v.dot(normalTgt_); //rot
    }
}

__inline__ __device__ void
computeJacobianBlockIntensityRow_i(Vec6f &jacBlockRow, const Vec2f &colorFocal, const Mat4f &transform_i,
                                   const Mat4f &invTransform_j, const Vec4f &camPosSrc, const Vec4f &camPosSrcToTgt,
                                   const Vec2f &intensityDerivTgt) {
    Vec3f camPosSrc_ = camPosSrc.head(3);
    Vec3f camPosSrcToTgt_ = camPosSrcToTgt.head(3);

    Mat36f jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc_);
    Mat23f dProj = dCameraToScreen(camPosSrcToTgt_, colorFocal(0), colorFocal(1));
    Vec2f dColorB = intensityDerivTgt;
    jacBlockRow = jac.transpose() * dProj.transpose() * dColorB;
}

__inline__ __device__ void
computeJacobianBlockIntensityRow_j(Vec6f &jacBlockRow, const Vec2f &colorFocal, const Mat4f &invTransform_i,
                                   const Mat4f &transform_j, const Vec4f &camPosSrc, const Vec4f &camPosSrcToTgt,
                                   const Vec2f &intensityDerivTgt) {
    Vec3f camPosSrc_ = camPosSrc.head(3);
    Vec3f camPosSrcToTgt_ = camPosSrcToTgt.head(3);

    Mat36f jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc_);
    Mat23f dProj = dCameraToScreen(camPosSrcToTgt_, colorFocal(0), colorFocal(1));
    Vec2f dColorB = intensityDerivTgt;
    jacBlockRow = jac.transpose() * dProj.transpose() * dColorB;
}

////////////////////////////////////////
// build jtj/jtr
////////////////////////////////////////
__inline__ __device__ void
addToLocalSystem(bool isValidCorr, float *d_JtJ, float *d_Jtr, float *d_J, unsigned int dim,
                 const Vec6f &jacobianBlockRow_i, const Vec6f &jacobianBlockRow_j,
                 unsigned int vi, unsigned int vj, float residual, float weight, unsigned int tidx) {
    //fill in bottom half (vi < vj) -> x < y
    for (unsigned int i = 0; i < 6; i++) {
        for (unsigned int j = i; j < 6; j++) {
            float dii = 0.0f;
            float djj = 0.0f;
            float dij = 0.0f;
            float dji = 0.0f;
            __shared__ float s_partJtJ[4];
            if (tidx == 0) {
                for (unsigned int k = 0; k < 4; k++)
                    s_partJtJ[k] = 0;
            } //TODO try with first 4 threads for all tidx == 0

            if (isValidCorr) {
                if (vi > 0) {
                    dii = jacobianBlockRow_i(i) * jacobianBlockRow_i(j) * weight;
                }
                if (vj > 0) {
                    djj = jacobianBlockRow_j(i) * jacobianBlockRow_j(j) * weight;
                }
                if (vi > 0 && vj > 0) {
                    dij = jacobianBlockRow_i(i) * jacobianBlockRow_j(j) * weight;
                    if (i != j) {
                        dji = jacobianBlockRow_i(j) * jacobianBlockRow_j(i) * weight;
                    }
                }
            }
            dii = warpReduce(dii);
            djj = warpReduce(djj);
            dij = warpReduce(dij);
            dji = warpReduce(dji);
            __syncthreads();
            if (tidx % WARP_SIZE == 0) {
                atomicAdd(&s_partJtJ[0], dii);
                atomicAdd(&s_partJtJ[1], djj);
                atomicAdd(&s_partJtJ[2], dij);
                atomicAdd(&s_partJtJ[3], dji);
            }
            __syncthreads();
            if (tidx == 0) {
                atomicAdd(&d_JtJ[(vi * 6 + j) * dim + (vi * 6 + i)], s_partJtJ[0]);
                atomicAdd(&d_JtJ[(vj * 6 + j) * dim + (vj * 6 + i)], s_partJtJ[1]);
                ////JitJj 和 JjtJi 互为转置，填充在一个矩阵块的上下两个半区
                atomicAdd(&d_JtJ[(vj * 6 + j) * dim + (vi * 6 + i)], s_partJtJ[2]);
                atomicAdd(&d_JtJ[(vj * 6 + i) * dim + (vi * 6 + j)], s_partJtJ[3]);
            }
        }
        float jtri = 0.0f;
        float jtrj = 0.0f;
        __shared__ float s_partJtr[2];
        if (tidx == 0) { for (unsigned int k = 0; k < 2; k++) s_partJtr[k] = 0; }
        if (isValidCorr) {
            if (vi > 0) jtri = jacobianBlockRow_i(i) * residual * weight;
            if (vj > 0) jtrj = jacobianBlockRow_j(i) * residual * weight;
        }
        jtri = warpReduce(jtri);
        jtrj = warpReduce(jtrj);
        __syncthreads();
        if (tidx % WARP_SIZE == 0) {
            atomicAdd(&s_partJtr[0], jtri);
            atomicAdd(&s_partJtr[1], jtrj);
        }
        __syncthreads();
        if (tidx == 0) {
            atomicAdd(&d_Jtr[vi * 6 + i], s_partJtr[0]);
            atomicAdd(&d_Jtr[vj * 6 + i], s_partJtr[1]);
        }

#ifdef DEBUG
        float Ji = 0.f;
        float Jj = 0.f;
        if (isValidCorr) {
            if (vi > 0)
                Ji = jacobianBlockRow_i(i);
            if (vj > 0)
                Jj = jacobianBlockRow_j(i);
        }
        Ji = warpReduce(Ji);
        Jj = warpReduce(Jj);
        __syncthreads();
        if (tidx % WARP_SIZE == 0) {
            atomicAdd(&d_J[vi * 6 + i], Ji);
            atomicAdd(&d_J[vj * 6 + i], Jj);
        }
#endif
    }
}

//寻找帧对，为了加速，只计算部分区域
__global__ void FindImageImageCorr_Kernel(SolverInput input, SolverState state, SolverParameters parameters) {
    // image indices
    unsigned int i, j; // project from j to i
    i = blockIdx.x;
    j = blockIdx.y; // all pairwise
    if (i >= j) return;

    const unsigned int tidx = threadIdx.x;
    const unsigned int subWidth = input.denseDepthWidth / parameters.denseOverlapCheckSubsampleFactor;
    const unsigned int x = (tidx % subWidth) * parameters.denseOverlapCheckSubsampleFactor;
    const unsigned int y = (tidx / subWidth) * parameters.denseOverlapCheckSubsampleFactor;
    const unsigned int idx = y * input.denseDepthWidth + x;

    if (idx < (input.denseDepthWidth * input.denseDepthHeight)) {
        Mat4f transform = state.d_xTransformInverses[i] * state.d_xTransforms[j];
        //if (!computeAngleDiff(transform, 1.0f)) return; //~60 degrees
        if (!computeAngleDiff(transform, 0.52f)) return; //TODO ~30 degrees

        // find correspondence
        __shared__ int foundCorr[1];
        foundCorr[0] = 0;
        __syncthreads();
        if (findDenseCorr(idx, input.denseDepthWidth, input.denseDepthHeight,
                          parameters.denseDistThresh, state.d_xTransformInverses[i],
                          state.d_xTransforms[j], input.intrinsics,
                          input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[j].d_depthDownsampled,
                          parameters.denseDepthMin, parameters.denseDepthMax,
                          parameters.boundingMin, parameters.boundingMax)) { //i tgt, j src
            atomicAdd(foundCorr, 1);
        } // found correspondence
        __syncthreads();
        if (tidx == 0) {
            if (foundCorr[0] > parameters.minNumOverlapCorr) {
                int addr = atomicAdd(state.d_numDenseOverlappingImages, 1);
                state.d_denseOverlappingImages[addr] = make_uint2(i, j);
            }
        }
    } // valid image pixel
}

__global__ void FindDenseCorrespondences_Kernel(SolverInput input, SolverState state, SolverParameters parameters) {
    const int imPairIdx = blockIdx.x; //should not go out of bounds, no need to check
    uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
    unsigned int i = imageIndices.x;
    unsigned int j = imageIndices.y;

    const unsigned int tidx = threadIdx.x;
    const unsigned int gidx = tidx * gridDim.y + blockIdx.y;

    if (gidx < (input.denseDepthWidth * input.denseDepthHeight)) {
        // find correspondence
        const int numWarps = THREADS_PER_BLOCK_DENSE_DEPTH / WARP_SIZE;
        __shared__ int s_count[numWarps];
        s_count[0] = 0;
        int count = 0;
        if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
                          parameters.denseDistThresh, parameters.denseNormalThresh, state.d_xTransformInverses[i],
                          state.d_xTransforms[j], input.intrinsics,
                          input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
                          input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
                          parameters.denseDepthMin, parameters.denseDepthMax,
                          parameters.boundingMin, parameters.boundingMax)) { //i tgt, j src
            count++;
        } // found correspondence
        count = warpReduce(count);
        __syncthreads();
        if (tidx % WARP_SIZE == 0) {
            s_count[tidx / WARP_SIZE] = count;
        }
        __syncthreads();
        for (unsigned int stride = numWarps / 2; stride > 0; stride /= 2) {
            if (tidx < stride) s_count[tidx] = s_count[tidx] + s_count[tidx + stride];
            __syncthreads();
        }
        if (tidx == 0) {
            atomicAdd(&state.d_denseCorrCounts[imPairIdx], s_count[0]);
        }
    } // valid image pixel
}

__global__ void WeightDenseCorrespondences_Kernel(unsigned int N, SolverState state, SolverParameters parameters) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // apply ln to weights
        float x = state.d_denseCorrCounts[idx];
        if (x > 0) {
            if (x < parameters.minNumDenseCorr) {//TODO change
                state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr
            } else {
                state.d_denseCorrCounts[idx] = 1.0f / fminf(logf(x), 9.0f); // natural log 
            }
        }
    }
}

template<bool useDepth, bool useColor>
__global__ void BuildDenseSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters) {
    const int imPairIdx = blockIdx.x;
    uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
    unsigned int i = imageIndices.x;
    unsigned int j = imageIndices.y;

    float imPairWeight = state.d_denseCorrCounts[imPairIdx];
    if (imPairWeight == 0.0f) return;

    const unsigned int idx = threadIdx.x;
    const unsigned int srcIdx = idx * gridDim.y + blockIdx.y;

    if (srcIdx < (input.denseDepthWidth * input.denseDepthHeight)) {
        Mat4f transform_i = state.d_xTransforms[i];
        Mat4f transform_j = state.d_xTransforms[j];
        Mat4f invTransform_i = state.d_xTransformInverses[i];
        Mat4f invTransform_j = state.d_xTransformInverses[j];

        // point-to-plane term
        Vec6f depthJacBlockRow_i, depthJacBlockRow_j;
        depthJacBlockRow_i.setZero();
        depthJacBlockRow_j.setZero();
        float depthRes = 0.0f;
        float depthWeight = 0.0f;
        // color term
        Vec6f colorJacBlockRow_i, colorJacBlockRow_j;
        colorJacBlockRow_i.setZero();
        colorJacBlockRow_j.setZero();
        float colorRes = 0.0f;
        float colorWeight = 0.0f;

        // find correspondence
        Vec4f camPosSrc;
        Vec4f camPosSrcToTgt;
        Vec4f camPosTgt;
        Vec4f normalTgt;
        Vec2f tgtScreenPos;

        bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
                                       parameters.denseDistThresh, parameters.denseNormalThresh,
                                       invTransform_i, transform_j, input.intrinsics,
                                       input.d_cacheFrames[i].d_cameraposDownsampled,
                                       input.d_cacheFrames[i].d_normalsDownsampled,
                                       input.d_cacheFrames[j].d_cameraposDownsampled,
                                       input.d_cacheFrames[j].d_normalsDownsampled,
                                       parameters.denseDepthMin, parameters.denseDepthMax,
                                       parameters.boundingMin, parameters.boundingMax,
                                       camPosSrc, camPosSrcToTgt,
                                       tgtScreenPos, camPosTgt, normalTgt); //i tgt, j src
        if (useDepth) {
            if (foundCorr) {
                // point-to-plane residual
                Vec4f diff = camPosTgt - camPosSrcToTgt;
                depthRes = diff.dot(normalTgt);
//                depthWeight = parameters.weightDenseDepth * imPairWeight;
                depthWeight = parameters.weightDenseDepth * imPairWeight *
                              (powf(fmaxf(0.0f, 1.0f - camPosTgt(2) / (2.0f * parameters.denseDepthMax)),
                                    2.5f)); //fr2_xyz_half
                if (i > 0)
                    computeJacobianBlockRow_i(depthJacBlockRow_i, transform_i, invTransform_j, camPosSrc, normalTgt);
                if (j > 0)
                    computeJacobianBlockRow_j(depthJacBlockRow_j, invTransform_i, transform_j, camPosSrc, normalTgt);
            }
#ifdef DEBUG
            float res = 0.f;
            int num = 0;
            if (foundCorr) {
                res = depthRes;
                num = 1;
            }
            res = warpReduce(res);
            num = warpReduce(num);
            __syncthreads();
            if (idx % WARP_SIZE == 0) {
                atomicAdd(&state.d_sumResidualDEBUG[imPairIdx], res);
                atomicAdd(&state.d_numCorrDEBUG[imPairIdx], num);
            }
#endif
            addToLocalSystem(foundCorr, state.d_denseJtJ, state.d_denseJtr, state.d_J, input.numberOfImages * 6,
                             depthJacBlockRow_i, depthJacBlockRow_j, i, j, depthRes, depthWeight, idx);
        }
        if (useColor) {
            bool foundCorrColor = false;
            if (foundCorr) {
                const Vec2f intensityDerivTgt = bilinearInterpolation(tgtScreenPos(0), tgtScreenPos(1),
                                                                      input.d_cacheFrames[i].d_intensityDerivsDownsampled,
                                                                      input.denseDepthWidth,
                                                                      input.denseDepthHeight);
                const float intensityTgt = bilinearInterpolation(tgtScreenPos(0), tgtScreenPos(1),
                                                                 input.d_cacheFrames[i].d_intensityDownsampled,
                                                                 input.denseDepthWidth, input.denseDepthHeight);
                colorRes = intensityTgt - input.d_cacheFrames[j].d_intensityDownsampled[srcIdx];
                foundCorrColor = (intensityTgt != NAN && intensityDerivTgt(0) != NAN &&
                                  abs(colorRes) < parameters.denseColorThresh &&
                                  intensityDerivTgt.norm() > parameters.denseColorGradientMin);
                if (foundCorrColor) {
                    const Vec2f focalLength(input.intrinsics(0), input.intrinsics(1));
                    if (i > 0)
                        computeJacobianBlockIntensityRow_i(colorJacBlockRow_i, focalLength, transform_i, invTransform_j,
                                                           camPosSrc, camPosSrcToTgt, intensityDerivTgt);
                    if (j > 0)
                        computeJacobianBlockIntensityRow_j(colorJacBlockRow_j, focalLength, invTransform_i, transform_j,
                                                           camPosSrc, camPosSrcToTgt, intensityDerivTgt);

                    colorWeight = parameters.weightDenseColor * imPairWeight *
                                  fmaxf(0.0f, 1.0f - abs(colorRes) / (1.15f * parameters.denseColorThresh));
                }
            }
#ifdef DEBUG
            float res_c = 0.f;
            int num_c = 0;
            if (foundCorrColor) {
                res_c = depthRes;
                num_c = 1;
            }
            res_c = warpReduce(res_c);
            num_c = warpReduce(num_c);
            __syncthreads();
            if (idx % WARP_SIZE == 0) {
                atomicAdd(&state.d_sumResidualDEBUG[imPairIdx], res_c);
                atomicAdd(&state.d_numCorrDEBUG[imPairIdx], num_c);
            }
#endif
            addToLocalSystem(foundCorrColor, state.d_denseJtJ, state.d_denseJtr, state.d_J, input.numberOfImages * 6,
                             colorJacBlockRow_i, colorJacBlockRow_j, i, j, colorRes, colorWeight, idx);
        }
    } // valid image pixel
}

__global__ void FlipJtJ_Kernel(unsigned int total, unsigned int dim, float *d_JtJ) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const unsigned int x = idx % dim;
        const unsigned int y = idx / dim;
        if (x > y) {
            d_JtJ[y * dim + x] = d_JtJ[x * dim + y];
        }
    }
}

bool BuildDenseSystem(const SolverInput &input, SolverState &state, SolverParameters &parameters) {
    const unsigned int N = input.numberOfImages;
    const int sizeJtr = 6 * N;
    const int sizeJtJ = sizeJtr * sizeJtr;

    const unsigned int maxDenseImPairs = input.numberOfImages * (input.numberOfImages - 1) / 2;
    CUDA_SAFE_CALL(cudaMemset(state.d_denseCorrCounts, 0, sizeof(float) * maxDenseImPairs));
    CUDA_SAFE_CALL(cudaMemset(state.d_denseJtJ, 0, sizeof(float) * sizeJtJ));
    CUDA_SAFE_CALL(cudaMemset(state.d_denseJtr, 0, sizeof(float) * sizeJtr));
    CUDA_SAFE_CALL(cudaMemset(state.d_numDenseOverlappingImages, 0, sizeof(int)));
#ifdef DEBUG
    CUDA_SAFE_CALL(cudaMemset(state.d_sumResidualDEBUG, 0, sizeof(float) * maxDenseImPairs));
    CUDA_SAFE_CALL(cudaMemset(state.d_numCorrDEBUG, 0, sizeof(int) * maxDenseImPairs));
    CUDA_SAFE_CALL(cudaMemset(state.d_J, 0, sizeof(float) * sizeJtr));
#endif

    dim3 gridImImOverlap(N, N, 1);

    FindImageImageCorr_Kernel << < gridImImOverlap, THREADS_PER_BLOCK_DENSE_OVERLAP >> > (input, state, parameters);

    int numOverlapImagePairs;
    CUDA_SAFE_CALL(cudaMemcpy(&numOverlapImagePairs, state.d_numDenseOverlappingImages,
                              sizeof(int), cudaMemcpyDeviceToHost));
    if (numOverlapImagePairs == 0) {
        printf("warning: no overlapping images for dense solve\n");
        return false;
    }

    const int reductionGlobal = (input.denseDepthWidth * input.denseDepthHeight + THREADS_PER_BLOCK_DENSE_DEPTH - 1) /
                                THREADS_PER_BLOCK_DENSE_DEPTH;
    dim3 grid(numOverlapImagePairs, reductionGlobal);

    FindDenseCorrespondences_Kernel << < grid, THREADS_PER_BLOCK_DENSE_DEPTH >> > (input, state, parameters);

    int wgrid = (numOverlapImagePairs + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
    WeightDenseCorrespondences_Kernel << < wgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >
                                                  (maxDenseImPairs, state, parameters);

    bool useDepth = parameters.weightDenseDepth > 0.0f;
    bool useColor = parameters.weightDenseColor > 0.0f;
    if (useDepth && useColor)
        BuildDenseSystem_Kernel<true, true> << < grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >
                                                       (input, state, parameters);
    else if (useDepth)
        BuildDenseSystem_Kernel<true, false> << < grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >
                                                        (input, state, parameters);
    else {
        printf("useDepth and useColor error!\n");
        return false;
    }

    const unsigned int flipgrid =
            (sizeJtJ + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
    FlipJtJ_Kernel << < flipgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> > (sizeJtJ, sizeJtr, state.d_denseJtJ);

#ifdef DEBUG
    uint2 *denseOverlappingImages = new uint2[numOverlapImagePairs];
    CUDA_SAFE_CALL(cudaMemcpy(denseOverlappingImages, state.d_denseOverlappingImages,
                              sizeof(uint2) * numOverlapImagePairs, cudaMemcpyDeviceToHost));
    float *denseCorrCounts = new float[numOverlapImagePairs];
    CUDA_SAFE_CALL(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts,
                              sizeof(float) * numOverlapImagePairs, cudaMemcpyDeviceToHost));
    float *sumResidualDEBUG = new float[numOverlapImagePairs];
    int *numCorrDEBUG = new int[numOverlapImagePairs];
    CUDA_SAFE_CALL(cudaMemcpy(sumResidualDEBUG, state.d_sumResidualDEBUG, sizeof(float) * numOverlapImagePairs,
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(numCorrDEBUG, state.d_numCorrDEBUG, sizeof(float) * numOverlapImagePairs,
                              cudaMemcpyDeviceToHost));
    float *J = new float[sizeJtr];
    CUDA_SAFE_CALL(cudaMemcpy(J, state.d_J, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));

    printf("image pair num: %d\n", numOverlapImagePairs);
    for (int i = 0; i < numOverlapImagePairs; ++i) {
        printf("image pair (%d, %d): %f %d %f\n", denseOverlappingImages[i].x, denseOverlappingImages[i].y,
               denseCorrCounts[i], numCorrDEBUG[i], sumResidualDEBUG[i]);
    }
    printf("J:\n");
    for (int i = 0; i < sizeJtr; ++i) {
        printf("%f ", J[i]);
    }
    printf("\n");

    delete[] denseOverlappingImages;
    delete[] denseCorrCounts;
    delete[] sumResidualDEBUG;
    delete[] numCorrDEBUG;
    delete[] J;
#endif

    return true;
}

float EvalGNConvergence(SolverInput &input, SolverState &state) {
    const unsigned int N = input.numberOfImages;
    Vec3f *deltaRot = new Vec3f[N];
    Vec3f *deltaTrans = new Vec3f[N];
    CUDA_SAFE_CALL(cudaMemcpy(deltaRot, state.d_deltaRot, sizeof(Vec3f) * N, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(deltaTrans, state.d_deltaTrans, sizeof(Vec3f) * N, cudaMemcpyDeviceToHost));
    float maxVal = 0.f;
    for (int i = 0; i < N; ++i) {
        float r1 = deltaRot[i].cwiseAbs().maxCoeff();
        float r2 = deltaTrans[i].cwiseAbs().maxCoeff();
        maxVal = fmaxf(maxVal, fmaxf(r1, r2));
    }
    delete[] deltaRot;
    delete[] deltaTrans;

    return maxVal;
}

void solve(SolverInput &input, SolverState &state, SolverParameters &parameters) {
    for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++) {
        parameters.weightDenseDepth = input.weightsDenseDepth[nIter];
        parameters.weightDenseColor = input.weightsDenseColor[nIter];
        convertLiePosesToMatrices(state.d_xRot, state.d_xTrans, input.numberOfImages, state.d_xTransforms,
                                  state.d_xTransformInverses);
        bool ok = BuildDenseSystem(input, state, parameters);
        if (!ok) {
            printf("solve failed!\n");
            break;
        }
#ifdef DEBUG
        float *denseJtJ = new float[36 * input.numberOfImages * input.numberOfImages];
        float *denseJtr = new float[6 * input.numberOfImages];
        cudaMemcpy(denseJtJ, state.d_denseJtJ, 36 * input.numberOfImages * input.numberOfImages,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(denseJtr, state.d_denseJtr, 6 * input.numberOfImages, cudaMemcpyDeviceToHost);

//        printf("denseJtJ:\n");
//        for (int k = 0; k < input.numberOfImages * 6; ++k) {
//            for (int m = 0; m < input.numberOfImages * 6; ++m) {
//                printf("%f ", denseJtJ[k * input.numberOfImages * 6 + m]);
//            }
//            printf("\n");
//        }
        printf("denseJtr:\n");
        for (int m = 0; m < input.numberOfImages * 6; ++m) {
            printf("%f ", denseJtr[m]);
        }
        printf("\n");
        delete[] denseJtJ;
        delete[] denseJtr;
#endif

        Initialization(input, state, parameters);

#ifdef DEBUG
        Vec3f *rRot = new Vec3f[input.numberOfImages];
        Vec3f *rTrans = new Vec3f[input.numberOfImages];
        Vec3f *zRot = new Vec3f[input.numberOfImages];
        Vec3f *zTrans = new Vec3f[input.numberOfImages];
        Vec3f *pRot = new Vec3f[input.numberOfImages];
        Vec3f *pTrans = new Vec3f[input.numberOfImages];
        cudaMemcpy(rRot, state.d_rRot, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        cudaMemcpy(rTrans, state.d_rTrans, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        cudaMemcpy(zRot, state.d_zRot, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        cudaMemcpy(zTrans, state.d_zTrans, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        cudaMemcpy(pRot, state.d_pRot, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        cudaMemcpy(pTrans, state.d_pTrans, sizeof(Vec3f) * input.numberOfImages, cudaMemcpyDeviceToHost);
        for (int k = 0; k < input.numberOfImages; ++k) {
            std::cout << rRot[k] << std::endl;
            std::cout << rTrans[k] << std::endl;
//            std::cout << zRot[k] << std::endl;
//            std::cout << zTrans[k] << std::endl;
//            std::cout << pRot[k] << std::endl;
//            std::cout << pTrans[k] << std::endl;
        }
        delete[] rRot;
        delete[] rTrans;
        delete[] zRot;
        delete[] zTrans;
        delete[] pRot;
        delete[] pTrans;
#endif

        for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
            if (PCGIteration(input, state, parameters, linIter == parameters.nLinIterations - 1)) {
                break;
            }
        }

        if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state) < 0.005f) {
            printf("EARLY OUT\n");
            break;
        }
    }

    convertLiePosesToMatrices(state.d_xRot, state.d_xTrans, input.numberOfImages, state.d_xTransforms,
                              state.d_xTransformInverses);
    cudaDeviceSynchronize();
}

__global__ void copyVec3ToVec4_kernel(Vec4f *dst, Vec3f *src, int num, float w) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num) {
        dst[id] = Vec4f(src[id](0), src[id](1), src[id](2), w);
    }
}

void copyVec3ToVec4(Vec4f *dst, Vec3f *src, int num, float w) {
    copyVec3ToVec4_kernel << < (num + 8 - 1) / 8, 8 >> > (dst, src, num, w);
}