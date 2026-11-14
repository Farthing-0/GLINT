// GLINT transit kernels. Host code owns allocation, validation, and device selection.
#include <math.h>
#include <limits.h>

__device__ int lower_bound_time(const double* time, int size, double value) {
    int left = 0, right = size;
    while (left < right) {
        int middle = left + (right - left) / 2;
        if (time[middle] < value) left = middle + 1;
        else right = middle;
    }
    return left;
}

__device__ double shape_at(const float* shape, int size, double phase) {
    if (phase < -0.5 || phase > 0.5) return 0.0;
    double position = (phase + 0.5) * (size - 1);
    int index = min((int)position, size - 2);
    double fraction = position - index;
    return shape[index] + fraction * (shape[index + 1] - shape[index]);
}

__device__ bool better(double score, int index, double best, int best_index) {
    return isfinite(score) && (score > best || (score == best && index < best_index));
}

__device__ void reduce_best(double* scores, int* indices) {
    for (int width = blockDim.x / 2; width; width /= 2) {
        if (threadIdx.x < width) {
            int other = threadIdx.x + width;
            if (better(scores[other], indices[other], scores[threadIdx.x], indices[threadIdx.x])) {
                scores[threadIdx.x] = scores[other];
                indices[threadIdx.x] = indices[other];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void linear_search(
    const double* time, const float* residual, const double* weight, int ndata,
    const float* shape, int nshape, const double* durations,
    double origin, double step, int nepoch, int min_points,
    float* likelihood, float* depth, float* variance
) {
    extern __shared__ double storage[];
    double* cross = storage;
    double* information = cross + blockDim.x;
    int* counts = (int*)(information + blockDim.x);
    double duration = durations[blockIdx.y];

    for (long long epoch = blockIdx.x; epoch < nepoch; epoch += gridDim.x) {
        double center = origin + epoch * step;
        int first = lower_bound_time(time, ndata, center - 0.5 * duration);
        double a = 0.0, b = 0.0;
        int count = 0;
        for (int i = first + threadIdx.x; i < ndata; i += blockDim.x) {
            if (time[i] > center + 0.5 * duration) break;
            if (weight[i] == 0.0) continue;
            double model = shape_at(shape, nshape, (time[i] - center) / duration);
            if (model <= 0.0) continue;
            a += weight[i] * model * residual[i];
            b += weight[i] * model * model;
            ++count;
        }
        cross[threadIdx.x] = a;
        information[threadIdx.x] = b;
        counts[threadIdx.x] = count;
        __syncthreads();
        for (int width = blockDim.x / 2; width; width /= 2) {
            if (threadIdx.x < width) {
                cross[threadIdx.x] += cross[threadIdx.x + width];
                information[threadIdx.x] += information[threadIdx.x + width];
                counts[threadIdx.x] += counts[threadIdx.x + width];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            long long index = (long long)blockIdx.y * nepoch + epoch;
            float d = (float)(cross[0] / information[0]);
            float v = (float)(1.0 / information[0]);
            float score = (float)(0.5 * cross[0] * cross[0] / information[0]);
            bool valid = counts[0] >= min_points && information[0] > 0.0
                && isfinite(d) && isfinite(v) && v > 0.0f && isfinite(score);
            likelihood[index] = valid ? score : nanf("");
            depth[index] = valid ? d : nanf("");
            variance[index] = valid ? v : INFINITY;
        }
        __syncthreads();
    }
}

extern "C" __global__ void periodic_search(
    const float* likelihood, const float* depth, const float* variance,
    const double* period_steps, const int* first_duration, const int* last_duration,
    int nepoch, int nduration, int ntiles, int min_transits, double depth_min, int two_pass,
    double* tile_score, double* tile_depth, double* tile_variance,
    int* tile_epoch, int* tile_count
) {
    int period_index = blockIdx.y, duration = blockIdx.z;
    int epoch = blockIdx.x * blockDim.x + threadIdx.x;
    double spacing = period_steps[period_index];
    double score = -INFINITY, joint_depth = NAN, joint_variance = NAN;
    int count = 0;
    if (duration >= first_duration[period_index] && duration <= last_duration[period_index]
        && epoch < nepoch && epoch < spacing) {
        double a = 0.0, b = 0.0, correction = 0.0;
        for (int event = 0; ; ++event) {
            // Round each event displacement independently, with ties to even.
            double displacement = nearbyint(spacing * event);
            if (displacement >= nepoch - epoch) break;
            long long index = (long long)duration * nepoch + epoch + (int)displacement;
            double v = variance[index], d = depth[index], local_score = likelihood[index];
            if (!(v > 0.0) || !isfinite(v) || !isfinite(d) || !isfinite(local_score)) continue;
            double w = 1.0 / v;
            a += d * w;
            b += w;
            correction += local_score - 0.5 * d * d * w;
            ++count;
        }
        if (count >= min_transits && b > 0.0) {
            joint_depth = a / b;
            joint_variance = 1.0 / b;
            if (joint_depth > depth_min) {
                score = correction + 0.5 * a * joint_depth;
                if (two_pass) {
                    score = 0.0;
                    for (int event = 0; ; ++event) {
                        double displacement = nearbyint(spacing * event);
                        if (displacement >= nepoch - epoch) break;
                        long long index = (long long)duration * nepoch + epoch + (int)displacement;
                        double v = variance[index], d = depth[index], local_score = likelihood[index];
                        if (!(v > 0.0) || !isfinite(v) || !isfinite(d) || !isfinite(local_score)) continue;
                        double delta = d - joint_depth;
                        score += local_score - 0.5 * delta * delta / v;
                    }
                }
            }
        }
    }
    extern __shared__ double storage[];
    double* scores = storage;
    int* indices = (int*)(scores + blockDim.x);
    scores[threadIdx.x] = score;
    indices[threadIdx.x] = epoch;
    __syncthreads();
    reduce_best(scores, indices);
    if (epoch == indices[0]) {
        long long index = ((long long)period_index * nduration + duration) * ntiles + blockIdx.x;
        bool valid = isfinite(scores[0]);
        tile_score[index] = valid ? scores[0] : NAN;
        tile_depth[index] = valid ? joint_depth : NAN;
        tile_variance[index] = valid ? joint_variance : NAN;
        tile_epoch[index] = valid ? epoch : -1;
        tile_count[index] = valid ? count : 0;
    }
}

extern "C" __global__ void reduce_periods(
    const double* tile_score, const double* tile_depth, const double* tile_variance,
    const int* tile_epoch, const int* tile_count, int nduration, int ntiles,
    double* score, double* depth, double* variance, int* epoch, int* duration, int* count
) {
    int size = nduration * ntiles;
    long long start = (long long)blockIdx.x * size;
    double best = -INFINITY;
    int best_index = INT_MAX;
    for (int index = threadIdx.x; index < size; index += blockDim.x) {
        if (better(tile_score[start + index], index, best, best_index)) {
            best = tile_score[start + index];
            best_index = index;
        }
    }
    extern __shared__ double storage[];
    double* scores = storage;
    int* indices = (int*)(scores + blockDim.x);
    scores[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    reduce_best(scores, indices);
    if (threadIdx.x == 0) {
        int output = blockIdx.x, index = indices[0];
        bool valid = index != INT_MAX;
        score[output] = valid ? scores[0] : NAN;
        depth[output] = valid ? tile_depth[start + index] : NAN;
        variance[output] = valid ? tile_variance[start + index] : NAN;
        epoch[output] = valid ? tile_epoch[start + index] : -1;
        duration[output] = valid ? index / ntiles : -1;
        count[output] = valid ? tile_count[start + index] : 0;
    }
}
