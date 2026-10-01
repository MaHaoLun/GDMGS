#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

namespace {
constexpr int kCapacity = 32;

int one_box(const double* box, const double* eye, double* output) {
    constexpr double tolerance = 1e-12;
    double rays[8][3];
    for (int corner = 0; corner < 8; ++corner) {
        for (int axis = 0; axis < 3; ++axis) {
            const int bit = (corner >> (2-axis)) & 1;
            rays[corner][axis] = box[axis + (bit ? 3 : 0)] - eye[axis];
        }
    }
    int count = 0;
    for (int first = 0; first < 8; ++first) {
        for (int second = first+1; second < 8; ++second) {
            double normal[3] = {
                rays[first][1]*rays[second][2] - rays[first][2]*rays[second][1],
                rays[first][2]*rays[second][0] - rays[first][0]*rays[second][2],
                rays[first][0]*rays[second][1] - rays[first][1]*rays[second][0]
            };
            const double length = std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            if (!(length > tolerance)) continue;
            for (double& component : normal) component /= length;
            double minimum = INFINITY, maximum = -INFINITY;
            for (const auto& ray : rays) {
                const double side = normal[0]*ray[0]+normal[1]*ray[1]+normal[2]*ray[2];
                minimum = std::min(minimum, side);
                maximum = std::max(maximum, side);
            }
            if (!(minimum >= -tolerance || maximum <= tolerance)) continue;
            if (maximum <= tolerance)
                for (double& component : normal) component = -component;
            bool duplicate = false;
            for (int previous = 0; previous < count; ++previous) {
                const double* old = output+4*previous;
                if (std::fabs(normal[0]-old[0]) <= 1e-10 &&
                    std::fabs(normal[1]-old[1]) <= 1e-10 &&
                    std::fabs(normal[2]-old[2]) <= 1e-10) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            if (count >= kCapacity) return -1;
            double* plane = output+4*count++;
            plane[0] = normal[0]; plane[1] = normal[1]; plane[2] = normal[2];
            plane[3] = -(normal[0]*eye[0]+normal[1]*eye[1]+normal[2]*eye[2]);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (eye[axis] < box[axis]) {
            if (count >= kCapacity) return -1;
            double* plane = output+4*count++;
            std::fill(plane, plane+4, 0.0);
            plane[axis] = 1.0;
            plane[3] = -box[axis];
        } else if (eye[axis] > box[axis+3]) {
            if (count >= kCapacity) return -1;
            double* plane = output+4*count++;
            std::fill(plane, plane+4, 0.0);
            plane[axis] = -1.0;
            plane[3] = box[axis+3];
        }
    }
    return count;
}
}

extern "C" int cpu_build_holes(const double* boxes, const double* eye,
                                 int holes, int threads,
                                 double* planes, int32_t* starts) {
    if (!boxes || !eye || !planes || !starts || holes < 0 || threads < 1) return -1;
    std::vector<double> scratch(size_t(holes)*kCapacity*4);
    std::vector<int> counts(static_cast<size_t>(holes));
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int hole = 0; hole < holes; ++hole)
        counts[size_t(hole)] = one_box(boxes+6*hole, eye, scratch.data()+size_t(hole)*kCapacity*4);
    starts[0] = 0;
    for (int hole = 0; hole < holes; ++hole) {
        if (counts[size_t(hole)] < 0) return -2;
        starts[hole+1] = starts[hole] + counts[size_t(hole)];
        std::memcpy(planes+size_t(starts[hole])*4,
                    scratch.data()+size_t(hole)*kCapacity*4,
                    size_t(counts[size_t(hole)])*4*sizeof(double));
    }
    return 0;
}
