#pragma once
#include <cfloat>
// Exact copy of the frozen GPU scan triangle predicate; no centroid culling.
__device__ bool triangle_relevant(const double* vertices, const int64_t* face,
                                 const double* planes, int plane_count) {
    for (int plane = 0; plane < plane_count; ++plane) {
        const double* p = planes + plane * 8;
        bool has_inside_vertex = false;
        for (int corner = 0; corner < 3; ++corner) {
            const double* vertex = vertices + face[corner] * 3;
            double value = p[3], magnitude = fabs(p[3]), uncertainty = p[7];
            for (int axis = 0; axis < 3; ++axis) {
                const double term = p[axis] * vertex[axis];
                value += term;
                magnitude += fabs(term);
                uncertainty += p[axis + 4] * fabs(vertex[axis]);
            }
            const double tolerance = 64 * DBL_EPSILON * (magnitude + 1) + uncertainty;
            // Finite inputs can overflow a dot product. Uncertain geometry
            // must remain a candidate rather than treating NaN as outside.
            if (!isfinite(value) || !isfinite(magnitude) || !isfinite(uncertainty)
                    || !isfinite(tolerance)) {
                has_inside_vertex = true;
                break;
            }
            if (value >= -tolerance) {
                has_inside_vertex = true;
                break;
            }
        }
        if (!has_inside_vertex) return false;
    }
    return true;
}

