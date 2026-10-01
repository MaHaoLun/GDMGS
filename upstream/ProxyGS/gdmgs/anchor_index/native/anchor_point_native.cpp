#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using I = int64_t;
using Box = std::array<double, 6>;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void require_array(
    const py::array& value,
    const py::dtype& dtype,
    int dimensions,
    const char* name
) {
    if (!value.dtype().is(dtype) || value.ndim() != dimensions ||
        !(value.flags() & py::array::c_style)) {
        throw std::invalid_argument(
            std::string(name) + " has the wrong dtype, rank, or contiguity"
        );
    }
}

static Box empty_box() {
    return {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
}

static void extend(Box& box, const float* point) {
    for (int axis = 0; axis < 3; ++axis) {
        const double value = static_cast<double>(point[axis]);
        box[axis] = std::min(box[axis], value);
        box[axis + 3] = std::max(box[axis + 3], value);
    }
}

static bool valid_box(const Box& box) {
    return box[0] <= box[3] && box[1] <= box[4] && box[2] <= box[5];
}

static void extend(Box& target, const Box& source) {
    if (!valid_box(source)) {
        return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        target[axis] = std::min(target[axis], source[axis]);
        target[axis + 3] = std::max(target[axis + 3], source[axis + 3]);
    }
}

static double surface_area(const Box& box) {
    if (!valid_box(box)) {
        return 0.0;
    }
    const double x = std::max(0.0, box[3] - box[0]);
    const double y = std::max(0.0, box[4] - box[1]);
    const double z = std::max(0.0, box[5] - box[2]);
    return 2.0 * (x * y + y * z + z * x);
}

struct Node {
    Box partition;
    Box bounds;
    I begin = 0;
    I end = 0;
    std::array<I, 8> children{};
};

struct Counters {
    I visited_nodes = 0;
    I certified_nodes = 0;
    I certified_anchors = 0;
    I empty_nodes = 0;
    I leaf_checks = 0;
    I anchor_fallback_checks = 0;
    I anchor_checks = 0;
    I culled_nonpositive_z = 0;
    I kept_out_of_image = 0;
    I kept_infinite_depth = 0;
    I kept_finite_depth = 0;
    I culled_finite_depth = 0;
    I certificate_near_or_w = 0;
    I certificate_partial_image = 0;
    I certificate_unknown_depth = 0;
    I certificate_depth_fail = 0;
    I certificate_attempts = 0;
    I certificate_small_node_skips = 0;
    I terminal_outside_keep_nodes = 0;
    I terminal_outside_keep_anchors = 0;
    I terminal_nonpositive_cull_nodes = 0;
    I terminal_nonpositive_cull_anchors = 0;
    I terminal_depth_keep_nodes = 0;
    I terminal_depth_keep_anchors = 0;
    I range_query_nodes = 0;
};

static constexpr I kCertificateMinimumCandidates = 256;

class DepthRangePyramid {
public:
    explicit DepthRangePyramid(const py::array& depth) {
        require_array(depth, py::dtype::of<float>(), 2, "depth");
        height_ = static_cast<I>(depth.shape(0));
        width_ = static_cast<I>(depth.shape(1));
        if (height_ < 1 || width_ < 1) {
            throw std::invalid_argument("depth must be nonempty");
        }
        const float* source = static_cast<const float*>(depth.data());
        maximum_levels_.push_back(std::vector<float>(source, source + height_ * width_));
        minimum_levels_.push_back(std::vector<float>(source, source + height_ * width_));
        heights_.push_back(height_);
        widths_.push_back(width_);
        while (heights_.back() > 1 || widths_.back() > 1) {
            const I previous_height = heights_.back();
            const I previous_width = widths_.back();
            const I next_height = (previous_height + 1) / 2;
            const I next_width = (previous_width + 1) / 2;
            const std::vector<float>& previous_maximum = maximum_levels_.back();
            const std::vector<float>& previous_minimum = minimum_levels_.back();
            std::vector<float> next_maximum(
                static_cast<size_t>(next_height * next_width),
                -std::numeric_limits<float>::infinity()
            );
            std::vector<float> next_minimum(
                static_cast<size_t>(next_height * next_width),
                std::numeric_limits<float>::infinity()
            );
            for (I row = 0; row < next_height; ++row) {
                for (I column = 0; column < next_width; ++column) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    float minimum = std::numeric_limits<float>::infinity();
                    for (I dy = 0; dy < 2; ++dy) {
                        const I child_row = row * 2 + dy;
                        if (child_row >= previous_height) {
                            continue;
                        }
                        for (I dx = 0; dx < 2; ++dx) {
                            const I child_column = column * 2 + dx;
                            if (child_column < previous_width) {
                                const size_t child_index = static_cast<size_t>(
                                    child_row * previous_width + child_column
                                );
                                maximum = std::max(maximum, previous_maximum[child_index]);
                                minimum = std::min(minimum, previous_minimum[child_index]);
                            }
                        }
                    }
                    const size_t next_index = static_cast<size_t>(row * next_width + column);
                    next_maximum[next_index] = maximum;
                    next_minimum[next_index] = minimum;
                }
            }
            maximum_levels_.push_back(std::move(next_maximum));
            minimum_levels_.push_back(std::move(next_minimum));
            heights_.push_back(next_height);
            widths_.push_back(next_width);
        }
    }

    float query_maximum(I row0, I column0, I row1, I column1, Counters& counters) const {
        return query(row0, column0, row1, column1, counters, true);
    }

    float query_minimum(I row0, I column0, I row1, I column1, Counters& counters) const {
        return query(row0, column0, row1, column1, counters, false);
    }

private:
    float query(
        I row0,
        I column0,
        I row1,
        I column1,
        Counters& counters,
        bool maximum_query
    ) const {
        if (row0 < 0 || column0 < 0 || row1 < row0 || column1 < column0 ||
            row1 >= height_ || column1 >= width_) {
            throw std::invalid_argument("depth range query is outside the image");
        }
        float result = maximum_query ? -std::numeric_limits<float>::infinity()
                                     : std::numeric_limits<float>::infinity();
        const I level = static_cast<I>(maximum_levels_.size()) - 1;
        for (I row = 0; row < heights_[static_cast<size_t>(level)]; ++row) {
            for (I column = 0; column < widths_[static_cast<size_t>(level)]; ++column) {
                const float value = query_cell(
                    level, row, column, row0, column0, row1, column1, counters,
                    maximum_query
                );
                result = maximum_query ? std::max(result, value) : std::min(result, value);
                if (maximum_query && std::isinf(result) && result > 0) {
                    return result;
                }
            }
        }
        return result;
    }

    float query_cell(
        I level,
        I row,
        I column,
        I query_row0,
        I query_column0,
        I query_row1,
        I query_column1,
        Counters& counters,
        bool maximum_query
    ) const {
        ++counters.range_query_nodes;
        const I scale = I(1) << level;
        const I row0 = row * scale;
        const I column0 = column * scale;
        const I row1 = std::min(height_ - 1, row0 + scale - 1);
        const I column1 = std::min(width_ - 1, column0 + scale - 1);
        if (row1 < query_row0 || query_row1 < row0 ||
            column1 < query_column0 || query_column1 < column0) {
            return maximum_query ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
        }
        if (query_row0 <= row0 && row1 <= query_row1 &&
            query_column0 <= column0 && column1 <= query_column1) {
            const auto& levels = maximum_query ? maximum_levels_ : minimum_levels_;
            return levels[static_cast<size_t>(level)][
                static_cast<size_t>(row * widths_[static_cast<size_t>(level)] + column)
            ];
        }
        if (level == 0) {
            const auto& levels = maximum_query ? maximum_levels_ : minimum_levels_;
            return levels[0][static_cast<size_t>(row * width_ + column)];
        }
        float result = maximum_query ? -std::numeric_limits<float>::infinity()
                                     : std::numeric_limits<float>::infinity();
        const I child_level = level - 1;
        for (I dy = 0; dy < 2; ++dy) {
            const I child_row = row * 2 + dy;
            if (child_row >= heights_[static_cast<size_t>(child_level)]) {
                continue;
            }
            for (I dx = 0; dx < 2; ++dx) {
                const I child_column = column * 2 + dx;
                if (child_column >= widths_[static_cast<size_t>(child_level)]) {
                    continue;
                }
                const float value = query_cell(
                        child_level,
                        child_row,
                        child_column,
                        query_row0,
                        query_column0,
                        query_row1,
                        query_column1,
                        counters,
                        maximum_query
                    );
                result = maximum_query ? std::max(result, value) : std::min(result, value);
                if (maximum_query && std::isinf(result) && result > 0) {
                    return result;
                }
            }
        }
        return result;
    }

    I height_ = 0;
    I width_ = 0;
    std::vector<std::vector<float>> maximum_levels_;
    std::vector<std::vector<float>> minimum_levels_;
    std::vector<I> heights_;
    std::vector<I> widths_;
};

// Formal G2-Index uses only maximum depth.  A minimum-depth hierarchy was
// measured as a negative ablation: it doubled build bandwidth while certifying
// too few additional Keep nodes.  Keep the exact min/max helper above for
// contract testing, but use this max-only layout on the timed path.
class DepthMaxPyramid {
public:
    explicit DepthMaxPyramid(const py::array& depth) {
        require_array(depth, py::dtype::of<float>(), 2, "depth");
        height_ = static_cast<I>(depth.shape(0));
        width_ = static_cast<I>(depth.shape(1));
        if (height_ < 1 || width_ < 1) {
            throw std::invalid_argument("depth must be nonempty");
        }
        const float* source = static_cast<const float*>(depth.data());
        levels_.push_back(std::vector<float>(source, source + height_ * width_));
        heights_.push_back(height_);
        widths_.push_back(width_);
        while (heights_.back() > 1 || widths_.back() > 1) {
            const I previous_height = heights_.back();
            const I previous_width = widths_.back();
            const I next_height = (previous_height + 1) / 2;
            const I next_width = (previous_width + 1) / 2;
            const auto& previous = levels_.back();
            std::vector<float> next(
                static_cast<size_t>(next_height * next_width),
                -std::numeric_limits<float>::infinity()
            );
            for (I row = 0; row < next_height; ++row) {
                for (I column = 0; column < next_width; ++column) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (I dy = 0; dy < 2; ++dy) {
                        const I child_row = row * 2 + dy;
                        if (child_row >= previous_height) {
                            continue;
                        }
                        for (I dx = 0; dx < 2; ++dx) {
                            const I child_column = column * 2 + dx;
                            if (child_column < previous_width) {
                                maximum = std::max(
                                    maximum,
                                    previous[static_cast<size_t>(
                                        child_row * previous_width + child_column
                                    )]
                                );
                            }
                        }
                    }
                    next[static_cast<size_t>(row * next_width + column)] = maximum;
                }
            }
            levels_.push_back(std::move(next));
            heights_.push_back(next_height);
            widths_.push_back(next_width);
        }
    }

    float query(I row0, I column0, I row1, I column1, Counters& counters) const {
        if (row0 < 0 || column0 < 0 || row1 < row0 || column1 < column0 ||
            row1 >= height_ || column1 >= width_) {
            throw std::invalid_argument("depth range query is outside the image");
        }
        const I level = static_cast<I>(levels_.size()) - 1;
        float maximum = -std::numeric_limits<float>::infinity();
        for (I row = 0; row < heights_[static_cast<size_t>(level)]; ++row) {
            for (I column = 0; column < widths_[static_cast<size_t>(level)]; ++column) {
                maximum = std::max(
                    maximum,
                    query_cell(
                        level, row, column, row0, column0, row1, column1, counters
                    )
                );
                if (std::isinf(maximum) && maximum > 0) {
                    return maximum;
                }
            }
        }
        return maximum;
    }

private:
    float query_cell(
        I level,
        I row,
        I column,
        I query_row0,
        I query_column0,
        I query_row1,
        I query_column1,
        Counters& counters
    ) const {
        ++counters.range_query_nodes;
        const I scale = I(1) << level;
        const I row0 = row * scale;
        const I column0 = column * scale;
        const I row1 = std::min(height_ - 1, row0 + scale - 1);
        const I column1 = std::min(width_ - 1, column0 + scale - 1);
        if (row1 < query_row0 || query_row1 < row0 ||
            column1 < query_column0 || query_column1 < column0) {
            return -std::numeric_limits<float>::infinity();
        }
        if (query_row0 <= row0 && row1 <= query_row1 &&
            query_column0 <= column0 && column1 <= query_column1) {
            return levels_[static_cast<size_t>(level)][
                static_cast<size_t>(row * widths_[static_cast<size_t>(level)] + column)
            ];
        }
        if (level == 0) {
            return levels_[0][static_cast<size_t>(row * width_ + column)];
        }
        const I child_level = level - 1;
        float maximum = -std::numeric_limits<float>::infinity();
        for (I dy = 0; dy < 2; ++dy) {
            const I child_row = row * 2 + dy;
            if (child_row >= heights_[static_cast<size_t>(child_level)]) {
                continue;
            }
            for (I dx = 0; dx < 2; ++dx) {
                const I child_column = column * 2 + dx;
                if (child_column >= widths_[static_cast<size_t>(child_level)]) {
                    continue;
                }
                maximum = std::max(
                    maximum,
                    query_cell(
                        child_level,
                        child_row,
                        child_column,
                        query_row0,
                        query_column0,
                        query_row1,
                        query_column1,
                        counters
                    )
                );
                if (std::isinf(maximum) && maximum > 0) {
                    return maximum;
                }
            }
        }
        return maximum;
    }

    I height_ = 0;
    I width_ = 0;
    std::vector<std::vector<float>> levels_;
    std::vector<I> heights_;
    std::vector<I> widths_;
};

struct QueryInputs {
    const float* depth;
    I height;
    I width;
    const float* world_view;
    const float* full_projection;
    float margin;
    float minimum_camera_z;
};

static float source_transform(const float* matrix, const float* point, int row) {
    float value = matrix[row] * point[0];
    value = value + matrix[4 + row] * point[1];
    value = value + matrix[8 + row] * point[2];
    value = value + matrix[12 + row];
    return value;
}

static bool point_is_culled(const float* point, const QueryInputs& inputs, Counters& counters) {
    ++counters.anchor_checks;
    const float camera_z = source_transform(inputs.world_view, point, 2);
    if (!(camera_z > inputs.minimum_camera_z)) {
        ++counters.culled_nonpositive_z;
        return true;
    }
    const float clip_x = source_transform(inputs.full_projection, point, 0);
    const float clip_y = source_transform(inputs.full_projection, point, 1);
    const float clip_w = source_transform(inputs.full_projection, point, 3);
    const float reciprocal_w = 1.0f / (clip_w + 1.0e-7f);
    const float projected_x = clip_x * reciprocal_w;
    const float projected_y = clip_y * reciprocal_w;
    float pixel_x = projected_x + 1.0f;
    pixel_x = pixel_x * static_cast<float>(inputs.width);
    pixel_x = pixel_x / 2.0f;
    float pixel_y = projected_y + 1.0f;
    pixel_y = pixel_y * static_cast<float>(inputs.height);
    pixel_y = pixel_y / 2.0f;
    if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y) ||
        pixel_x < static_cast<float>(std::numeric_limits<I>::min()) ||
        pixel_x > static_cast<float>(std::numeric_limits<I>::max()) ||
        pixel_y < static_cast<float>(std::numeric_limits<I>::min()) ||
        pixel_y > static_cast<float>(std::numeric_limits<I>::max())) {
        ++counters.kept_out_of_image;
        return false;
    }
    const I column = static_cast<I>(pixel_x);
    const I row = static_cast<I>(pixel_y);
    if (column < 0 || column >= inputs.width || row < 0 || row >= inputs.height) {
        ++counters.kept_out_of_image;
        return false;
    }
    const float depth = inputs.depth[row * inputs.width + column];
    if (!std::isfinite(depth)) {
        ++counters.kept_infinite_depth;
        return false;
    }
    if (camera_z > depth + inputs.margin) {
        ++counters.culled_finite_depth;
        return true;
    }
    ++counters.kept_finite_depth;
    return false;
}

static std::array<double, 2> affine_interval(
    const Box& box,
    const float* matrix,
    int row
) {
    double lower = static_cast<double>(matrix[12 + row]);
    double upper = lower;
    double magnitude = std::abs(lower);
    for (int axis = 0; axis < 3; ++axis) {
        const double coefficient = static_cast<double>(matrix[axis * 4 + row]);
        const double first = coefficient * box[axis];
        const double second = coefficient * box[axis + 3];
        lower += std::min(first, second);
        upper += std::max(first, second);
        magnitude += std::max(std::abs(first), std::abs(second));
    }
    const double error = 128.0 * std::numeric_limits<float>::epsilon() *
                         std::max(1.0, magnitude);
    return {
        std::nextafter(lower - error, -std::numeric_limits<double>::infinity()),
        std::nextafter(upper + error, std::numeric_limits<double>::infinity()),
    };
}

static std::array<double, 2> ratio_interval(
    const std::array<double, 2>& numerator,
    const std::array<double, 2>& denominator
) {
    const double values[4] = {
        numerator[0] / denominator[0],
        numerator[0] / denominator[1],
        numerator[1] / denominator[0],
        numerator[1] / denominator[1],
    };
    const auto minimum = std::min_element(values, values + 4);
    const auto maximum = std::max_element(values, values + 4);
    const double scale = std::max({1.0, std::abs(*minimum), std::abs(*maximum)});
    const double error = 128.0 * std::numeric_limits<float>::epsilon() * scale;
    return {
        std::nextafter(*minimum - error, -std::numeric_limits<double>::infinity()),
        std::nextafter(*maximum + error, std::numeric_limits<double>::infinity()),
    };
}

enum class NodeDecision {
    Descend,
    CullDepth,
    CullNonpositive,
    KeepOutside,
    KeepDepth,
};

static NodeDecision classify_node(
    const Box& box,
    const QueryInputs& inputs,
    Counters& counters
) {
    const auto camera_z = affine_interval(box, inputs.world_view, 2);
    if (!std::isfinite(camera_z[0]) || !std::isfinite(camera_z[1])) {
        ++counters.certificate_near_or_w;
        return NodeDecision::Descend;
    }
    if (camera_z[1] <= static_cast<double>(inputs.minimum_camera_z)) {
        return NodeDecision::CullNonpositive;
    }
    if (camera_z[0] <= static_cast<double>(inputs.minimum_camera_z)) {
        ++counters.certificate_near_or_w;
        return NodeDecision::Descend;
    }
    auto clip_w = affine_interval(box, inputs.full_projection, 3);
    clip_w[0] += 1.0e-7;
    clip_w[1] += 1.0e-7;
    if (!std::isfinite(clip_w[0]) || !std::isfinite(clip_w[1]) || clip_w[0] <= 0.0) {
        ++counters.certificate_near_or_w;
        return NodeDecision::Descend;
    }
    const auto projected_x = ratio_interval(
        affine_interval(box, inputs.full_projection, 0), clip_w
    );
    const auto projected_y = ratio_interval(
        affine_interval(box, inputs.full_projection, 1), clip_w
    );
    double pixel_x0 = (projected_x[0] + 1.0) * static_cast<double>(inputs.width) / 2.0;
    double pixel_x1 = (projected_x[1] + 1.0) * static_cast<double>(inputs.width) / 2.0;
    double pixel_y0 = (projected_y[0] + 1.0) * static_cast<double>(inputs.height) / 2.0;
    double pixel_y1 = (projected_y[1] + 1.0) * static_cast<double>(inputs.height) / 2.0;
    const double pixel_scale = std::max(
        {1.0, std::abs(pixel_x0), std::abs(pixel_x1), std::abs(pixel_y0), std::abs(pixel_y1)}
    );
    const double pixel_error = 128.0 * std::numeric_limits<float>::epsilon() * pixel_scale;
    pixel_x0 = std::nextafter(pixel_x0 - pixel_error, -std::numeric_limits<double>::infinity());
    pixel_x1 = std::nextafter(pixel_x1 + pixel_error, std::numeric_limits<double>::infinity());
    pixel_y0 = std::nextafter(pixel_y0 - pixel_error, -std::numeric_limits<double>::infinity());
    pixel_y1 = std::nextafter(pixel_y1 + pixel_error, std::numeric_limits<double>::infinity());
    if (pixel_x1 <= -1.0 || pixel_y1 <= -1.0 ||
        pixel_x0 >= static_cast<double>(inputs.width) ||
        pixel_y0 >= static_cast<double>(inputs.height)) {
        return NodeDecision::KeepOutside;
    }
    if (!(pixel_x0 >= 0.0 && pixel_y0 >= 0.0 &&
          pixel_x1 < static_cast<double>(inputs.width) &&
          pixel_y1 < static_cast<double>(inputs.height))) {
        ++counters.certificate_partial_image;
    } else {
        // Depth-dependent node certificates were a measured negative
        // ablation.  Exact depth occlusion remains in the fused pointwise
        // fallback; the hierarchy handles only spatially terminal cases.
        ++counters.certificate_depth_fail;
    }
    return NodeDecision::Descend;
}

class AnchorPointTree {
public:
    AnchorPointTree() = default;

    AnchorPointTree(
        const py::array& positions,
        I leaf_capacity,
        I max_depth,
        const std::string& build_method
    ) : leaf_capacity_(leaf_capacity), max_depth_(max_depth), build_method_(build_method) {
        require_array(positions, py::dtype::of<float>(), 2, "positions");
        if (positions.shape(1) != 3 || leaf_capacity_ < 1 ||
            max_depth_ < 1 || max_depth_ > 64) {
            throw std::invalid_argument("invalid positions or point-BVH settings");
        }
        if (build_method_ != "binned_sah") {
            throw std::invalid_argument("AnchorPointTree requires binned_sah build_method");
        }
        const I count = static_cast<I>(positions.shape(0));
        const float* source = static_cast<const float*>(positions.data());
        positions_.assign(source, source + count * 3);
        dfs_to_row_.resize(static_cast<size_t>(count));
        rank_of_row_.resize(static_cast<size_t>(count));
        std::iota(dfs_to_row_.begin(), dfs_to_row_.end(), I(0));
        for (I row = 0; row < count; ++row) {
            const float* point = &positions_[static_cast<size_t>(row * 3)];
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(point[axis])) {
                    throw std::invalid_argument("anchor positions must be finite");
                }
            }
        }
        if (count) {
            build_node(0, count, 0);
        }
        for (I rank = 0; rank < count; ++rank) {
            rank_of_row_[static_cast<size_t>(dfs_to_row_[static_cast<size_t>(rank)])] = rank;
        }
    }

    py::dict layout() const {
        const I count = static_cast<I>(dfs_to_row_.size());
        const I node_count = static_cast<I>(nodes_.size());
        py::array_t<float> positions({count, I(3)});
        py::array_t<I> dfs(count);
        py::array_t<I> rank(count);
        py::array_t<I> intervals({node_count, I(2)});
        py::array_t<I> children({node_count, I(8)});
        py::array_t<double> node_bounds({node_count, I(6)});
        py::array_t<double> partition_bounds({node_count, I(6)});
        std::copy(positions_.begin(), positions_.end(), positions.mutable_data());
        std::copy(dfs_to_row_.begin(), dfs_to_row_.end(), dfs.mutable_data());
        std::copy(rank_of_row_.begin(), rank_of_row_.end(), rank.mutable_data());
        for (I index = 0; index < node_count; ++index) {
            const Node& node = nodes_[static_cast<size_t>(index)];
            intervals.mutable_at(index, 0) = node.begin;
            intervals.mutable_at(index, 1) = node.end;
            for (int child = 0; child < 8; ++child) {
                children.mutable_at(index, child) = node.children[child];
            }
            for (int coordinate = 0; coordinate < 6; ++coordinate) {
                node_bounds.mutable_at(index, coordinate) = node.bounds[coordinate];
                partition_bounds.mutable_at(index, coordinate) = node.partition[coordinate];
            }
        }
        py::dict result;
        result["positions"] = positions;
        result["dfs_to_row"] = dfs;
        result["rank_of_row"] = rank;
        result["intervals"] = intervals;
        result["children"] = children;
        result["node_bounds"] = node_bounds;
        result["partition_bounds"] = partition_bounds;
        return result;
    }

    static AnchorPointTree from_layout(
        const py::dict& layout,
        I leaf_capacity,
        I max_depth,
        const std::string& build_method
    ) {
        AnchorPointTree tree;
        tree.leaf_capacity_ = leaf_capacity;
        tree.max_depth_ = max_depth;
        tree.build_method_ = build_method;
        if (leaf_capacity < 1 || max_depth < 1 || max_depth > 64 ||
            build_method != "binned_sah") {
            throw std::invalid_argument("invalid saved point-BVH settings");
        }
        auto array = [&](const char* name, const py::dtype& dtype, int dimensions) {
            py::array result = py::cast<py::array>(layout[name]);
            require_array(result, dtype, dimensions, name);
            return result;
        };
        const py::array positions = array("positions", py::dtype::of<float>(), 2);
        const py::array dfs = array("dfs_to_row", py::dtype::of<I>(), 1);
        const py::array rank = array("rank_of_row", py::dtype::of<I>(), 1);
        const py::array intervals = array("intervals", py::dtype::of<I>(), 2);
        const py::array children = array("children", py::dtype::of<I>(), 2);
        const py::array node_bounds = array("node_bounds", py::dtype::of<double>(), 2);
        const py::array partition_bounds = array("partition_bounds", py::dtype::of<double>(), 2);
        const I count = static_cast<I>(dfs.size());
        const I node_count = static_cast<I>(intervals.shape(0));
        if (positions.shape(0) != count || positions.shape(1) != 3 || rank.size() != count ||
            intervals.shape(1) != 2 || children.shape(0) != node_count || children.shape(1) != 8 ||
            node_bounds.shape(0) != node_count || node_bounds.shape(1) != 6 ||
            partition_bounds.shape(0) != node_count || partition_bounds.shape(1) != 6 ||
            ((count == 0) != (node_count == 0))) {
            throw std::invalid_argument("saved anchor point layout dimensions disagree");
        }
        const float* position_data = static_cast<const float*>(positions.data());
        const I* dfs_data = static_cast<const I*>(dfs.data());
        const I* rank_data = static_cast<const I*>(rank.data());
        tree.positions_.assign(position_data, position_data + count * 3);
        tree.dfs_to_row_.assign(dfs_data, dfs_data + count);
        tree.rank_of_row_.assign(rank_data, rank_data + count);
        tree.nodes_.resize(static_cast<size_t>(node_count));
        std::vector<unsigned char> seen(static_cast<size_t>(count), 0);
        for (I index = 0; index < count; ++index) {
            const I row = dfs_data[index];
            if (row < 0 || row >= count || seen[static_cast<size_t>(row)] || rank_data[row] != index) {
                throw std::invalid_argument("saved DFS/rank binding is not bijective");
            }
            seen[static_cast<size_t>(row)] = 1;
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(position_data[row * 3 + axis])) {
                    throw std::invalid_argument("saved positions are nonfinite");
                }
            }
        }
        const I* interval_data = static_cast<const I*>(intervals.data());
        const I* child_data = static_cast<const I*>(children.data());
        const double* bound_data = static_cast<const double*>(node_bounds.data());
        const double* partition_data = static_cast<const double*>(partition_bounds.data());
        std::vector<I> parent_counts(static_cast<size_t>(node_count), 0);
        for (I node_id = 0; node_id < node_count; ++node_id) {
            Node& node = tree.nodes_[static_cast<size_t>(node_id)];
            node.begin = interval_data[node_id * 2];
            node.end = interval_data[node_id * 2 + 1];
            if (node.begin < 0 || node.begin >= node.end || node.end > count) {
                throw std::invalid_argument("saved node interval is invalid");
            }
            for (int coordinate = 0; coordinate < 6; ++coordinate) {
                node.bounds[coordinate] = bound_data[node_id * 6 + coordinate];
                node.partition[coordinate] = partition_data[node_id * 6 + coordinate];
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(node.bounds[axis]) || !std::isfinite(node.bounds[axis + 3]) ||
                    node.bounds[axis] > node.bounds[axis + 3] ||
                    !std::isfinite(node.partition[axis]) ||
                    !std::isfinite(node.partition[axis + 3]) ||
                    node.partition[axis] > node.partition[axis + 3]) {
                    throw std::invalid_argument("saved node bounds are invalid");
                }
            }
            I cursor = node.begin;
            bool has_child = false;
            for (int child_slot = 0; child_slot < 8; ++child_slot) {
                const I child = child_data[node_id * 8 + child_slot];
                node.children[child_slot] = child;
                if (child < 0) {
                    continue;
                }
                has_child = true;
                if (child <= node_id || child >= node_count ||
                    ++parent_counts[static_cast<size_t>(child)] != 1 ||
                    interval_data[child * 2] != cursor) {
                    throw std::invalid_argument("saved child topology is invalid");
                }
                cursor = interval_data[child * 2 + 1];
            }
            if (has_child && cursor != node.end) {
                throw std::invalid_argument("saved child intervals do not partition the parent");
            }
            for (I item = node.begin; item < node.end; ++item) {
                const I row = dfs_data[item];
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = static_cast<double>(position_data[row * 3 + axis]);
                    if (value < node.bounds[axis] || value > node.bounds[axis + 3]) {
                        throw std::invalid_argument("saved node does not enclose descendants");
                    }
                }
            }
        }
        if (node_count) {
            if (tree.nodes_[0].begin != 0 || tree.nodes_[0].end != count) {
                throw std::invalid_argument("saved root does not cover every row");
            }
            for (I node_id = 1; node_id < node_count; ++node_id) {
                if (parent_counts[static_cast<size_t>(node_id)] != 1) {
                    throw std::invalid_argument("saved topology contains unreachable nodes");
                }
            }
        }
        return tree;
    }

    py::dict query(
        const py::array& candidate_ids,
        const py::array& depth,
        const py::array& world_view,
        const py::array& full_projection,
        const std::string& mode,
        float margin,
        float minimum_camera_z,
        bool trusted_buffers,
        bool reuse_buffers
    ) const {
        const auto total_start = Clock::now();
        require_array(candidate_ids, py::dtype::of<I>(), 1, "candidate_ids");
        require_array(depth, py::dtype::of<float>(), 2, "depth");
        require_array(world_view, py::dtype::of<float>(), 2, "world_view");
        require_array(full_projection, py::dtype::of<float>(), 2, "full_projection");
        if (world_view.shape(0) != 4 || world_view.shape(1) != 4 ||
            full_projection.shape(0) != 4 || full_projection.shape(1) != 4 ||
            depth.shape(0) < 1 || depth.shape(1) < 1 ||
            !std::isfinite(margin) || margin < 0.0f ||
            !std::isfinite(minimum_camera_z) || minimum_camera_z < 0.0f) {
            throw std::invalid_argument("invalid camera, depth, or predicate settings");
        }
        const float* depth_data = static_cast<const float*>(depth.data());
        if (!trusted_buffers) {
            for (I index = 0; index < static_cast<I>(depth.size()); ++index) {
                const float value = depth_data[index];
                if (std::isnan(value) || (std::isinf(value) && value < 0.0f) ||
                    (std::isfinite(value) && value <= 0.0f)) {
                    throw std::invalid_argument("depth must contain positive finite values or +infinity");
                }
            }
        }
        const float* view_data = static_cast<const float*>(world_view.data());
        const float* projection_data = static_cast<const float*>(full_projection.data());
        for (int index = 0; index < 16; ++index) {
            if (!std::isfinite(view_data[index]) || !std::isfinite(projection_data[index])) {
                throw std::invalid_argument("camera matrices must be finite");
            }
        }
        const I* ids = static_cast<const I*>(candidate_ids.data());
        const I candidate_count = static_cast<I>(candidate_ids.size());
        const I anchor_count = static_cast<I>(dfs_to_row_.size());
        if (!trusted_buffers) {
            for (I index = 0; index < candidate_count; ++index) {
                if (ids[index] < 0 || ids[index] >= anchor_count ||
                    (index && ids[index] <= ids[index - 1])) {
                    throw std::invalid_argument(
                        "candidate IDs must be valid sorted unique original rows"
                    );
                }
            }
        } else if (candidate_count && (ids[0] < 0 || ids[candidate_count - 1] >= anchor_count)) {
            throw std::invalid_argument("trusted candidate endpoint is out of bounds");
        }
        QueryInputs inputs{
            depth_data,
            static_cast<I>(depth.shape(0)),
            static_cast<I>(depth.shape(1)),
            view_data,
            projection_data,
            margin,
            minimum_camera_z,
        };
        Counters counters;
        const auto prepare_start = Clock::now();
        // 0=unresolved, 1=outside Keep, 2=depth Cull, 3=nonpositive Cull.
        // Tree traversal
        // writes only this bitmap; ordered output and unresolved point tests
        // are fused into one final source-order scan.
        std::vector<unsigned char> local_terminal_state;
        std::vector<unsigned char>& terminal_state_by_row = reuse_buffers
            ? terminal_state_scratch_ : local_terminal_state;
        if (mode == "tree") {
            terminal_state_by_row.assign(static_cast<size_t>(anchor_count), 0);
        }
        const double candidate_prepare_ms = elapsed_ms(prepare_start);
        double depth_build_ms = 0.0;
        const auto traversal_start = Clock::now();
        if (mode == "linear") {
            // The single ordered materialization pass below is the brute scan.
        } else if (mode == "tree") {
            if (anchor_count) {
                traverse(
                    0,
                    inputs,
                    terminal_state_by_row,
                    counters
                );
            }
        } else {
            throw std::invalid_argument("mode must be linear or tree");
        }
        const double traversal_ms = elapsed_ms(traversal_start) - depth_build_ms;
        const auto materialize_start = Clock::now();
        std::vector<I> local_selected;
        std::vector<I>& selected = reuse_buffers ? selected_scratch_ : local_selected;
        selected.clear();
        selected.reserve(static_cast<size_t>(candidate_count));
        std::vector<std::array<I, 2>> local_ranges;
        std::vector<std::array<I, 2>>& ranges = reuse_buffers ? ranges_scratch_ : local_ranges;
        ranges.clear();
        I range_begin = -1;
        for (I index = 0; index < candidate_count; ++index) {
            const I row = ids[index];
            const unsigned char state = mode == "tree"
                ? terminal_state_by_row[static_cast<size_t>(row)]
                : 0;
            bool keep = state == 1;
            if (state == 1) {
                ++counters.terminal_outside_keep_anchors;
            } else if (state == 2) {
                ++counters.certified_anchors;
            } else if (state == 3) {
                ++counters.certified_anchors;
                ++counters.terminal_nonpositive_cull_anchors;
            }
            if (state == 0) {
                if (mode == "tree") {
                    ++counters.anchor_fallback_checks;
                }
                keep = !point_is_culled(
                    &positions_[static_cast<size_t>(row * 3)], inputs, counters
                );
            }
            if (keep) {
                selected.push_back(row);
                if (range_begin < 0) {
                    range_begin = index;
                }
            } else if (range_begin >= 0) {
                ranges.push_back({range_begin, index});
                range_begin = -1;
            }
        }
        if (range_begin >= 0) {
            ranges.push_back({range_begin, candidate_count});
        }
        py::array_t<I> selected_array(static_cast<I>(selected.size()));
        py::array_t<I> raw_ranges({static_cast<I>(ranges.size()), I(2)});
        py::array_t<I> formal_ranges({static_cast<I>(ranges.size()), I(2)});
        std::copy(selected.begin(), selected.end(), selected_array.mutable_data());
        for (I index = 0; index < static_cast<I>(ranges.size()); ++index) {
            for (int endpoint = 0; endpoint < 2; ++endpoint) {
                raw_ranges.mutable_at(index, endpoint) = ranges[static_cast<size_t>(index)][endpoint];
                formal_ranges.mutable_at(index, endpoint) = ranges[static_cast<size_t>(index)][endpoint];
            }
        }
        const double materialization_ms = elapsed_ms(materialize_start);
        py::dict counter_values;
#define ADD_COUNTER(name) counter_values[#name] = counters.name
        ADD_COUNTER(visited_nodes);
        ADD_COUNTER(certified_nodes);
        ADD_COUNTER(certified_anchors);
        ADD_COUNTER(empty_nodes);
        ADD_COUNTER(leaf_checks);
        ADD_COUNTER(anchor_fallback_checks);
        ADD_COUNTER(anchor_checks);
        ADD_COUNTER(culled_nonpositive_z);
        ADD_COUNTER(kept_out_of_image);
        ADD_COUNTER(kept_infinite_depth);
        ADD_COUNTER(kept_finite_depth);
        ADD_COUNTER(culled_finite_depth);
        ADD_COUNTER(certificate_near_or_w);
        ADD_COUNTER(certificate_partial_image);
        ADD_COUNTER(certificate_unknown_depth);
        ADD_COUNTER(certificate_depth_fail);
        ADD_COUNTER(certificate_attempts);
        ADD_COUNTER(certificate_small_node_skips);
        ADD_COUNTER(terminal_outside_keep_nodes);
        ADD_COUNTER(terminal_outside_keep_anchors);
        ADD_COUNTER(terminal_nonpositive_cull_nodes);
        ADD_COUNTER(terminal_nonpositive_cull_anchors);
        ADD_COUNTER(terminal_depth_keep_nodes);
        ADD_COUNTER(terminal_depth_keep_anchors);
        ADD_COUNTER(range_query_nodes);
#undef ADD_COUNTER
        counter_values["candidate_count"] = candidate_count;
        counter_values["selected_count"] = static_cast<I>(selected.size());
        counter_values["raw_range_count"] = static_cast<I>(ranges.size());
        counter_values["formal_range_count"] = static_cast<I>(ranges.size());
        py::dict timings;
        timings["depth_range_max_build_ms"] = depth_build_ms;
        timings["candidate_rank_prepare_ms"] = candidate_prepare_ms;
        timings["anchor_tree_traversal_ms"] = traversal_ms;
        timings["anchor_tree_materialization_ms"] = materialization_ms;
        timings["anchor_index_total_ms"] = elapsed_ms(total_start);
        timings["trusted_buffers"] = trusted_buffers ? 1.0 : 0.0;
        timings["reused_buffers"] = reuse_buffers ? 1.0 : 0.0;
        py::dict result;
        result["selected_anchor_ids"] = selected_array;
        result["raw_ranges"] = raw_ranges;
        result["formal_ranges"] = formal_ranges;
        result["range_space"] = "candidate_source_ordinal_half_open";
        result["counters"] = counter_values;
        result["timings"] = timings;
        return result;
    }

private:
    I build_node(I begin, I end, I depth) {
        const I node_id = static_cast<I>(nodes_.size());
        Node node;
        node.bounds = empty_box();
        node.begin = begin;
        node.end = end;
        node.children.fill(-1);
        nodes_.push_back(node);
        for (I rank = begin; rank < end; ++rank) {
            const I row = dfs_to_row_[static_cast<size_t>(rank)];
            extend(nodes_[static_cast<size_t>(node_id)].bounds,
                   &positions_[static_cast<size_t>(row * 3)]);
        }
        nodes_[static_cast<size_t>(node_id)].partition =
            nodes_[static_cast<size_t>(node_id)].bounds;
        if (end - begin <= leaf_capacity_ || depth >= max_depth_) {
            outward(nodes_[static_cast<size_t>(node_id)].bounds);
            return node_id;
        }
        constexpr int bins = 16;
        const Box center_bounds = nodes_[static_cast<size_t>(node_id)].bounds;
        double best_cost = std::numeric_limits<double>::infinity();
        int best_axis = -1;
        int best_split = -1;
        for (int axis = 0; axis < 3; ++axis) {
            const double span = center_bounds[axis + 3] - center_bounds[axis];
            if (!(span > 0.0) || !std::isfinite(span)) {
                continue;
            }
            std::array<Box, bins> bin_bounds;
            std::array<I, bins> bin_counts{};
            for (Box& box : bin_bounds) {
                box = empty_box();
            }
            for (I rank = begin; rank < end; ++rank) {
                const I row = dfs_to_row_[static_cast<size_t>(rank)];
                const float* point = &positions_[static_cast<size_t>(row * 3)];
                const int bin = std::clamp(
                    static_cast<int>((static_cast<double>(point[axis]) - center_bounds[axis]) /
                                     span * bins),
                    0,
                    bins - 1
                );
                extend(bin_bounds[static_cast<size_t>(bin)], point);
                ++bin_counts[static_cast<size_t>(bin)];
            }
            std::array<Box, bins> left_bounds;
            std::array<Box, bins> right_bounds;
            std::array<I, bins> left_counts{};
            std::array<I, bins> right_counts{};
            Box left = empty_box();
            Box right = empty_box();
            I left_count = 0;
            I right_count = 0;
            for (int bin = 0; bin < bins; ++bin) {
                extend(left, bin_bounds[static_cast<size_t>(bin)]);
                left_count += bin_counts[static_cast<size_t>(bin)];
                left_bounds[static_cast<size_t>(bin)] = left;
                left_counts[static_cast<size_t>(bin)] = left_count;
                const int reverse = bins - 1 - bin;
                extend(right, bin_bounds[static_cast<size_t>(reverse)]);
                right_count += bin_counts[static_cast<size_t>(reverse)];
                right_bounds[static_cast<size_t>(reverse)] = right;
                right_counts[static_cast<size_t>(reverse)] = right_count;
            }
            for (int split = 0; split < bins - 1; ++split) {
                const I first_count = left_counts[static_cast<size_t>(split)];
                const I second_count = right_counts[static_cast<size_t>(split + 1)];
                if (!first_count || !second_count) {
                    continue;
                }
                const double cost =
                    surface_area(left_bounds[static_cast<size_t>(split)]) * first_count +
                    surface_area(right_bounds[static_cast<size_t>(split + 1)]) * second_count;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = split;
                }
            }
        }
        I middle = begin;
        bool partitioned = false;
        if (best_axis >= 0) {
            const double span = center_bounds[best_axis + 3] - center_bounds[best_axis];
            const auto boundary = std::partition(
                dfs_to_row_.begin() + begin,
                dfs_to_row_.begin() + end,
                [&](I row) {
                    const float value = positions_[static_cast<size_t>(row * 3 + best_axis)];
                    const int bin = std::clamp(
                        static_cast<int>((static_cast<double>(value) - center_bounds[best_axis]) /
                                         span * bins),
                        0,
                        bins - 1
                    );
                    return bin <= best_split;
                }
            );
            middle = static_cast<I>(boundary - dfs_to_row_.begin());
            partitioned = middle > begin && middle < end;
        }
        if (!partitioned) {
            int axis = 0;
            for (int candidate_axis = 1; candidate_axis < 3; ++candidate_axis) {
                if (center_bounds[candidate_axis + 3] - center_bounds[candidate_axis] >
                    center_bounds[axis + 3] - center_bounds[axis]) {
                    axis = candidate_axis;
                }
            }
            middle = (begin + end) / 2;
            std::nth_element(
                dfs_to_row_.begin() + begin,
                dfs_to_row_.begin() + middle,
                dfs_to_row_.begin() + end,
                [&](I first, I second) {
                    const float a = positions_[static_cast<size_t>(first * 3 + axis)];
                    const float b = positions_[static_cast<size_t>(second * 3 + axis)];
                    return a < b || (a == b && first < second);
                }
            );
        }
        if (middle <= begin || middle >= end) {
            outward(nodes_[static_cast<size_t>(node_id)].bounds);
            return node_id;
        }
        nodes_[static_cast<size_t>(node_id)].children[0] =
            build_node(begin, middle, depth + 1);
        nodes_[static_cast<size_t>(node_id)].children[1] =
            build_node(middle, end, depth + 1);
        outward(nodes_[static_cast<size_t>(node_id)].bounds);
        return node_id;
    }

    static void outward(Box& box) {
        for (int axis = 0; axis < 3; ++axis) {
            box[axis] = std::nextafter(box[axis], -std::numeric_limits<double>::infinity());
            box[axis + 3] = std::nextafter(
                box[axis + 3], std::numeric_limits<double>::infinity()
            );
        }
    }

    void traverse(
        I node_id,
        const QueryInputs& inputs,
        std::vector<unsigned char>& terminal_state_by_row,
        Counters& counters
    ) const {
        const Node& node = nodes_[static_cast<size_t>(node_id)];
        ++counters.visited_nodes;
        const I node_row_count = node.end - node.begin;
        NodeDecision decision = NodeDecision::Descend;
        if (node_row_count >= kCertificateMinimumCandidates) {
            ++counters.certificate_attempts;
            decision = classify_node(node.bounds, inputs, counters);
        } else {
            ++counters.certificate_small_node_skips;
        }
        if (decision == NodeDecision::KeepOutside || decision == NodeDecision::KeepDepth) {
            if (decision == NodeDecision::KeepOutside) {
                ++counters.terminal_outside_keep_nodes;
            } else {
                ++counters.terminal_depth_keep_nodes;
            }
            for (I rank = node.begin; rank < node.end; ++rank) {
                terminal_state_by_row[static_cast<size_t>(
                    dfs_to_row_[static_cast<size_t>(rank)]
                )] = 1;
            }
            return;
        }
        if (decision == NodeDecision::CullDepth ||
            decision == NodeDecision::CullNonpositive) {
            if (decision == NodeDecision::CullNonpositive) {
                ++counters.terminal_nonpositive_cull_nodes;
            }
            ++counters.certified_nodes;
            const unsigned char state = static_cast<unsigned char>(
                decision == NodeDecision::CullNonpositive ? 3 : 2
            );
            for (I rank = node.begin; rank < node.end; ++rank) {
                terminal_state_by_row[static_cast<size_t>(
                    dfs_to_row_[static_cast<size_t>(rank)]
                )] = state;
            }
            return;
        }
        bool leaf = true;
        for (I child : node.children) {
            if (child >= 0) {
                leaf = false;
                traverse(
                    child,
                    inputs,
                    terminal_state_by_row,
                    counters
                );
            }
        }
        if (!leaf) {
            return;
        }
        ++counters.leaf_checks;
        // Unresolved leaf candidates are evaluated once, in original source
        // order, by the fused materialization pass in query().
    }

    std::vector<float> positions_;
    std::vector<I> dfs_to_row_;
    std::vector<I> rank_of_row_;
    std::vector<Node> nodes_;
    mutable std::vector<unsigned char> terminal_state_scratch_;
    mutable std::vector<I> selected_scratch_;
    mutable std::vector<std::array<I, 2>> ranges_scratch_;
    I leaf_capacity_ = 0;
    I max_depth_ = 0;
    std::string build_method_ = "binned_sah";
};

PYBIND11_MODULE(ProxyGS_anchor_point_native, module) {
    module.attr("certificate_minimum_candidates") = kCertificateMinimumCandidates;
    module.attr("outside_image_terminal_keep") = true;
    module.attr("build_method") = "binned_sah";
    py::class_<AnchorPointTree>(module, "AnchorPointTree")
        .def(py::init<const py::array&, I, I, const std::string&>())
        .def("layout", &AnchorPointTree::layout)
        .def(
            "query",
            &AnchorPointTree::query,
            py::arg("candidate_ids"),
            py::arg("depth"),
            py::arg("world_view"),
            py::arg("full_projection"),
            py::arg("mode"),
            py::arg("margin"),
            py::arg("minimum_camera_z"),
            py::arg("trusted_buffers") = false,
            py::arg("reuse_buffers") = false
        )
        .def_static("from_layout", &AnchorPointTree::from_layout);
    module.def(
        "depth_range_max",
        [](const py::array& depth, I row0, I column0, I row1, I column1) {
            Counters counters;
            DepthRangePyramid pyramid(depth);
            py::dict result;
            result["maximum"] = pyramid.query_maximum(
                row0, column0, row1, column1, counters
            );
            result["minimum"] = pyramid.query_minimum(
                row0, column0, row1, column1, counters
            );
            result["visited_cells"] = counters.range_query_nodes;
            return result;
        }
    );
}
