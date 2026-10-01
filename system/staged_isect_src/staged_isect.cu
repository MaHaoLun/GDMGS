// Adapted from installed gsplat 1.4.0. See LICENSE and provenance.json.
#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename T>
__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

    OpT tile_radius = radius / static_cast<OpT>(tile_size);
    OpT tile_x = mean2d.x / static_cast<OpT>(tile_size);
    OpT tile_y = mean2d.y / static_cast<OpT>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}


void check_args(const torch::Tensor &means2d, const torch::Tensor &radii,
 const torch::Tensor &depths, uint32_t tile_size,uint32_t tile_width,uint32_t tile_height) {
 GSPLAT_CHECK_INPUT(means2d); GSPLAT_CHECK_INPUT(radii); GSPLAT_CHECK_INPUT(depths);
 TORCH_CHECK(means2d.scalar_type()==torch::kFloat32 && depths.scalar_type()==torch::kFloat32,
   "Only float32 production projection inputs are supported");
 TORCH_CHECK(radii.scalar_type()==torch::kInt32,"radii must be int32");
 TORCH_CHECK(means2d.dim()==3 && means2d.size(2)==2 && means2d.size(0)>0,"requires [C,N,2], C>0");
 TORCH_CHECK(radii.sizes()==depths.sizes() && radii.dim()==2 && radii.size(0)==means2d.size(0) && radii.size(1)==means2d.size(1),"shape mismatch");
 TORCH_CHECK(means2d.device()==radii.device() && means2d.device()==depths.device(),"device mismatch");
 TORCH_CHECK(tile_size>0 && tile_width>0 && tile_height>0,"tile dimensions must be positive");
 TORCH_CHECK(radii.numel() <= INT32_MAX && uint64_t(tile_width)*tile_height<=INT32_MAX,"index range overflow");
 TORCH_CHECK(uint64_t(tile_width)*tile_height*radii.numel() <= INT64_MAX,"count overflow");
 TORCH_CHECK(uint32_t(floor(log2(uint64_t(tile_width)*tile_height)))+1+uint32_t(floor(log2(means2d.size(0))))+1<=32,"key encoding overflow");
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> staged_count(
 const torch::Tensor &means2d,const torch::Tensor &radii,const torch::Tensor &depths,
 uint32_t tile_size,uint32_t tile_width,uint32_t tile_height) {
 GSPLAT_DEVICE_GUARD(means2d);
 check_args(means2d,radii,depths,tile_size,tile_width,tile_height);
 uint32_t C=means2d.size(0), N=means2d.size(1), total_elems=C*N;
 uint32_t tile_n_bits=uint32_t(floor(log2(tile_width*tile_height)))+1;
 auto stream=at::cuda::getCurrentCUDAStream();
 auto tiles=torch::empty_like(radii);
 torch::Tensor prefix,total;
 if(total_elems) {
   isect_tiles<float><<<(total_elems+GSPLAT_N_THREADS-1)/GSPLAT_N_THREADS,GSPLAT_N_THREADS,0,stream>>>(
     false,C,N,0,nullptr,nullptr,means2d.data_ptr<float>(),radii.data_ptr<int32_t>(),depths.data_ptr<float>(),
     nullptr,tile_size,tile_width,tile_height,tile_n_bits,tiles.data_ptr<int32_t>(),nullptr,nullptr);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
   prefix=torch::cumsum(tiles.view({-1}),0);
   total=prefix.select(0,total_elems-1);
 } else {
   prefix=torch::empty({0},depths.options().dtype(torch::kInt64));
   total=torch::zeros({},depths.options().dtype(torch::kInt64));
 }
 return std::make_tuple(tiles,prefix,total);
}
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> staged_finish(
 const torch::Tensor &means2d,const torch::Tensor &radii,const torch::Tensor &depths,
 const torch::Tensor &tiles_per_gauss,const torch::Tensor &cum_tiles_per_gauss,
 int64_t n_isects,uint32_t tile_size,uint32_t tile_width,uint32_t tile_height,
 bool sort,bool double_buffer) {
 GSPLAT_DEVICE_GUARD(means2d);
 check_args(means2d,radii,depths,tile_size,tile_width,tile_height);
 GSPLAT_CHECK_INPUT(tiles_per_gauss); GSPLAT_CHECK_INPUT(cum_tiles_per_gauss);
 TORCH_CHECK(tiles_per_gauss.device()==depths.device() && cum_tiles_per_gauss.device()==depths.device(),"device mismatch");
 TORCH_CHECK(tiles_per_gauss.scalar_type()==torch::kInt32 && tiles_per_gauss.sizes()==radii.sizes(),"tiles mismatch");
 TORCH_CHECK(cum_tiles_per_gauss.scalar_type()==torch::kInt64 && cum_tiles_per_gauss.numel()==radii.numel(),"prefix mismatch");
 TORCH_CHECK(n_isects>=0 && n_isects<=INT32_MAX,"intersection count exceeds int32 downstream range");
 uint32_t C=means2d.size(0), N=means2d.size(1), total_elems=C*N;
 TORCH_CHECK(total_elems || n_isects==0,"nonzero count for empty inputs");
 bool packed=false; uint32_t nnz=0; int64_t *camera_ids_ptr=nullptr,*gaussian_ids_ptr=nullptr;
 uint32_t tile_n_bits=uint32_t(floor(log2(tile_width*tile_height)))+1;
 uint32_t cam_n_bits=uint32_t(floor(log2(C)))+1;
 auto stream=at::cuda::getCurrentCUDAStream();
    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_n_isects",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>()
                );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

} // namespace gsplat
