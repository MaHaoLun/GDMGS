#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

// Same conservative support test as anchor_frustum/native/bounds.h.
static inline bool outside(const double* b, const double* c) {
    if (!(b[6] >= 0) || !std::isfinite(b[6])) return false;
    double lo[3], hi[3];
    for (int r = 0; r < 3; ++r) {
        lo[r] = hi[r] = c[4*r+3];
        double magnitude = std::fabs(c[4*r+3]);
        for (int k = 0; k < 3; ++k) {
            double x = c[4*r+k]*b[k], y = c[4*r+k]*b[k+3];
            lo[r] += std::fmin(x,y); hi[r] += std::fmax(x,y);
            magnitude += std::fmax(std::fabs(x),std::fabs(y));
        }
        double error = 2e-5*(1+magnitude);
        lo[r] -= error; hi[r] += error;
        if (!std::isfinite(lo[r]) || !std::isfinite(hi[r])) return false;
    }
    if (hi[2] < 0.01 || lo[2] > 1e10) return true;
    if (lo[2] <= 0.01) return false;
    double xlo=1e300,xhi=-1e300,ylo=1e300,yhi=-1e300;
    for (int i=0;i<2;++i) for (int j=0;j<2;++j) {
        double z=j?hi[2]:lo[2];
        double x=(i?hi[0]:lo[0])/z, y=(i?hi[1]:lo[1])/z;
        xlo=std::fmin(xlo,x);xhi=std::fmax(xhi,x);
        ylo=std::fmin(ylo,y);yhi=std::fmax(yhi,y);
    }
    double tx=std::fmax(c[18],c[20]-c[18])/c[16]+0.15*c[20]/c[16];
    double ty=std::fmax(c[19],c[21]-c[19])/c[17]+0.15*c[21]/c[17];
    double trace=b[6]*b[6]*c[22]*(c[16]*c[16]*(1+tx*tx)+c[17]*c[17]*(1+ty*ty))/(lo[2]*lo[2]);
    double radius=std::ceil(3*std::sqrt(trace*1.001+0.401))+2;
    if (!std::isfinite(radius)) return false;
    return c[16]*xhi+c[18]+radius<0 || c[16]*xlo+c[18]-radius>c[20]
        || c[17]*yhi+c[19]+radius<0 || c[17]*ylo+c[19]-radius>c[21];
}

static inline bool inside_hole(const double* b, const double* planes, int begin, int end) {
    for (int j=begin;j<end;++j) {
        const double* p=planes+4*j;
        double minimum=p[3];
        if (!(b[6] >= 0) || !std::isfinite(b[6])) return false;
        // Include finite 3-sigma support, not only the candidate centers.
        for (int k=0;k<3;++k) {
            minimum += p[k]*(p[k]>=0 ? b[k] : b[k+3]);
            minimum -= 3.0*b[6]*std::fabs(p[k]);
        }
        if (!(minimum>1e-8)) return false;
    }
    return true;
}

extern "C" int cpu_select(const double* bounds, const double* camera,
                           const uint8_t* candidates, int64_t n,
                           const double* planes, const int32_t* starts,
                           int holes, int threads, uint8_t* output) {
    if (!bounds || !camera || !candidates || !planes || !starts || !output ||
        n<0 || holes<0 || threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0;i<n;++i) {
        bool keep=candidates[i] && !outside(bounds+7*i,camera);
        if (keep) for (int h=0;h<holes;++h) {
            if (inside_hole(bounds+7*i,planes,starts[h],starts[h+1])) {keep=false;break;}
        }
        output[i]=keep ? 1 : 0;
    }
    return 0;
}

extern "C" int cpu_select_tree(const double* bounds, const double* nodes,
                                const int32_t* left, const int32_t* right,
                                const int64_t* order, const double* camera,
                                const uint8_t* candidates, int64_t n, int64_t leaves,
                                const double* planes, const int32_t* starts,
                                int holes, int threads, uint8_t* output) {
    if (!bounds || !nodes || !left || !right || !order || !camera || !candidates ||
        !planes || !starts || !output || n<0 || leaves<1 || holes<0 || threads<1) return -1;
    std::memset(output, 0, size_t(n));
    int64_t internal = leaves-1;
    auto reject = [&](int node) {
        const double* box=nodes+7*node;
        if (outside(box,camera)) return true;
        for (int h=0;h<holes;++h)
            if (inside_hole(box,planes,starts[h],starts[h+1])) return true;
        return false;
    };
    // Split the read-only tree into enough independent subtrees to balance
    // the OpenMP workers. Each subtree owns disjoint original-row outputs.
    std::vector<int> frontier{0};
    while (int(frontier.size()) < threads*8) {
        std::vector<int> next;
        next.reserve(frontier.size()*2);
        bool expanded=false;
        for (int node:frontier) {
            if (reject(node)) continue;
            if (node<internal) {
                next.push_back(left[node]);next.push_back(right[node]);
                expanded=true;
            } else next.push_back(node);
        }
        frontier.swap(next);
        if (!expanded || frontier.empty()) break;
    }
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for (int task=0;task<int(frontier.size());++task) {
        std::vector<int> stack{frontier[task]};
        while (!stack.empty()) {
            int node=stack.back();stack.pop_back();
            if (reject(node)) continue;
            if (node<internal) {
                stack.push_back(left[node]);stack.push_back(right[node]);
                continue;
            }
            int64_t leaf=node-internal;
            int64_t end=std::min<int64_t>(n,(leaf+1)*32);
            for (int64_t j=leaf*32;j<end;++j) {
                int64_t id=order[j];
                if (!candidates[id] || outside(bounds+7*id,camera)) continue;
                bool hidden=false;
                for (int h=0;h<holes;++h)
                    if (inside_hole(bounds+7*id,planes,starts[h],starts[h+1])) {hidden=true;break;}
                if (!hidden) output[id]=1;
            }
        }
    }
    return 0;
}

extern "C" int cpu_candidates(const float* lod_position, const int32_t* levels,
                               const float* extra, const float* center,
                               int64_t n, float standard_dist, float fork,
                               float resolution_scale, int max_level,
                               int threads, uint8_t* output) {
    if (!lod_position || !levels || !extra || !center || !output || n<0 ||
        standard_dist<=0 || fork<=1 || threads<1) return -1;
    float log_fork=std::log2(fork);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0;i<n;++i) {
        const float* p=lod_position+3*i;
        float dx=p[0]-center[0],dy=p[1]-center[1],dz=p[2]-center[2];
        float distance=std::sqrt(dx*dx+dy*dy+dz*dz)*resolution_scale;
        float predicted=std::log2(standard_dist/distance)/log_fork+extra[i];
        int level=std::isfinite(predicted) ? int(std::nearbyint(predicted))
                                           : (predicted>0 ? max_level : 0);
        level=std::max(0,std::min(max_level,level));
        output[i]=levels[i]<=level ? 1 : 0;
    }
    return 0;
}

static inline bool box_outside_hole(const double* b, const double* planes,
                                    int begin, int end) {
    for (int j=begin;j<end;++j) {
        const double* p=planes+4*j;
        double maximum=p[3];
        for (int k=0;k<3;++k)
            maximum += p[k]*(p[k]>=0 ? b[k+3] : b[k]) + 3.0*b[6]*std::fabs(p[k]);
        if (maximum < -1e-8) return true;
    }
    return false;
}

static inline bool centers_inside_frustum(const double* b, const double* planes) {
    for (int j=0;j<6;++j) {
        // Camera planes carry four coefficients plus four uncertainty terms.
        const double* p=planes+8*j;
        double minimum=p[3];
        for (int k=0;k<3;++k)
            minimum += p[k]*(p[k]>=0 ? b[k] : b[k+3]);
        double uncertainty=p[7];
        for (int k=0;k<3;++k)
            uncertainty += p[4+k]*std::fmax(std::fabs(b[k]),std::fabs(b[k+3]));
        if (!(minimum > uncertainty+1e-7)) return false;
    }
    return true;
}

struct TreeTask {
    int node;
    std::vector<int> active_holes;
    bool keep=false;
};

extern "C" int cpu_select_tree_keep(const double* bounds, const double* nodes,
                                     const int32_t* left, const int32_t* right,
                                     const int64_t* order, const int64_t* range_begin,
                                     const int64_t* range_end, const double* camera,
                                     const double* frustum_planes,
                                     const uint8_t* candidates, int64_t n,
                                     int64_t leaves, const double* hole_planes,
                                     const int32_t* hole_starts, int holes,
                                     int threads, uint8_t* output,
                                     int64_t* counters) {
    if (!bounds || !nodes || !left || !right || !order || !range_begin ||
        !range_end || !camera || !frustum_planes || !candidates ||
        !hole_planes || !hole_starts || !output || !counters ||
        n<0 || leaves<1 || holes<0 || threads<1) return -1;
    std::memset(output,0,size_t(n));
    const int64_t internal=leaves-1;
    auto classify = [&](const TreeTask& task, std::vector<int>& filtered,
                        bool& keep) -> bool {
        const double* box=nodes+7*task.node;
        if (outside(box,camera)) return false;
        filtered.clear();
        for (int h:task.active_holes) {
            if (inside_hole(box,hole_planes,hole_starts[h],hole_starts[h+1]))
                return false;
            if (!box_outside_hole(box,hole_planes,hole_starts[h],hole_starts[h+1]))
                filtered.push_back(h);
        }
        keep=filtered.empty() && centers_inside_frustum(box,frustum_planes);
        return true;
    };
    auto report_keep = [&](int node) {
        for (int64_t j=range_begin[node];j<range_end[node];++j) {
            int64_t id=order[j];
            output[id]=candidates[id];
        }
    };
    std::vector<int> all_holes;
    for (int h=0;h<holes;++h) all_holes.push_back(h);
    std::vector<TreeTask> frontier{{0,all_holes,false}},ready;
    while (int(frontier.size()+ready.size()) < threads*8) {
        std::vector<TreeTask> next;
        bool expanded=false;
        for (const TreeTask& t:frontier) {
            std::vector<int> active;bool keep=false;
            if (!classify(t,active,keep)) continue;
            if (keep) ready.push_back({t.node,{},true});
            else if (t.node>=internal) ready.push_back({t.node,std::move(active),false});
            else {
                next.push_back({left[t.node],active,false});
                next.push_back({right[t.node],std::move(active),false});
                expanded=true;
            }
        }
        frontier.swap(next);
        if (!expanded || frontier.empty()) break;
    }
    ready.insert(ready.end(),frontier.begin(),frontier.end());
    int64_t visited=0,refined=0,kept=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(+:visited,refined,kept)
    for (int task=0;task<int(ready.size());++task) {
        std::vector<TreeTask> stack{ready[task]};
        while (!stack.empty()) {
            TreeTask t=std::move(stack.back());stack.pop_back();
            ++visited;
            if (t.keep) {report_keep(t.node);++kept;continue;}
            std::vector<int> active;bool keep=false;
            if (!classify(t,active,keep)) continue;
            if (keep) {report_keep(t.node);++kept;continue;}
            if (t.node<internal) {
                stack.push_back({left[t.node],active,false});
                stack.push_back({right[t.node],std::move(active),false});
                continue;
            }
            int64_t leaf=t.node-internal;
            int64_t end=std::min<int64_t>(n,(leaf+1)*32);
            for (int64_t j=leaf*32;j<end;++j) {
                int64_t id=order[j];
                if (!candidates[id]) continue;
                ++refined;
                if (outside(bounds+7*id,camera)) continue;
                bool hidden=false;
                for (int h:active)
                    if (inside_hole(bounds+7*id,hole_planes,hole_starts[h],hole_starts[h+1]))
                        {hidden=true;break;}
                if (!hidden) output[id]=1;
            }
        }
    }
    counters[0]=visited;counters[1]=refined;counters[2]=kept;
    return 0;
}

// Algebraic round-LoD predicate with shared precomputed double thresholds.
// Avoid device-specific log2 rounding at half-integer level boundaries.
extern "C" int cpu_lod_threshold(const double* position, const double* radius2,
                                  const int32_t* levels, const double* eye,
                                  double resolution, int64_t n, int threads,
                                  uint8_t* output) {
    if (!position || !radius2 || !levels || !eye || !output ||
        !(resolution > 0) || !std::isfinite(resolution) || n<0 || threads<1) return -1;
    const double res2=resolution*resolution;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i=0;i<n;++i) {
        const double dx=position[3*i]-eye[0],dy=position[3*i+1]-eye[1],dz=position[3*i+2]-eye[2];
        const double distance2=((dx*dx+dy*dy)+dz*dz)*res2;
        output[i]=(levels[i]==0 || distance2<radius2[i] ||
                   (distance2==radius2[i] && (levels[i]%2)==0)) ? 1 : 0;
    }
    return 0;
}
