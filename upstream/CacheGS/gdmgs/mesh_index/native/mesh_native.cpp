#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Lazy_exact_nt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
using Rational = CGAL::Lazy_exact_nt<CGAL::Gmpq>;
using Point3 = std::array<double, 3>;
using Face = std::array<int64_t, 3>;
template<class T> struct Point2 {
    std::array<T,2> value;
    Point2() = default;
    Point2(const T& x,const T& y):value{{x,y}}{}
    const T& operator[](size_t i) const{return value[i];}
    bool operator==(const Point2& p)const{return value==p.value;}
    bool operator!=(const Point2& p)const{return value!=p.value;}
};
template<> struct Point2<Rational> {
    std::array<Rational,2> value;
    std::array<long double,2> lo,hi;
    Point2()=default;
    Point2(const Rational& x,const Rational& y):value{{x,y}}{
        for(int i=0;i<2;++i){
            // CGAL supplies a certified enclosure without eagerly constructing
            // the exact rational intersection. Ambiguous predicates trigger it.
            auto bounds=CGAL::to_interval(value[i]);lo[i]=bounds.first;hi[i]=bounds.second;
        }
    }
    const Rational& operator[](size_t i)const{return value[i];}
    bool operator==(const Point2& p)const{return value==p.value;}
    bool operator!=(const Point2& p)const{return value!=p.value;}
};
template<class T> using Polygon = std::vector<Point2<T>>;
double milliseconds(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}

struct Camera {
    std::array<double,16> w;
    std::array<double,4> domain;
    double near_z, far_z;
    std::array<std::array<long double,4>,6> planes;
    Camera(py::array_t<double,py::array::c_style> matrix,
           std::array<double,4> angular_domain, double near, double far)
      : domain(angular_domain), near_z(near), far_z(far) {
        if(matrix.ndim()!=2 || matrix.shape(0)!=4 || matrix.shape(1)!=4)
            throw std::invalid_argument("w2c must have shape [4,4]");
        std::copy(matrix.data(),matrix.data()+16,w.begin());
        for(double x:w) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite camera");
        for(double x:domain) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite domain");
        if(w[12]!=0 || w[13]!=0 || w[14]!=0 || w[15]!=1 || !(near>0) || !std::isfinite(near) || !(far>near) || !(domain[0]<domain[1]) || !(domain[2]<domain[3]))
            throw std::invalid_argument("invalid affine camera/domain/depth interval");
        // World-space halfspaces with positive side inside the camera pyramid.
        for(int j=0;j<4;++j) {
            planes[0][j]=(long double)w[j]-domain[0]*(long double)w[8+j];
            planes[1][j]=domain[1]*(long double)w[8+j]-w[j];
            planes[2][j]=(long double)w[4+j]-domain[2]*(long double)w[8+j];
            planes[3][j]=domain[3]*(long double)w[8+j]-w[4+j];
            planes[4][j]=w[8+j]; planes[5][j]=-w[8+j];
        }
        planes[4][3]-=near;
        planes[5][3]+=far;
    }
    std::array<long double,3> transform(const Point3& p) const {
        std::array<long double,3> q{};
        for(int i=0;i<3;++i) {
            q[i]=w[4*i+3];
            for(int j=0;j<3;++j) q[i]+=(long double)w[4*i+j]*p[j];
        }
        return q;
    }
};

struct Box {
    Point3 lo{{INFINITY,INFINITY,INFINITY}}, hi{{-INFINITY,-INFINITY,-INFINITY}};
    void add(const Point3& p) { for(int j=0;j<3;++j) {lo[j]=std::min(lo[j],p[j]);hi[j]=std::max(hi[j],p[j]);} }
    void add(const Box& b) {add(b.lo);add(b.hi);}
    double area() const {if(lo[0]>hi[0])return 0;double a=hi[0]-lo[0],b=hi[1]-lo[1],c=hi[2]-lo[2];return 2*(a*b+b*c+c*a);}
};
bool relevant(const Box& b, const Camera& c) {
    for(int i=0;i<(std::isfinite(c.far_z)?6:5);++i) {
        const auto& p=c.planes[i]; long double v=p[3], magnitude=std::abs(p[3]);
        for(int j=0;j<3;++j) {long double x=p[j]*(p[j]>=0?b.hi[j]:b.lo[j]);v+=x;magnitude+=std::abs(x);}
        if(v < -64*std::numeric_limits<long double>::epsilon()*(magnitude+1))return false;
    }
    return true;
}

struct Node {Box box; int64_t begin=0,end=0,left=-1,right=-1;};
struct MeshIndex {
    std::vector<Point3> vertices;
    std::vector<Face> faces;
    std::vector<Box> face_boxes;
    std::vector<Point3> centers;
    std::vector<int64_t> refs;
    std::vector<Node> nodes;
    std::string method;
    int leaf_size;
    double build_ms=0;
    MeshIndex() = default;
    MeshIndex(py::array_t<double,py::array::c_style> v,
              py::array_t<int64_t,py::array::c_style> f,
              std::string build_method,int leaf):method(build_method),leaf_size(leaf) {
        auto start=Clock::now();
        if(v.ndim()!=2 || v.shape(1)!=3 || f.ndim()!=2 || f.shape(1)!=3)throw std::invalid_argument("mesh arrays must have shape [N,3]");
        if(leaf<1 || (method!="median" && method!="binned_sah"))throw std::invalid_argument("invalid BVH settings");
        vertices.resize(v.shape(0));faces.resize(f.shape(0));
        for(ssize_t i=0;i<v.shape(0);++i) for(int j=0;j<3;++j){double x=v.data()[i*3+j];if(!std::isfinite(x))throw std::invalid_argument("nonfinite vertex");vertices[i][j]=x;}
        for(ssize_t i=0;i<f.shape(0);++i) for(int j=0;j<3;++j){auto id=f.data()[i*3+j];if(id<0 || id>=v.shape(0))throw std::invalid_argument("invalid vertex ID");faces[i][j]=id;}
        py::gil_scoped_release release;
        face_boxes.resize(faces.size());centers.resize(faces.size());refs.resize(faces.size());
        std::iota(refs.begin(),refs.end(),0);
        for(size_t i=0;i<faces.size();++i){for(auto id:faces[i])face_boxes[i].add(vertices[id]);for(int j=0;j<3;++j)centers[i][j]=face_boxes[i].lo[j]*0.5+face_boxes[i].hi[j]*0.5;}
        nodes.reserve(faces.empty()?0:2*faces.size());
        if(!faces.empty())build(0,refs.size());
        build_ms=milliseconds(start);
    }
    static std::unique_ptr<MeshIndex> from_layout(
            py::array_t<double,py::array::c_style> v,
            py::array_t<int64_t,py::array::c_style> f,
            py::array_t<int64_t,py::array::c_style> references,
            py::array_t<int64_t,py::array::c_style> topology,
            py::array_t<double,py::array::c_style> bounds,
            std::string method,int leaf_size) {
        if(v.ndim()!=2 || v.shape(1)!=3 || f.ndim()!=2 || f.shape(1)!=3 || references.ndim()!=1 || references.size()!=f.shape(0) || topology.ndim()!=2 || topology.shape(1)!=4 || bounds.ndim()!=3 || bounds.shape(0)!=topology.shape(0) || bounds.shape(1)!=2 || bounds.shape(2)!=3)
            throw std::invalid_argument("invalid persisted mesh index array shape");
        if(leaf_size<=0 || (method!="median" && method!="binned_sah"))throw std::invalid_argument("invalid persisted BVH settings");
        auto mesh=std::make_unique<MeshIndex>();mesh->method=method;mesh->leaf_size=leaf_size;
        mesh->vertices.resize(v.shape(0));mesh->faces.resize(f.shape(0));mesh->nodes.resize(topology.shape(0));
        for(ssize_t i=0;i<v.shape(0);++i)for(int j=0;j<3;++j){double x=v.data()[i*3+j];if(!std::isfinite(x))throw std::invalid_argument("nonfinite persisted vertex");mesh->vertices[i][j]=x;}
        for(ssize_t i=0;i<f.shape(0);++i)for(int j=0;j<3;++j){auto id=f.data()[i*3+j];if(id<0 || id>=v.shape(0))throw std::invalid_argument("invalid persisted vertex ID");mesh->faces[i][j]=id;}
        mesh->refs.assign(references.data(),references.data()+references.size());
        std::vector<bool> seen_ref(mesh->faces.size(),false);
        for(auto id:mesh->refs){if(id<0 || (size_t)id>=seen_ref.size() || seen_ref[id])throw std::invalid_argument("persisted refs must contain every triangle exactly once");seen_ref[id]=true;}
        for(size_t i=0;i<mesh->nodes.size();++i){auto& n=mesh->nodes[i];const auto* p=topology.data()+i*4;n.left=p[0];n.right=p[1];n.begin=p[2];n.end=p[3];for(int j=0;j<3;++j){n.box.lo[j]=bounds.data()[i*6+j];n.box.hi[j]=bounds.data()[i*6+3+j];if(!std::isfinite(n.box.lo[j]) || !std::isfinite(n.box.hi[j]) || n.box.lo[j]>n.box.hi[j])throw std::invalid_argument("invalid persisted BVH bounds");}if(n.begin<0 || n.end<=n.begin || (size_t)n.end>mesh->refs.size())throw std::invalid_argument("invalid persisted reference interval");}
        if(mesh->faces.empty()){if(!mesh->nodes.empty())throw std::invalid_argument("empty mesh has nonempty BVH");return mesh;}
        if(mesh->nodes.empty() || mesh->nodes[0].begin!=0 || (size_t)mesh->nodes[0].end!=mesh->faces.size())throw std::invalid_argument("invalid persisted BVH root");
        std::vector<bool> visited(mesh->nodes.size(),false);std::vector<int64_t> stack{0};size_t count=0;
        auto contains=[](const Box& a,const Box& b){for(int j=0;j<3;++j)if(a.lo[j]>b.lo[j] || a.hi[j]<b.hi[j])return false;return true;};
        while(!stack.empty()){auto id=stack.back();stack.pop_back();if(id<0 || (size_t)id>=mesh->nodes.size() || visited[id])throw std::invalid_argument("persisted BVH contains cycle or invalid child");visited[id]=true;++count;const auto& n=mesh->nodes[id];
            if(n.left<0 || n.right<0){if(n.left!=-1 || n.right!=-1 || n.end-n.begin>leaf_size)throw std::invalid_argument("invalid persisted leaf");Box actual;for(auto i=n.begin;i<n.end;++i)for(auto vi:mesh->faces[mesh->refs[i]])actual.add(mesh->vertices[vi]);if(!contains(n.box,actual))throw std::invalid_argument("persisted leaf loses triangle geometry");}
            else {if((size_t)n.left>=mesh->nodes.size() || (size_t)n.right>=mesh->nodes.size())throw std::invalid_argument("invalid persisted child");const auto& l=mesh->nodes[n.left];const auto& r=mesh->nodes[n.right];if(l.begin!=n.begin || l.end!=r.begin || r.end!=n.end || !contains(n.box,l.box) || !contains(n.box,r.box))throw std::invalid_argument("invalid persisted subtree intervals/bounds");stack.push_back(n.right);stack.push_back(n.left);}}
        if(count!=mesh->nodes.size())throw std::invalid_argument("persisted BVH has unreachable nodes");
        return mesh;
    }
    int64_t build(int64_t begin,int64_t end) {
        int64_t idx=nodes.size();nodes.emplace_back();
        Box box,cb;for(auto i=begin;i<end;++i){box.add(face_boxes[refs[i]]);cb.add(centers[refs[i]]);}
        nodes[idx].box=box;nodes[idx].begin=begin;nodes[idx].end=end;
        if(end-begin<=leaf_size)return idx;
        int axis=0;for(int j=1;j<3;++j)if(cb.hi[j]-cb.lo[j]>cb.hi[axis]-cb.lo[axis])axis=j;
        int64_t mid=(begin+end)/2;
        bool partitioned=false;
        if(method=="binned_sah") {
            constexpr int B=16; double best=INFINITY; int best_axis=-1,best_split=-1;
            for(int a=0;a<3;++a) {
                double span=cb.hi[a]-cb.lo[a];if(!(span>0) || !std::isfinite(span))continue;
                std::array<Box,B> boxes;std::array<int64_t,B> count{};
                for(auto i=begin;i<end;++i){int b=std::clamp(int((centers[refs[i]][a]-cb.lo[a])/span*B),0,B-1);boxes[b].add(face_boxes[refs[i]]);++count[b];}
                std::array<double,B> la{},ra{};std::array<int64_t,B> lc{},rc{};Box l,r;int64_t nl=0,nr=0;
                for(int b=0;b<B;++b){if(count[b])l.add(boxes[b]);nl+=count[b];la[b]=l.area();lc[b]=nl;int z=B-1-b;if(count[z])r.add(boxes[z]);nr+=count[z];ra[z]=r.area();rc[z]=nr;}
                for(int b=0;b<B-1;++b)if(lc[b] && rc[b+1]){double cost=la[b]*lc[b]+ra[b+1]*rc[b+1];if(cost<best){best=cost;best_axis=a;best_split=b;}}
            }
            if(best_axis>=0){double span=cb.hi[best_axis]-cb.lo[best_axis];auto p=std::partition(refs.begin()+begin,refs.begin()+end,[&](int64_t id){int bin=std::clamp(int((centers[id][best_axis]-cb.lo[best_axis])/span*B),0,B-1);return bin<=best_split;});mid=p-refs.begin();partitioned=mid>begin && mid<end;}
        }
        if(!partitioned){mid=(begin+end)/2;std::nth_element(refs.begin()+begin,refs.begin()+mid,refs.begin()+end,[&](int64_t a,int64_t b){return centers[a][axis]<centers[b][axis] || (centers[a][axis]==centers[b][axis] && a<b);});}
        int64_t left=build(begin,mid),right=build(mid,end);nodes[idx].left=left;nodes[idx].right=right;return idx;
    }
    bool triangle_relevant(int64_t id,const Camera& c) const {
        // Reject only if every triangle vertex is outside one frustum plane.
        // The conservative predicate permits corner false positives but never
        // uses a centroid or clips the mesh to the anchor partition root.
        for(int i=0;i<(std::isfinite(c.far_z)?6:5);++i){const auto& p=c.planes[i];bool all_out=true;
            for(auto vi:faces[id]){long double v=p[3],mag=std::abs(p[3]);for(int j=0;j<3;++j){long double term=p[j]*vertices[vi][j];v+=term;mag+=std::abs(term);}if(v>=-64*std::numeric_limits<long double>::epsilon()*(mag+1)){all_out=false;break;}}
            if(all_out)return false;
        }return true;
    }
    py::dict query(py::array_t<double,py::array::c_style> w,std::array<double,4> d,double near,double far,bool brute) const {
        Camera c(w,d,near,far);auto start=Clock::now();std::vector<int64_t> out;int64_t tested=0,visited=0;
        {py::gil_scoped_release release;
        if(brute){for(size_t id=0;id<faces.size();++id){++tested;if(triangle_relevant(id,c))out.push_back(id);}}
        else if(!nodes.empty()){std::vector<int64_t> stack{0};while(!stack.empty()){auto id=stack.back();stack.pop_back();++visited;const auto& n=nodes[id];if(!relevant(n.box,c))continue;if(n.left<0){for(auto i=n.begin;i<n.end;++i){++tested;if(triangle_relevant(refs[i],c))out.push_back(refs[i]);}}else{stack.push_back(n.right);stack.push_back(n.left);}}std::sort(out.begin(),out.end());}}
        double elapsed=milliseconds(start);py::array_t<int64_t> ids(out.size());std::copy(out.begin(),out.end(),ids.mutable_data());
        py::dict r;r["triangle_ids"]=ids;r["elapsed_ms"]=elapsed;r["visited_nodes"]=visited;r["tested_triangles"]=tested;r["returned_triangles"]=out.size();r["complete"]=true;return r;
    }
    py::dict layout() const {
        py::array_t<int64_t> r(refs.size());std::copy(refs.begin(),refs.end(),r.mutable_data());
        py::array_t<int64_t> children({(ssize_t)nodes.size(),(ssize_t)4});py::array_t<double> bounds({(ssize_t)nodes.size(),(ssize_t)2,(ssize_t)3});
        for(size_t i=0;i<nodes.size();++i){auto& n=nodes[i];auto* p=children.mutable_data()+i*4;p[0]=n.left;p[1]=n.right;p[2]=n.begin;p[3]=n.end;for(int j=0;j<3;++j){bounds.mutable_data()[i*6+j]=n.box.lo[j];bounds.mutable_data()[i*6+3+j]=n.box.hi[j];}}
        py::dict o;o["triangle_refs"]=r;o["nodes"]=children;o["bounds"]=bounds;o["build_ms"]=build_ms;return o;
    }
};

template<class T> T cross(const Point2<T>& a,const Point2<T>& b,const Point2<T>& p) {
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);
}
template<class T> int orientation(const Point2<T>& a,const Point2<T>& b,const Point2<T>& p){T c=cross(a,b,p);return c>0?1:c<0?-1:0;}
struct Interval {long double lo,hi;};
Interval subtract(const Interval& a,const Interval& b){return {std::nextafter(a.lo-b.hi,-(long double)INFINITY),std::nextafter(a.hi-b.lo,(long double)INFINITY)};}
Interval multiply(const Interval& a,const Interval& b){std::array<long double,4> v{{a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi}};for(auto x:v)if(std::isnan(x))return {-(long double)INFINITY,(long double)INFINITY};return {std::nextafter(*std::min_element(v.begin(),v.end()),-(long double)INFINITY),std::nextafter(*std::max_element(v.begin(),v.end()),(long double)INFINITY)};}
template<> int orientation<Rational>(const Point2<Rational>& a,const Point2<Rational>& b,const Point2<Rational>& p){
    auto coordinate=[](const Point2<Rational>& p,int k){return Interval{p.lo[k],p.hi[k]};};
    Interval dx=subtract(coordinate(b,0),coordinate(a,0)),dy=subtract(coordinate(b,1),coordinate(a,1));
    Interval px=subtract(coordinate(p,0),coordinate(a,0)),py=subtract(coordinate(p,1),coordinate(a,1));
    Interval result=subtract(multiply(dx,py),multiply(dy,px));
    if(result.lo>0)return 1;
    if(result.hi<0)return -1;
    Rational exact=cross(a,b,p);return exact>0?1:exact<0?-1:0;
}
template<class T> Polygon<T> halfplane(const Polygon<T>& p,const Point2<T>& a,const Point2<T>& b,bool inside) {
    Polygon<T> out;if(p.empty())return out;
    auto prev=p.back();int sp=orientation(a,b,prev);bool ip=inside?sp>=0:sp<=0;
    for(const auto& cur:p){int sc=orientation(a,b,cur);bool ic=inside?sc>=0:sc<=0;
        if(ip!=ic){T cp=cross(a,b,prev),cc=cross(a,b,cur);T t=cp/(cp-cc);out.push_back({prev[0]+t*(cur[0]-prev[0]),prev[1]+t*(cur[1]-prev[1])});}
        if(ic)out.push_back(cur);
        prev=cur;ip=ic;
    }
    Polygon<T> clean;
    for(const auto& p:out)if(clean.empty() || clean.back()!=p)clean.push_back(p);
    if(clean.size()>1 && clean.front()==clean.back())clean.pop_back();
    return clean;
}
template<class T> T signed_area(const Polygon<T>& p){T a=0;if(p.size()<3)return a;for(size_t i=0;i<p.size();++i){const auto& q=p[(i+1)%p.size()];a+=p[i][0]*q[1]-p[i][1]*q[0];}return a;}
template<class T> bool positive_convex_area(const Polygon<T>& p){for(size_t i=1;i+1<p.size();++i)if(orientation(p[0],p[i],p[i+1])>0)return true;return false;}
template<class T> void subtract_cover(std::vector<Polygon<T>>& uncovered,const Polygon<T>& cover) {
    std::vector<Polygon<T>> next;
    for(const auto& poly:uncovered){
        // Exact convex SAT prevents fragmentation by triangles that never
        // overlap this piece. It also proves a single-triangle full cover.
        bool disjoint=false, contained=true;
        for(size_t j=0;j<cover.size();++j){
            const auto& a=cover[j];const auto& b=cover[(j+1)%cover.size()];
            if(a==b)continue;
            bool any_positive=false;
            for(const auto& p:poly){int value=orientation(a,b,p);if(value>0)any_positive=true;if(value<0)contained=false;}
            if(!any_positive){disjoint=true;break;}
        }
        if(!disjoint && contained)continue;
        if(!disjoint)for(size_t j=0;j<poly.size();++j){
            if(poly[j]==poly[(j+1)%poly.size()])continue;
            bool any_positive=false;for(const auto& p:cover)if(orientation(poly[j],poly[(j+1)%poly.size()],p)>0){any_positive=true;break;}
            if(!any_positive){disjoint=true;break;}
        }
        if(disjoint){next.push_back(poly);continue;}
        Polygon<T> remainder=poly;
        for(size_t j=0;j<cover.size() && !remainder.empty();++j){const auto& a=cover[j];const auto& b=cover[(j+1)%cover.size()];if(a==b)continue;auto outside=halfplane(remainder,a,b,false);if(positive_convex_area(outside))next.push_back(std::move(outside));remainder=halfplane(remainder,a,b,true);}
    }uncovered.swap(next);
}
template<class T> std::vector<std::array<T,3>> clip_depth(std::vector<std::array<T,3>> p,T z,bool near) {
    std::vector<std::array<T,3>> out;if(p.empty())return out;auto prev=p.back();T cp=near?T(prev[2]-z):T(z-prev[2]);bool ip=cp>=0;
    for(const auto& cur:p){T cc=near?T(cur[2]-z):T(z-cur[2]);bool ic=cc>=0;if(ip!=ic){T t=cp/(cp-cc);out.push_back({prev[0]+t*(cur[0]-prev[0]),prev[1]+t*(cur[1]-prev[1]),z});}if(ic)out.push_back(cur);prev=cur;cp=cc;ip=ic;}return out;
}
struct Projected {int64_t id;Polygon<long double> p;long double zmax;};
struct ExactProjected {Polygon<Rational> p;Rational zmax=0;};
Polygon<Rational> exact_projection(const MeshIndex& mesh,int64_t id,const Camera& c,Rational& zmax) {
    std::vector<std::array<Rational,3>> p;
    for(auto vi:mesh.faces[id]){std::array<Rational,3> q;for(int i=0;i<3;++i){q[i]=Rational(c.w[4*i+3]);for(int j=0;j<3;++j)q[i]+=Rational(c.w[4*i+j])*Rational(mesh.vertices[vi][j]);}p.push_back(std::move(q));}
    p=clip_depth(std::move(p),Rational(c.near_z),true);if(std::isfinite(c.far_z))p=clip_depth(std::move(p),Rational(c.far_z),false);
    Polygon<Rational> out;for(const auto& v:p){out.push_back({v[0]/v[2],v[1]/v[2]});if(v[2]>zmax)zmax=v[2];}auto area=signed_area(out);if(area<0)std::reverse(out.begin(),out.end());if(area==0)out.clear();return out;
}
py::dict build_ori(const MeshIndex& mesh,py::array_t<int64_t,py::array::c_style> ids,
        py::array_t<double,py::array::c_style> w,std::array<double,4> d,double near,double far,int height,int width) {
    if(ids.ndim()!=1 || height<=0 || width<=0 || (int64_t)height*width>std::numeric_limits<ssize_t>::max()/8)throw std::invalid_argument("invalid triangle IDs/ORI dimensions");
    Camera camera(w,d,near,far);int64_t last=-1;for(ssize_t i=0;i<ids.size();++i){auto id=ids.data()[i];if(id<=last || id<0 || (size_t)id>=mesh.faces.size())throw std::invalid_argument("triangle IDs must be valid, sorted, unique");last=id;}
    auto start=Clock::now();py::array_t<double> depth({height,width});std::fill(depth.mutable_data(),depth.mutable_data()+depth.size(),INFINITY);
    int64_t covered=0,approx_covered=0,exact_rejected=0,projected_count=0,cell_references=0,fragment_peak=0;
    double projection_ms=0,prefilter_ms=0,exact_ms=0;
    const bool trace=std::getenv("GDMGS_MESH_PROFILE")!=nullptr;
    {py::gil_scoped_release release;
    std::vector<Projected> projected;projected.reserve(ids.size());
    for(ssize_t i=0;i<ids.size();++i){auto id=ids.data()[i];std::vector<std::array<long double,3>> p;for(auto vi:mesh.faces[id])p.push_back(camera.transform(mesh.vertices[vi]));p=clip_depth(std::move(p),(long double)near,true);if(std::isfinite(far))p=clip_depth(std::move(p),(long double)far,false);if(p.size()<3)continue;
        Projected q;q.id=id;q.zmax=0;bool finite=true;for(const auto& v:p){q.zmax=std::max(q.zmax,v[2]);long double x=v[0]/v[2],y=v[1]/v[2];if(!std::isfinite(x)||!std::isfinite(y))finite=false;q.p.push_back({x,y});}if(!finite)continue;auto area=signed_area(q.p);if(area==0)continue;if(area<0)std::reverse(q.p.begin(),q.p.end());projected.push_back(std::move(q));}
    // A common, deterministic order gives the two retrieval backends identical ORI.
    std::sort(projected.begin(),projected.end(),[](const Projected&a,const Projected&b){return a.zmax<b.zmax || (a.zmax==b.zmax && a.id<b.id);});projected_count=projected.size();
    std::vector<std::unique_ptr<ExactProjected>> exact_projected(projected.size());
    std::vector<std::vector<int64_t>> cells((int64_t)height*width);
    const long double dx=((long double)d[1]-d[0])/width,dy=((long double)d[3]-d[2])/height;
    for(size_t i=0;i<projected.size();++i){auto& p=projected[i].p;long double xmin=INFINITY,xmax=-INFINITY,ymin=INFINITY,ymax=-INFINITY;for(const auto& v:p){xmin=std::min(xmin,v[0]);xmax=std::max(xmax,v[0]);ymin=std::min(ymin,v[1]);ymax=std::max(ymax,v[1]);}if(xmax<d[0]||xmin>d[1]||ymax<d[2]||ymin>d[3])continue;
        int x0=(int)std::clamp(std::floor((xmin-d[0])/dx)-1,(long double)0,(long double)width-1),x1=(int)std::clamp(std::floor((xmax-d[0])/dx)+1,(long double)0,(long double)width-1);int y0=(int)std::clamp(std::floor((ymin-d[2])/dy)-1,(long double)0,(long double)height-1),y1=(int)std::clamp(std::floor((ymax-d[2])/dy)+1,(long double)0,(long double)height-1);
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){cells[(int64_t)y*width+x].push_back(i);++cell_references;}}
    projection_ms=milliseconds(start);
    if(trace)std::cerr<<"ORI projected="<<projected_count<<" cell_refs="<<cell_references<<" projection_ms="<<projection_ms<<std::endl;
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){const auto& candidates=cells[(int64_t)y*width+x];
        if(trace && ((int64_t)y*width+x)%1024==0)std::cerr<<"ORI cell="<<((int64_t)y*width+x)<<" candidates="<<candidates.size()<<" prefilter_ms="<<prefilter_ms<<" exact_ms="<<exact_ms<<" fragment_peak="<<fragment_peak<<std::endl;
        if(candidates.empty())continue;
        auto prefilter_start=Clock::now();
        long double xl=d[0]+x*dx,xr=d[0]+(x+1)*dx,yl=d[2]+y*dy,yr=d[2]+(y+1)*dy;
        std::vector<Polygon<long double>> remaining{{{xl,yl},{xr,yl},{xr,yr},{xl,yr}}};size_t used=0;
        for(;used<candidates.size();++used){subtract_cover(remaining,projected[candidates[used]].p);fragment_peak=std::max(fragment_peak,(int64_t)remaining.size());if(remaining.empty()){++used;break;}}
        prefilter_ms+=milliseconds(prefilter_start);
        if(!remaining.empty())continue;
        ++approx_covered;
        auto exact_start=Clock::now();
        // Floating overlay is only a prefilter. Certification repeats continuous
        // set difference with exact rational arithmetic on the input doubles.
        // No sample count, area tolerance, or quantization can fill a small hole.
        Rational exl=Rational(d[0])+(Rational(d[1])-Rational(d[0]))*x/width,exr=Rational(d[0])+(Rational(d[1])-Rational(d[0]))*(x+1)/width;
        Rational eyl=Rational(d[2])+(Rational(d[3])-Rational(d[2]))*y/height,eyr=Rational(d[2])+(Rational(d[3])-Rational(d[2]))*(y+1)/height;
        std::vector<Polygon<Rational>> exact{{{exl,eyl},{exr,eyl},{exr,eyr},{exl,eyr}}};Rational zmax=0;
        for(size_t k=0;k<used;++k){auto id=candidates[k];if(!exact_projected[id]){auto ep=std::make_unique<ExactProjected>();ep->p=exact_projection(mesh,projected[id].id,camera,ep->zmax);exact_projected[id]=std::move(ep);}const auto& ep=*exact_projected[id];if(ep.p.empty())continue;zmax=std::max(zmax,ep.zmax);subtract_cover(exact,ep.p);if(exact.empty())break;}
        exact_ms+=milliseconds(exact_start);
        if(!exact.empty()){++exact_rejected;continue;}
        // Every clipped triangle point has z <= its largest vertex z. Convert
        // the exact rational maximum outward, including cancellation in W2C.
        double z=CGAL::to_interval(zmax).second;
        if(std::isfinite(z)){depth.mutable_data()[(int64_t)y*width+x]=z;++covered;}
    }}
    py::dict result;result["depth_bounds"]=depth;result["elapsed_ms"]=milliseconds(start);result["covered_cells"]=covered;result["unknown_cells"]=(int64_t)height*width-covered;result["approx_covered_cells"]=approx_covered;result["exact_rejected_cells"]=exact_rejected;result["projected_triangles"]=projected_count;result["cell_triangle_references"]=cell_references;result["fragment_peak"]=fragment_peak;result["projection_ms"]=projection_ms;result["prefilter_ms"]=prefilter_ms;result["exact_ms"]=exact_ms;result["complete"]=true;return result;
}

PYBIND11_MODULE(GDMGS_mesh_native,m) {
    m.doc()="Object-split BVH and continuous union ORI with CGAL lazy exact certification";
    py::class_<MeshIndex>(m,"MeshIndex")
      .def(py::init<py::array_t<double,py::array::c_style>,py::array_t<int64_t,py::array::c_style>,std::string,int>(),py::arg("vertices").noconvert(),py::arg("triangles").noconvert(),py::arg("method")="median",py::arg("leaf_size")=8)
      .def("query",&MeshIndex::query,py::arg("w2c").noconvert(),py::arg("angular_domain"),py::arg("near"),py::arg("far"),py::arg("brute_force")=false)
      .def("layout",&MeshIndex::layout)
      .def_static("from_layout",&MeshIndex::from_layout,py::arg("vertices").noconvert(),py::arg("triangles").noconvert(),py::arg("triangle_refs").noconvert(),py::arg("nodes").noconvert(),py::arg("bounds").noconvert(),py::arg("method"),py::arg("leaf_size"))
      .def_property_readonly("build_ms",[](const MeshIndex&m){return m.build_ms;});
    m.def("build_ori",&build_ori,py::arg("mesh"),py::arg("triangle_ids").noconvert(),py::arg("w2c").noconvert(),py::arg("angular_domain"),py::arg("near"),py::arg("far"),py::arg("height"),py::arg("width"));
}
