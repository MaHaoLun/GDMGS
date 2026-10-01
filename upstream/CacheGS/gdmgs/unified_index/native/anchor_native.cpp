#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace py = pybind11;
using I = int64_t;
using Box = std::array<double, 6>;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
static void require_array(const py::array& a, const py::dtype& type, int ndim, const char* name) {
    if (!a.dtype().is(type) || a.ndim()!=ndim || !(a.flags() & py::array::c_style))
        throw std::invalid_argument(std::string(name)+" has wrong dtype, dimensions or contiguity");
}
static bool bounded(const Box& b) {
    for (double x:b) if (!std::isfinite(x)) return false;
    return true;
}
static Box empty_box() { return {INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY}; }
static void extend(Box& a, const Box& b) {
    for(int k=0;k<3;++k) { a[k]=std::min(a[k],b[k]); a[k+3]=std::max(a[k+3],b[k+3]); }
}
struct Node { Box partition, support, centers; double radius; bool known; I begin,end; std::array<I,8> children; };
struct SupportRecord { Box bounds,centers; double radius; bool known; };
struct Stats {
    I visited_nodes=0, certified_nodes=0, leaf_checks=0, anchor_checks=0, empty_nodes=0;
    I unbounded=0, near_plane=0, unknown_cells=0, depth_fail=0, outside=0, certified_anchors=0;
    I cell_checks=0, early_unknown_cells=0;
};
struct Camera { const double* w; const double* ori; I h,width; std::array<double,4> domain;
    double pad_x,pad_y,near_z,margin,rotation_norm; };
// 0 is unknown/visible; 1 is a complete conservative occlusion certificate.
static bool certify(const Box& b, const Box& centers,double radius,bool known,const Camera& c, Stats& s) {
    if(!known) { ++s.unbounded; return false; }
    // Negative-only shortcut: the all-offset center-box midpoint lies inside
    // the full screen envelope. An Unknown cell there forbids certification;
    // a finite cell here never authorizes removal. Keep the exact envelope
    // calculation below for every candidate which survives this cheap test.
    double midpoint[3],q[3];
    for(int j=0;j<3;++j)midpoint[j]=centers[j]*.5+centers[j+3]*.5;
    for(int j=0;j<3;++j)q[j]=c.w[j*4]*midpoint[0]+c.w[j*4+1]*midpoint[1]+c.w[j*4+2]*midpoint[2]+c.w[j*4+3];
    if(std::isfinite(q[2])&&q[2]>c.near_z){
        const double x=q[0]/q[2],y=q[1]/q[2];
        if(x>=c.domain[0]&&x<=c.domain[1]&&y>=c.domain[2]&&y<=c.domain[3]){
            const I ix=std::min<I>(c.width-1,I((x-c.domain[0])/(c.domain[1]-c.domain[0])*c.width));
            const I iy=std::min<I>(c.h-1,I((y-c.domain[2])/(c.domain[3]-c.domain[2])*c.h));
            ++s.cell_checks;
            if(!std::isfinite(c.ori[iy*c.width+ix])){++s.early_unknown_cells;++s.unknown_cells;return false;}
        }
    }
    auto project=[&](const Box& box,double& xmin,double& xmax,double& ymin,double& ymax,double& zmin) {
        xmin=ymin=zmin=INFINITY;xmax=ymax=-INFINITY;
        for(int corner=0;corner<8;++corner) {
            double p[3]={box[(corner&1)?3:0],box[(corner&2)?4:1],box[(corner&4)?5:2]};
            double qlo[3],qhi[3];
            for(int j=0;j<3;++j) {
                double q=c.w[j*4+3],scale=std::abs(q);
                for(int k=0;k<3;++k){q+=c.w[j*4+k]*p[k];scale+=std::abs(c.w[j*4+k]*p[k]);}
                // The extra 16 double ulps dominate addition rounding and
                // the previous nextafter step, preserving an outward envelope
                // without two libm calls per axis/corner.
                const double error=(64*std::numeric_limits<float>::epsilon()+16*std::numeric_limits<double>::epsilon())*std::max(1.0,scale);
                qlo[j]=q-error;qhi[j]=q+error;
            }
            if(!std::isfinite(qlo[2])||!std::isfinite(qhi[2])||qlo[2]<=c.near_z) return false;
            zmin=std::min(zmin,qlo[2]);
            for(int k=0;k<2;++k){
                // Positive denominator: select the extremizing endpoint by
                // numerator sign instead of evaluating all four quotients.
                const double low=qlo[k]/(qlo[k]>=0?qhi[2]:qlo[2]);
                const double high=qhi[k]/(qhi[k]>=0?qlo[2]:qhi[2]);
                if(!std::isfinite(low)||!std::isfinite(high))return false;
                if(k==0){xmin=std::min(xmin,low);xmax=std::max(xmax,high);}else{ymin=std::min(ymin,low);ymax=std::max(ymax,high);}
            }
        }
        return true;
    };
    double xlo,xhi,ylo,yhi,zlo;
    if(!project(b,xlo,xhi,ylo,yhi,zlo)){++s.near_plane;return false;}
    double cxlo,cxhi,cylo,cyhi,czlo;
    if(!project(centers,cxlo,cxhi,cylo,cyhi,czlo)){++s.near_plane;return false;}
    // gsplat projects covariance with a clamped pinhole Jacobian. Projecting
    // a world ellipsoid alone need not enclose this *linearized* footprint.
    // Each Jacobian row norm is sqrt(1+clamped_ratio^2)/z. Use a spectral
    // upper bound for the camera rotation, all-offset centers, and max sigma.
    const double xlimit=std::max(std::abs(c.domain[0]),std::abs(c.domain[1]))+.15*(c.domain[1]-c.domain[0]);
    const double ylimit=std::max(std::abs(c.domain[2]),std::abs(c.domain[3]))+.15*(c.domain[3]-c.domain[2]);
    const double xr=std::min(std::max(std::abs(cxlo),std::abs(cxhi)),xlimit);
    const double yr=std::min(std::max(std::abs(cylo),std::abs(cyhi)),ylimit);
    const double dx=radius*c.rotation_norm*std::sqrt(1+xr*xr)/czlo;
    const double dy=radius*c.rotation_norm*std::sqrt(1+yr*yr)/czlo;
    xlo=std::min(xlo,cxlo-dx);xhi=std::max(xhi,cxhi+dx);
    ylo=std::min(ylo,cylo-dy);yhi=std::max(yhi,cyhi+dy);
    xlo=std::nextafter(xlo-c.pad_x,-INFINITY); xhi=std::nextafter(xhi+c.pad_x,INFINITY);
    ylo=std::nextafter(ylo-c.pad_y,-INFINITY); yhi=std::nextafter(yhi+c.pad_y,INFINITY);
    // Empty screen intersection does not constitute a mesh certificate.
    if(xhi<c.domain[0]||xlo>c.domain[1]||yhi<c.domain[2]||ylo>c.domain[3]) {++s.outside;return false;}
    xlo=std::max(xlo,c.domain[0]);xhi=std::min(xhi,c.domain[1]);
    ylo=std::max(ylo,c.domain[2]);yhi=std::min(yhi,c.domain[3]);
    auto cell=[](double x,double lo,double hi,I n) {return std::max<I>(0,std::min<I>(n-1,I(std::floor((x-lo)/(hi-lo)*n))));};
    I ix0=cell(xlo,c.domain[0],c.domain[1],c.width),ix1=cell(xhi,c.domain[0],c.domain[1],c.width);
    I iy0=cell(ylo,c.domain[2],c.domain[3],c.h),iy1=cell(yhi,c.domain[2],c.domain[3],c.h);
    double maxz=-INFINITY;
    for(I iy=iy0;iy<=iy1;++iy) for(I ix=ix0;ix<=ix1;++ix) {
        ++s.cell_checks;double z=c.ori[iy*c.width+ix];
        if(!std::isfinite(z)) { ++s.unknown_cells;return false; }
        maxz=std::max(maxz,z);
    }
    const double roundoff=128.0*std::numeric_limits<float>::epsilon()*std::max({1.0,std::abs(zlo),std::abs(maxz)});
    if(maxz+c.margin+roundoff<std::nextafter(zlo,-INFINITY)) return true;
    ++s.depth_fail;return false;
}

class AnchorTree {
    std::vector<double> positions_;
    std::vector<Box> support_,centers_;
    std::vector<double> radii_;
    std::vector<Node> nodes_;
    std::vector<SupportRecord> dfs_support_;
    std::vector<unsigned char> known_;
    void derive_query_layout() {
        known_.resize(support_.size());dfs_support_.resize(support_.size());
        for(I row=0;row<I(support_.size());++row)known_[row]=bounded(support_[row])&&bounded(centers_[row])&&std::isfinite(radii_[row]);
        for(I rank=0;rank<I(dfs_.size());++rank){I row=dfs_[rank];dfs_support_[rank]={support_[row],centers_[row],radii_[row],bool(known_[row])};}
        for(Node& node:nodes_)node.known=bounded(node.support)&&bounded(node.centers)&&std::isfinite(node.radius);
    }
    std::vector<I> dfs_,rank_;
    I leaf_capacity_, max_depth_;
    I build_node(I begin,I end,const Box& partition,I depth) {
        const I id=I(nodes_.size()); Node node; node.partition=partition;node.support=empty_box();node.centers=empty_box();node.radius=0;
        node.begin=begin;node.end=end;node.children.fill(-1);nodes_.push_back(node);
        for(I i=begin;i<end;++i) {extend(nodes_[id].support,support_[dfs_[i]]);extend(nodes_[id].centers,centers_[dfs_[i]]);nodes_[id].radius=std::max(nodes_[id].radius,radii_[dfs_[i]]);}
        if(end-begin<=leaf_capacity_||depth>=max_depth_) return id;
        const double mid[3]={partition[0]+(partition[3]-partition[0])*0.5,partition[1]+(partition[4]-partition[1])*0.5,partition[2]+(partition[5]-partition[2])*0.5};
        std::array<I,8> counts{};
        auto octant=[&](I row) {const double* p=&positions_[row*3];return (p[0]>=mid[0]?1:0)|(p[1]>=mid[1]?2:0)|(p[2]>=mid[2]?4:0);};
        for(I i=begin;i<end;++i) ++counts[octant(dfs_[i])];
        bool degenerate=true;for(int k=0;k<3;++k) if(partition[k]<mid[k]&&mid[k]<partition[k+3])degenerate=false;
        if(degenerate) return id;
        std::array<I,8> starts;I off=begin;for(int k=0;k<8;++k){starts[k]=off;off+=counts[k];}
        auto cursors=starts;std::vector<I> tmp(size_t(end-begin));
        for(I i=begin;i<end;++i){I row=dfs_[i];tmp[cursors[octant(row)]++-begin]=row;}
        std::copy(tmp.begin(),tmp.end(),dfs_.begin()+begin);
        for(int k=0;k<8;++k) if(counts[k]) {
            Box box=partition;for(int j=0;j<3;++j) if(k&(1<<j))box[j]=mid[j];else box[j+3]=mid[j];
            I child=build_node(starts[k],starts[k]+counts[k],box,depth+1);nodes_[id].children[k]=child;
        }
        return id;
    }
    void traverse(I id,const Camera& camera,const std::vector<I>& prefix,const std::vector<unsigned char>& fov,
                  std::vector<unsigned char>& removed,Stats& stats) const {
        const Node& node=nodes_[id];++stats.visited_nodes;
        if(prefix[node.end]==prefix[node.begin]){++stats.empty_nodes;return;}
        if(certify(node.support,node.centers,node.radius,node.known,camera,stats)) {
            ++stats.certified_nodes;
            for(I i=node.begin;i<node.end;++i) if(fov[i]) {removed[i]=1;++stats.certified_anchors;}
            return;
        }
        bool leaf=true;for(I child:node.children) if(child>=0){leaf=false;traverse(child,camera,prefix,fov,removed,stats);}
        if(leaf){++stats.leaf_checks;for(I i=node.begin;i<node.end;++i)if(fov[i]){
            ++stats.anchor_checks;const auto& support=dfs_support_[i];if(certify(support.bounds,support.centers,support.radius,support.known,camera,stats)){removed[i]=1;++stats.certified_anchors;}
        }}
    }
public:
    AnchorTree() = default;
    AnchorTree(const py::array& p,const py::array& support,const py::array& centers,const py::array& radii,I leaf_capacity,I max_depth):leaf_capacity_(leaf_capacity),max_depth_(max_depth) {
        require_array(p,py::dtype::of<double>(),2,"positions");require_array(support,py::dtype::of<double>(),2,"supports");
        if(p.shape(1)!=3||support.shape(1)!=6||support.shape(0)!=p.shape(0)||leaf_capacity<1||max_depth<1||max_depth>64)
            throw std::invalid_argument("invalid anchor geometry or partition settings");
        require_array(centers,py::dtype::of<double>(),2,"centers");require_array(radii,py::dtype::of<double>(),1,"radii");
        if(centers.shape(0)!=p.shape(0)||centers.shape(1)!=6||radii.size()!=p.shape(0))throw std::invalid_argument("Center/radius count mismatch");
        I n=p.shape(0);const double* pp=(const double*)p.data();const double* sp=(const double*)support.data();
        positions_.assign(pp,pp+n*3);support_.resize(n);centers_.resize(n);radii_.resize(n);dfs_.resize(n);rank_.resize(n);std::iota(dfs_.begin(),dfs_.end(),0);
        Box partition=empty_box();
        for(I i=0;i<n;++i){
            for(int k=0;k<3;++k){double x=pp[i*3+k];if(!std::isfinite(x))throw std::invalid_argument("anchor positions must be finite");partition[k]=std::min(partition[k],x);partition[k+3]=std::max(partition[k+3],x);}
            for(int k=0;k<6;++k){support_[i][k]=sp[i*6+k];centers_[i][k]=((const double*)centers.data())[i*6+k];}
            radii_[i]=((const double*)radii.data())[i];
            if(std::isnan(radii_[i])||radii_[i]<0)throw std::invalid_argument("Radius invalid");
            for(int k=0;k<3;++k)if(std::isnan(centers_[i][k])||std::isnan(centers_[i][k+3])||centers_[i][k]>centers_[i][k+3])throw std::invalid_argument("Center bounds invalid");
            for(int k=0;k<3;++k)if(std::isnan(support_[i][k])||std::isnan(support_[i][k+3])||support_[i][k]>support_[i][k+3])throw std::invalid_argument("invalid support bounds");
        }
        if(n)build_node(0,n,partition,0);
        for(I r=0;r<n;++r)rank_[dfs_[r]]=r;
        derive_query_layout();
    }
    py::dict layout() const {
        const I n=I(dfs_.size()),m=I(nodes_.size());py::array_t<I> dfs(n),rank(n),intervals({m,I(2)}),children({m,I(8)});
        py::array_t<double> partition({m,I(6)}),support({m,I(6)}),anchor_support({n,I(6)}),positions({n,I(3)}),anchor_centers({n,I(6)}),node_centers({m,I(6)}),anchor_radii(n),node_radii(m);
        std::copy(positions_.begin(),positions_.end(),positions.mutable_data());
        std::copy(dfs_.begin(),dfs_.end(),dfs.mutable_data());std::copy(rank_.begin(),rank_.end(),rank.mutable_data());
        for(I i=0;i<m;++i){intervals.mutable_at(i,0)=nodes_[i].begin;intervals.mutable_at(i,1)=nodes_[i].end;
            for(int k=0;k<8;++k)children.mutable_at(i,k)=nodes_[i].children[k];
            node_radii.mutable_at(i)=nodes_[i].radius;
            for(int k=0;k<6;++k){node_centers.mutable_at(i,k)=nodes_[i].centers[k];partition.mutable_at(i,k)=nodes_[i].partition[k];support.mutable_at(i,k)=nodes_[i].support[k];}}
        for(I i=0;i<n;++i){anchor_radii.mutable_at(i)=radii_[i];for(int k=0;k<6;++k){anchor_support.mutable_at(i,k)=support_[i][k];anchor_centers.mutable_at(i,k)=centers_[i][k];}}
        py::dict result;result["dfs_to_row"]=dfs;result["rank_of_row"]=rank;result["intervals"]=intervals;result["children"]=children;
        result["anchor_center_bounds"]=anchor_centers;result["node_center_bounds"]=node_centers;result["anchor_radii"]=anchor_radii;result["node_radii"]=node_radii;result["positions"]=positions;result["partition_bounds"]=partition;result["support_bounds"]=support;result["anchor_support_bounds"]=anchor_support;return result;
    }
    static AnchorTree from_layout(const py::dict& data,I leaf_capacity,I max_depth) {
        AnchorTree tree;tree.leaf_capacity_=leaf_capacity;tree.max_depth_=max_depth;
        if(leaf_capacity<1||max_depth<1||max_depth>64)throw std::invalid_argument("Invalid saved partition settings");
        auto arr=[&](const char* name,const py::dtype& dtype,int ndim){py::array a=py::cast<py::array>(data[name]);require_array(a,dtype,ndim,name);return a;};
        py::array dfs=arr("dfs_to_row",py::dtype::of<I>(),1),rank=arr("rank_of_row",py::dtype::of<I>(),1);
        py::array positions=arr("positions",py::dtype::of<double>(),2),supports=arr("anchor_support_bounds",py::dtype::of<double>(),2);
        py::array ac=arr("anchor_center_bounds",py::dtype::of<double>(),2),nc=arr("node_center_bounds",py::dtype::of<double>(),2),ar=arr("anchor_radii",py::dtype::of<double>(),1),nr=arr("node_radii",py::dtype::of<double>(),1);
        py::array intervals=arr("intervals",py::dtype::of<I>(),2),children=arr("children",py::dtype::of<I>(),2);
        py::array partition=arr("partition_bounds",py::dtype::of<double>(),2),bounds=arr("support_bounds",py::dtype::of<double>(),2);
        I n=dfs.size(),m=intervals.shape(0);
        if(ac.shape(0)!=n||ac.shape(1)!=6||nc.shape(0)!=m||nc.shape(1)!=6||ar.size()!=n||nr.size()!=m||rank.size()!=n||positions.shape(0)!=n||positions.shape(1)!=3||supports.shape(0)!=n||supports.shape(1)!=6||
           intervals.shape(1)!=2||children.shape(0)!=m||children.shape(1)!=8||partition.shape(0)!=m||partition.shape(1)!=6||bounds.shape(0)!=m||bounds.shape(1)!=6||((n==0)!=(m==0)))
            throw std::invalid_argument("Saved anchor layout dimensions disagree");
        const I* dp=(const I*)dfs.data();const I* rp=(const I*)rank.data();const double* pp=(const double*)positions.data();const double* sp=(const double*)supports.data();
        tree.dfs_.assign(dp,dp+n);tree.rank_.assign(rp,rp+n);tree.positions_.assign(pp,pp+n*3);tree.support_.resize(n);tree.centers_.resize(n);tree.radii_.resize(n);tree.nodes_.resize(m);
        std::vector<unsigned char> seen(n,0);for(I r=0;r<n;++r){if(dp[r]<0||dp[r]>=n||seen[dp[r]]||rp[dp[r]]!=r)throw std::invalid_argument("Saved row/rank binding is not bijective");seen[dp[r]]=1;}
        for(I row=0;row<n;++row){for(int k=0;k<3;++k){if(!std::isfinite(pp[row*3+k]))throw std::invalid_argument("Saved positions are nonfinite");}
            tree.radii_[row]=((const double*)ar.data())[row];if(std::isnan(tree.radii_[row])||tree.radii_[row]<0)throw std::invalid_argument("Saved radius invalid");
            for(int k=0;k<6;++k){tree.support_[row][k]=sp[row*6+k];tree.centers_[row][k]=((const double*)ac.data())[row*6+k];}
            for(int k=0;k<3;++k)if(std::isnan(tree.centers_[row][k])||std::isnan(tree.centers_[row][k+3])||tree.centers_[row][k]>tree.centers_[row][k+3])throw std::invalid_argument("Saved center bounds invalid");
            for(int k=0;k<3;++k)if(std::isnan(sp[row*6+k])||std::isnan(sp[row*6+k+3])||sp[row*6+k]>sp[row*6+k+3])throw std::invalid_argument("Saved supports invalid");}
        const I* ip=(const I*)intervals.data();const I* cp=(const I*)children.data();const double* bp=(const double*)bounds.data();const double* pb=(const double*)partition.data();
        std::vector<I> parents(m,0);
        for(I id=0;id<m;++id){Node& node=tree.nodes_[id];node.begin=ip[id*2];node.end=ip[id*2+1];if(node.begin<0||node.begin>=node.end||node.end>n)throw std::invalid_argument("Saved node interval invalid");
            node.radius=((const double*)nr.data())[id];if(std::isnan(node.radius)||node.radius<0)throw std::invalid_argument("Saved node radius invalid");
            for(int k=0;k<6;++k){node.centers[k]=((const double*)nc.data())[id*6+k];node.partition[k]=pb[id*6+k];node.support[k]=bp[id*6+k];}
            for(int k=0;k<3;++k)if(!std::isfinite(node.partition[k])||!std::isfinite(node.partition[k+3])||node.partition[k]>node.partition[k+3]||std::isnan(node.support[k])||std::isnan(node.support[k+3])||node.support[k]>node.support[k+3])throw std::invalid_argument("Saved node bounds invalid");
            I cursor=node.begin;bool has_child=false;
            for(int k=0;k<8;++k){I child=cp[id*8+k];node.children[k]=child;if(child==-1)continue;has_child=true;
                if(child<=id||child>=m||++parents[child]!=1||ip[child*2]!=cursor)throw std::invalid_argument("Saved topology invalid");cursor=ip[child*2+1];}
            if(has_child&&cursor!=node.end)throw std::invalid_argument("Saved child intervals do not partition parent");
            for(int k=0;k<3;++k)if(std::isnan(node.centers[k])||std::isnan(node.centers[k+3])||node.centers[k]>node.centers[k+3])throw std::invalid_argument("Saved node center bounds invalid");
            for(I r=node.begin;r<node.end;++r){I row=dp[r];if(node.radius<tree.radii_[row])throw std::invalid_argument("Saved node radius does not bound descendants");for(int k=0;k<3;++k){if(tree.centers_[row][k]<node.centers[k]||tree.centers_[row][k+3]>node.centers[k+3]||pp[row*3+k]<node.partition[k]||pp[row*3+k]>node.partition[k+3]||sp[row*6+k]<node.support[k]||sp[row*6+k+3]>node.support[k+3])throw std::invalid_argument("Saved node does not enclose its descendants");}}
        }
        if(m){if(tree.nodes_[0].begin!=0||tree.nodes_[0].end!=n)throw std::invalid_argument("Saved root does not cover every row");for(I id=1;id<m;++id)if(parents[id]!=1)throw std::invalid_argument("Saved topology has unreachable nodes");}
        tree.derive_query_layout();
        return tree;
    }
    py::dict query(const py::array& fov_ids,const py::array& depth,const py::array& w2c,std::array<double,4> domain,
                   std::array<I,2> image_size,const std::string& mode,double pixel_pad,double near_z,double margin) const {
        auto start=Clock::now();require_array(fov_ids,py::dtype::of<I>(),1,"fov_ids");require_array(depth,py::dtype::of<double>(),2,"depth_bounds");
        require_array(w2c,py::dtype::of<double>(),2,"w2c");
        if(w2c.shape(0)!=4||w2c.shape(1)!=4||depth.shape(0)<1||depth.shape(1)<1||image_size[0]<1||image_size[1]<1||
           !std::isfinite(pixel_pad)||pixel_pad<0||!std::isfinite(near_z)||near_z<0||!std::isfinite(margin)||margin<0)
            throw std::invalid_argument("invalid camera/depth/certificate settings");
        for(double v:domain)if(!std::isfinite(v))throw std::invalid_argument("angular domain must be finite");
        if(domain[0]>=domain[1]||domain[2]>=domain[3])throw std::invalid_argument("angular domain must be ordered");
        const double* w=(const double*)w2c.data();for(int i=0;i<16;++i)if(!std::isfinite(w[i]))throw std::invalid_argument("w2c must be finite");
        if(w[12]!=0||w[13]!=0||w[14]!=0||w[15]!=1)throw std::invalid_argument("w2c must be affine row-major world-to-camera");
        const double* d=(const double*)depth.data();for(I i=0;i<depth.size();++i)if(std::isnan(d[i])||d[i]<=0)throw std::invalid_argument("ORI needs positive finite depth or +infinity Unknown");
        double rotation_norm_sq=0;for(int j=0;j<3;++j){double row_sum=0;for(int k=0;k<3;++k){double dot=0;for(int l=0;l<3;++l)dot+=w[j*4+l]*w[k*4+l];row_sum+=std::abs(dot);}rotation_norm_sq=std::max(rotation_norm_sq,row_sum);}
        Camera camera{w,d,depth.shape(0),depth.shape(1),domain,pixel_pad*(domain[1]-domain[0])/image_size[0],pixel_pad*(domain[3]-domain[2])/image_size[1],near_z,margin,std::sqrt(rotation_norm_sq)*(1+64*std::numeric_limits<float>::epsilon())};
        const I n=I(dfs_.size()),nf=fov_ids.shape(0);const I* ids=(const I*)fov_ids.data();
        std::vector<unsigned char> fov(n,0),removed(n,0);std::vector<I> prefix(n+1,0);
        for(I i=0;i<nf;++i){I row=ids[i];if(row<0||row>=n)throw std::invalid_argument("fov ID outside PLY rows");I rank=rank_[row];if(fov[rank])throw std::invalid_argument("duplicate fov ID");fov[rank]=1;}
        for(I i=0;i<n;++i)prefix[i+1]=prefix[i]+fov[i];double prepare_ms=ms(start);auto traversal_start=Clock::now();Stats stats;
        if(mode=="linear")for(I i=0;i<nf;++i){++stats.anchor_checks;I row=ids[i];if(certify(support_[row],centers_[row],radii_[row],known_[row],camera,stats)){removed[rank_[row]]=1;++stats.certified_anchors;}}
        else if(mode=="tree"){if(n)traverse(0,camera,prefix,fov,removed,stats);}
        else throw std::invalid_argument("mode must be tree or linear");
        double traversal_ms=ms(traversal_start);auto materialize_start=Clock::now();
        std::vector<I> selected;selected.reserve(nf);for(I i=0;i<nf;++i)if(!removed[rank_[ids[i]]])selected.push_back(ids[i]);
        std::vector<std::array<I,2>> raw;
        // Leaf boundaries give raw range ownership; formalization merges touching runs.
        if(n){for(const Node& node:nodes_){bool leaf=true;for(I child:node.children)if(child>=0){leaf=false;break;}if(!leaf||prefix[node.end]==prefix[node.begin])continue;
            I first=-1;for(I r=node.begin;r<node.end;++r){bool keep=fov[r]&&!removed[r];if(keep&&first<0)first=r;if(!keep&&first>=0){raw.push_back({first,r});first=-1;}}
            if(first>=0)raw.push_back({first,node.end});}}
        std::sort(raw.begin(),raw.end(),[](auto a,auto b){return a[0]<b[0];});
        std::vector<std::array<I,2>> formal;for(auto r:raw){if(!formal.empty()&&formal.back()[1]==r[0])formal.back()[1]=r[1];else formal.push_back(r);}
        py::array_t<I> output(I(selected.size())),raw_array({I(raw.size()),I(2)}),formal_array({I(formal.size()),I(2)});
        std::copy(selected.begin(),selected.end(),output.mutable_data());
        for(I i=0;i<I(raw.size());++i)for(int k=0;k<2;++k)raw_array.mutable_at(i,k)=raw[i][k];
        for(I i=0;i<I(formal.size());++i)for(int k=0;k<2;++k)formal_array.mutable_at(i,k)=formal[i][k];
        py::dict counters;
#define COUNTER(x) counters[#x]=stats.x
        COUNTER(visited_nodes);COUNTER(certified_nodes);COUNTER(leaf_checks);COUNTER(anchor_checks);COUNTER(empty_nodes);
        COUNTER(unbounded);COUNTER(near_plane);COUNTER(unknown_cells);COUNTER(depth_fail);COUNTER(outside);COUNTER(certified_anchors);COUNTER(cell_checks);COUNTER(early_unknown_cells);
#undef COUNTER
        counters["fov_count"]=nf;counters["selected_count"]=selected.size();counters["raw_range_count"]=raw.size();counters["formal_range_count"]=formal.size();
        py::dict timing;timing["prepare_ms"]=prepare_ms;timing["traversal_ms"]=traversal_ms;timing["materialization_ms"]=ms(materialize_start);timing["native_total_ms"]=ms(start);
        py::dict result;result["selected_anchor_ids"]=output;result["raw_ranges"]=raw_array;result["formal_ranges"]=formal_array;result["counters"]=counters;result["timings"]=timing;return result;
    }
};
PYBIND11_MODULE(_gdmgs_anchor_native,m) {
    py::class_<AnchorTree>(m,"AnchorTree").def(py::init<const py::array&,const py::array&,const py::array&,const py::array&,I,I>())
        .def("layout",&AnchorTree::layout).def("query",&AnchorTree::query)
        .def_static("from_layout", &AnchorTree::from_layout);
}
