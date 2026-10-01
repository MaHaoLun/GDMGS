#pragma once
#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
// 0=Cull, 1=Keep, 2=Descend. The input is a camera-dependent conservative
// envelope for all descendant Gaussian *screen footprints*, not center-only.
HD inline int classify_six_planes(const double* b,const double* c) {
    if (!(b[6]>=0) || !std::isfinite(b[6]))return 2;
    double lo[3],hi[3];
    for(int r=0;r<3;++r) {
        lo[r]=hi[r]=c[4*r+3];double mag=std::fabs(c[4*r+3]);
        for(int k=0;k<3;++k) {
            double x=c[4*r+k]*b[k],y=c[4*r+k]*b[k+3];
            lo[r]+=fmin(x,y);hi[r]+=fmax(x,y);mag+=fmax(std::fabs(x),std::fabs(y));
        }
        double err=2e-5*(1+mag);lo[r]-=err;hi[r]+=err;
        if(!std::isfinite(lo[r])||!std::isfinite(hi[r]))return 2;
    }
    if(hi[2]<0.01 || lo[2]>1e10)return 0;
    if(lo[2]<=0.01)return 2;
    double tx=fmax(c[18],c[20]-c[18])/c[16]+0.15*c[20]/c[16];
    double ty=fmax(c[19],c[21]-c[19])/c[17]+0.15*c[21]/c[17];
    double trace=b[6]*b[6]*c[22]*(c[16]*c[16]*(1+tx*tx)+c[17]*c[17]*(1+ty*ty))/(lo[2]*lo[2]);
    double radius=ceil(3*sqrt(trace*1.001+0.401))+2;
    if(!std::isfinite(radius))return 2;
    // A projected circular footprint of radius r pixels at z<=zmax fits in
    // this expanded camera-space AABB. Test its six frustum plane intervals.
    double ex=radius*hi[2]/c[16],ey=radius*hi[2]/c[17];
    if(!std::isfinite(ex)||!std::isfinite(ey))return 2;
    double xmin=lo[0]-ex,xmax=hi[0]+ex,ymin=lo[1]-ey,ymax=hi[1]+ey;
    double minv[6]={xmin+c[18]/c[16]*lo[2],-xmax+(c[20]-c[18])/c[16]*lo[2],
                    ymin+c[19]/c[17]*lo[2],-ymax+(c[21]-c[19])/c[17]*lo[2],
                    lo[2]-0.01,1e10-hi[2]};
    double maxv[6]={xmax+c[18]/c[16]*hi[2],-xmin+(c[20]-c[18])/c[16]*hi[2],
                    ymax+c[19]/c[17]*hi[2],-ymin+(c[21]-c[19])/c[17]*hi[2],
                    hi[2]-0.01,1e10-lo[2]};
    bool fully_inside=true;
    for(int k=0;k<6;++k) {
        if(!std::isfinite(minv[k])||!std::isfinite(maxv[k]))return 2;
        if(maxv[k]<0)return 0;
        if(minv[k]<0)fully_inside=false;
    }
    return fully_inside?1:2;
}
