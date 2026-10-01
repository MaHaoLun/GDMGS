#pragma once
#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
// Same conservative envelope equations as classify.h. Return 0 Cull,
// 2 Descend/uncertain, or 3 for the six-lane plane reduction.
HD inline int support_envelope(const double* b,const double* c,double* e) {
    if(!(b[6]>=0)||!std::isfinite(b[6]))return 2;
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
    double ex=radius*hi[2]/c[16],ey=radius*hi[2]/c[17];
    if(!std::isfinite(ex)||!std::isfinite(ey))return 2;
    e[0]=lo[0]-ex;e[1]=hi[0]+ex;
    e[2]=lo[1]-ey;e[3]=hi[1]+ey;
    e[4]=lo[2];e[5]=hi[2];
    return 3;
}
HD inline void plane_interval(int lane,const double* e,const double* c,double &minimum,double &maximum) {
    double zl=e[4],zh=e[5];
    switch(lane) {
    case 0: minimum=e[0]+c[18]/c[16]*zl;maximum=e[1]+c[18]/c[16]*zh;break;
    case 1: minimum=-e[1]+(c[20]-c[18])/c[16]*zl;maximum=-e[0]+(c[20]-c[18])/c[16]*zh;break;
    case 2: minimum=e[2]+c[19]/c[17]*zl;maximum=e[3]+c[19]/c[17]*zh;break;
    case 3: minimum=-e[3]+(c[21]-c[19])/c[17]*zl;maximum=-e[2]+(c[21]-c[19])/c[17]*zh;break;
    case 4: minimum=zl-0.01;maximum=zh-0.01;break;
    case 5: minimum=1e10-zh;maximum=1e10-zl;break;
    default: minimum=0;maximum=0;
    }
}
