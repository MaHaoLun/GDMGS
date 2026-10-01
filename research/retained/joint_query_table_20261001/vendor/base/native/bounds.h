#pragma once
#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
// b: world-space bounds of ALL offset centers, followed by max scale.
// c: row-major view[16], fx,fy,cx,cy,width,height, ||R||_2^2 upper bound.
// Reject only when gsplat's entire possible screen radius is outside.
HD inline bool outside(const double* b, const double* c) {
    if (!(b[6]>=0) || !std::isfinite(b[6])) return false;
    double lo[3], hi[3];
    for (int r=0;r<3;++r) {
        lo[r]=hi[r]=c[4*r+3]; double magnitude=std::fabs(c[4*r+3]);
        for(int k=0;k<3;++k) {
            double x=c[4*r+k]*b[k], y=c[4*r+k]*b[k+3];
            lo[r]+=fmin(x,y); hi[r]+=fmax(x,y);
            magnitude+=fmax(std::fabs(x),std::fabs(y));
        }
        // Enclose float32 transform roundoff and cancellation (absolute terms).
        double error=2e-5*(1+magnitude);
        lo[r]-=error; hi[r]+=error;
        if(!std::isfinite(lo[r]) || !std::isfinite(hi[r])) return false;
    }
    if(hi[2]<0.01 || lo[2]>1e10) return true;
    if(lo[2]<=0.01) return false;
    double xlo=1e300,xhi=-1e300,ylo=1e300,yhi=-1e300;
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        double z=j?hi[2]:lo[2];
        double x=(i?hi[0]:lo[0])/z, y=(i?hi[1]:lo[1])/z;
        xlo=fmin(xlo,x);xhi=fmax(xhi,x);ylo=fmin(ylo,y);yhi=fmax(yhi,y);
    }
    double tx=fmax(c[18],c[20]-c[18])/c[16]+0.15*c[20]/c[16];
    double ty=fmax(c[19],c[21]-c[19])/c[17]+0.15*c[21]/c[17];
    // trace(J Sigma J^T) bounds lambda_max. gsplat uses max(0.01,disc),
    // so add 0.1 as well as eps2d=0.3. Inflate for float32 covariance ops.
    double trace=b[6]*b[6]*c[22]*(c[16]*c[16]*(1+tx*tx)+c[17]*c[17]*(1+ty*ty))/(lo[2]*lo[2]);
    double radius=ceil(3*sqrt(trace*1.001+0.401))+2;
    if(!std::isfinite(radius)) return false;
    return c[16]*xhi+c[18]+radius<0 || c[16]*xlo+c[18]-radius>c[20]
        || c[17]*yhi+c[19]+radius<0 || c[17]*ylo+c[19]-radius>c[21];
}
