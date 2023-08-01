/** @file rodrigues.c
 ** @brief Rodrigues formulas - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-13 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "generic.h"
#include "mathop.h"
#include "rodrigues.h"

#include <math.h>

/** @brief Rodrigues' formula
 ** @param R_pt  3x3 matrix - array of 9 double (in) .
 ** @param dR_pt 9x3 matrix - array of 27 double (in).
 ** @param om_pt 3 vector - array of 3 dobule (out).
 **/

void
vl_rodrigues(double* R_pt, double* dR_pt, const double* om_pt)
{
  /*
    Let

       th = |om|,  r=w/th,
       sth=sin(th),  cth=cos(th),
       ^om = hat(om)

    Then the rodrigues formula is an expansion of the exponential
    function:

     rodrigues(om) = exp ^om = I + ^r sth + ^r^2 (1 - cth).

    The derivative can be computed by elementary means and
    results:

    d(vec rodrigues(om))    sth  d ^r    1 - cth  d (^r)^2
    -------------------- =  ---- ----- + -------  -------- +
          d om^T             th  d r^T     th      d r^T

                          sth                     1 - cth
          + vec^r (cth - -----) + vec^r^2 (sth - 2-------)r^T
                          th                         th
  */

#define OM(i)   om_pt[(i)]
#define R(i,j)  R_pt[(i)+3*(j)]
#define DR(i,j) dR_pt[(i)+9*(j)]
#undef small

  const double small = 1e-6 ;

  double th = sqrt( OM(0)*OM(0) +
                    OM(1)*OM(1) +
                    OM(2)*OM(2) ) ;

  if( th < small ) {
    R(0,0) = 1.0 ; R(0,1) = 0.0 ; R(0,2) = 0.0 ;
    R(1,0) = 0.0 ; R(1,1) = 1.0 ; R(1,2) = 0.0 ;
    R(2,0) = 0.0 ; R(2,1) = 0.0 ; R(2,2) = 1.0 ;

    if(dR_pt) {
      DR(0,0) = 0  ; DR(0,1) = 0   ; DR(0,2) = 0 ;
      DR(1,0) = 0  ; DR(1,1) = 0   ; DR(1,2) = 1 ;
      DR(2,0) = 0  ; DR(2,1) = -1  ; DR(2,2) = 0 ;

      DR(3,0) = 0  ; DR(3,1) = 0   ; DR(3,2) = -1 ;
      DR(4,0) = 0  ; DR(4,1) = 0   ; DR(4,2) = 0 ;
      DR(5,0) = 1  ; DR(5,1) = 0   ; DR(5,2) = 0 ;

      DR(6,0) = 0  ; DR(6,1) = 1   ; DR(6,2) = 0 ;
      DR(7,0) = -1 ; DR(7,1) = 0   ; DR(7,2) = 0 ;
      DR(8,0) = 0  ; DR(8,1) = 0   ; DR(8,2) = 0 ;
    }
    return ;
  }

  {
    double x = OM(0) / th ;
    double y = OM(1) / th ;
    double z = OM(2) / th ;

    double xx = x*x ;
    double xy = x*y ;
    double xz = x*z ;
    double yy = y*y ;
    double yz = y*z ;
    double zz = z*z ;

    const double yx = xy ;
    const double zx = xz ;
    const double zy = yz ;

    double sth  = sin(th) ;
    double cth  = cos(th) ;
    double mcth = 1.0 - cth ;

    R(0,0) = 1          - mcth * (yy+zz) ;
    R(1,0) =     sth*z  + mcth * xy ;
    R(2,0) =   - sth*y  + mcth * xz ;

    R(0,1) =   - sth*z  + mcth * yx ;
    R(1,1) = 1          - mcth * (zz+xx) ;
    R(2,1) =     sth*x  + mcth * yz ;

    R(0,2) =     sth*y  + mcth * xz ;
    R(1,2) =   - sth*x  + mcth * yz ;
    R(2,2) = 1          - mcth * (xx+yy) ;

    if(dR_pt) {
      double a =  sth / th ;
      double b = mcth / th ;
      double c = cth - a ;
      double d = sth - 2*b ;

      DR(0,0) =                         - d * (yy+zz) * x ;
      DR(1,0) =        b*y   + c * zx   + d * xy      * x ;
      DR(2,0) =        b*z   - c * yx   + d * xz      * x ;

      DR(3,0) =        b*y   - c * zx   + d * xy      * x ;
      DR(4,0) =     -2*b*x              - d * (zz+xx) * x ;
      DR(5,0) =  a           + c * xx   + d * yz      * x ;

      DR(6,0) =        b*z   + c * yx   + d * zx      * x ;
      DR(7,0) = -a           - c * xx   + d * zy      * x ;
      DR(8,0) =     -2*b*x              - d * (yy+xx) * x ;

      DR(0,1) =     -2*b*y              - d * (yy+zz) * y ;
      DR(1,1) =        b*x   + c * zy   + d * xy      * y ;
      DR(2,1) = -a           - c * yy   + d * xz      * y ;

      DR(3,1) =        b*x   - c * zy   + d * xy      * y ;
      DR(4,1) =                         - d * (zz+xx) * y ;
      DR(5,1) =        b*z   + c * xy   + d * yz      * y ;

      DR(6,1) = a            + c * yy   + d * zx      * y ;
      DR(7,1) =        b*z   - c * xy   + d * zy      * y ;
      DR(8,1) =     -2*b*y              - d * (yy+xx) * y ;

      DR(0,2) =     -2*b*z              - d * (yy+zz) * z ;
      DR(1,2) =  a           + c * zz   + d * xy      * z ;
      DR(2,2) =        b*x   - c * yz   + d * xz      * z ;

      DR(3,2) =  -a          - c * zz   + d * xy      * z ;
      DR(4,2) =     -2*b*z              - d * (zz+xx) * z ;
      DR(5,2) =        b*y   + c * xz   + d * yz      * z ;

      DR(6,2) =        b*x   + c * yz   + d * zx      * z ;
      DR(7,2) =        b*y   - c * xz   + d * zy      * z ;
      DR(8,2) =                         - d * (yy+xx) * z ;
    }
  }

#undef OM
#undef R
#undef DR

}

/** @brief Inverse Rodrigues formula
 ** @param om_pt  3    vector - array of 3   dobule (out).
 ** @param dom_pt 3x9  matrix - array of 3x9 dobule (out).
 ** @param R_pt   3x3  matrix - array of 9   double (in).
 **
 ** This function computes the Rodrigues formula of the argument @a
 ** om_pt. The result is stored int the matrix @a R_pt. If @a dR_pt is
 ** non null, then the derivative of the Rodrigues formula is computed
 ** and stored into the matrix @a dR_pt.
 **/

VL_EXPORT
void vl_irodrigues(double* om_pt, double* dom_pt, const double* R_pt)
{
  /*
                    tr R - 1          1    [ R32 - R23 ]
      th = cos^{-1} --------,  r =  ------ [ R13 - R31 ], w = th r.
                        2           2 sth  [ R12 - R21 ]

      sth = sin(th)

       dw    th*cth-sth      dw     th   [di3 dj2 - di2 dj3]
      ---- = ---------- r,  ---- = ----- [di1 dj3 - di3 dj1].
      dRii     2 sth^2      dRij   2 sth [di1 dj2 - di2 dj1]

      trace(A) < -1 only for small num. errors.
  */

#define OM(i)    om_pt[(i)]
#define DOM(i,j) dom_pt[(i)+3*(j)]
#define R(i,j)   R_pt[(i)+3*(j)]
#define W(i,j)   W_pt[(i)+3*(j)]

  const double small = 1e-6 ;

  double th = acos
    (0.5*(VL_MAX(R(0,0)+R(1,1)+R(2,2),-1.0) - 1.0)) ;

  double sth = sin(th) ;
  double cth = cos(th) ;

  if(fabs(sth) < small && cth < 0) {
    /*
      we have this singularity when the rotation  is about pi (or -pi)
      we use the fact that in this case

      hat( sqrt(1-cth) * r )^2 = W = (0.5*(R+R') - eye(3))

      which gives

      (1-cth) rx^2 = 0.5 * (W(1,1)-W(2,2)-W(3,3))
      (1-cth) ry^2 = 0.5 * (W(2,2)-W(3,3)-W(1,1))
      (1-cth) rz^2 = 0.5 * (W(3,3)-W(1,1)-W(2,2))
    */

    double W_pt [9], x, y, z ;
    W_pt[0] = 0.5*( R(0,0) + R(0,0) ) - 1.0 ;
    W_pt[1] = 0.5*( R(1,0) + R(0,1) ) ;
    W_pt[2] = 0.5*( R(2,0) + R(0,2) );

    W_pt[3] = 0.5*( R(0,1) + R(1,0) );
    W_pt[4] = 0.5*( R(1,1) + R(1,1) ) - 1.0;
    W_pt[5] = 0.5*( R(2,1) + R(1,2) );

    W_pt[6] =  0.5*( R(0,2) + R(2,0) ) ;
    W_pt[7] =  0.5*( R(1,2) + R(2,1) ) ;
    W_pt[8] =  0.5*( R(2,2) + R(2,2) ) - 1.0 ;

    /* these are only absolute values */
    x = sqrt( 0.5 * (W(0,0)-W(1,1)-W(2,2)) ) ;
    y = sqrt( 0.5 * (W(1,1)-W(2,2)-W(0,0)) ) ;
    z = sqrt( 0.5 * (W(2,2)-W(0,0)-W(1,1)) ) ;

    /* set the biggest component to + and use the element of the
    ** matrix W to determine the sign of the other components
    ** then the solution is either (x,y,z) or its opposite */
    if( x >= y && x >= z ) {
      y = (W(1,0) >=0) ? y : -y ;
      z = (W(2,0) >=0) ? z : -z ;
    } else if( y >= x && y >= z ) {
      z = (W(2,1) >=0) ? z : -z ;
      x = (W(1,0) >=0) ? x : -x ;
    } else {
      x = (W(2,0) >=0) ? x : -x ;
      y = (W(2,1) >=0) ? y : -y ;
    }

    /* we are left to chose between (x,y,z) and (-x,-y,-z)
    ** unfortunately we cannot (as the rotation is too close to pi) and
    ** we just keep what we have. */
    {
      double scale = th / sqrt( 1 - cth ) ;
      OM(0) = scale * x ;
      OM(1) = scale * y ;
      OM(2) = scale * z ;

      if( dom_pt ) {
        int k ;
        for(k=0; k<3*9; ++k)
          dom_pt [k] = VL_NAN_D ;
      }
      return ;
    }

  } else {
    double a = (fabs(sth) < small) ? 1 : th/sin(th) ;
    double b ;
    OM(0) = 0.5*a*(R(2,1) - R(1,2)) ;
    OM(1) = 0.5*a*(R(0,2) - R(2,0)) ;
    OM(2) = 0.5*a*(R(1,0) - R(0,1)) ;

    if( dom_pt ) {
      if( fabs(sth) < small ) {
        a = 0.5 ;
        b = 0 ;
      } else {
        a = th/(2*sth) ;
        b = (th*cth - sth)/(2*sth*sth)/th ;
      }

      DOM(0,0) = b*OM(0) ;
      DOM(1,0) = b*OM(1) ;
      DOM(2,0) = b*OM(2) ;

      DOM(0,1) = 0 ;
      DOM(1,1) = 0 ;
      DOM(2,1) = a ;

      DOM(0,2) = 0 ;
      DOM(1,2) = -a ;
      DOM(2,2) = 0 ;

      DOM(0,3) = 0 ;
      DOM(1,3) = 0 ;
      DOM(2,3) = -a ;

      DOM(0,4) = b*OM(0) ;
      DOM(1,4) = b*OM(1) ;
      DOM(2,4) = b*OM(2) ;

      DOM(0,5) = a ;
      DOM(1,5) = 0 ;
      DOM(2,5) = 0 ;

      DOM(0,6) = 0 ;
      DOM(1,6) = a ;
      DOM(2,6) = 0 ;

      DOM(0,7) = -a ;
      DOM(1,7) = 0 ;
      DOM(2,7) = 0 ;

      DOM(0,8) = b*OM(0) ;
      DOM(1,8) = b*OM(1) ;
      DOM(2,8) = b*OM(2) ;
    }
  }

#undef OM
#undef DOM
#undef R
#undef W
}
