//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSift.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8*2+1];
__constant__ float d_ScaleDownKernel[5]; 
__constant__ float d_LowPassKernel[2*LOWPASS_R+1]; 
__constant__ float d_LaplaceKernel[8*12*16]; 

///////////////////////////////////////////////////////////////////////////////
// Lowpass filter and subsample image
///////////////////////////////////////////////////////////////////////////////
__global__ void ScaleDownDenseShift(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
#define BW (SCALEDOWN_W+4)
#define BH (SCALEDOWN_H+4)
#define W2 (SCALEDOWN_W/2)
#define H2 (SCALEDOWN_H/2)
  __shared__ float brows[BH*BW];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*SCALEDOWN_W + tx;
  const int yp = blockIdx.y*SCALEDOWN_H + ty;
  const float k0 = d_ScaleDownKernel[0];
  const float k1 = d_ScaleDownKernel[1];
  const float k2 = d_ScaleDownKernel[2];
  const int xl = min(width-1,  max(0, xp-2));
  const int yl = min(height-1, max(0, yp-2));
  if (xp<(width+4) && yp<(height+4)) {
    float v = d_Data[yl*pitch + xl];
    brows[BW*ty + tx]  = k0*(v + ShiftDown(v, 4)) + k1*(ShiftDown(v, 1) + ShiftDown(v, 3)) + k2*ShiftDown(v, 2);
  }
  __syncthreads();
  const int xs = blockIdx.x*W2 + tx;
  const int ys = blockIdx.y*H2 + ty;
  if (tx<W2 && ty<H2 && xs<(width/2) && ys<(height/2)) {
    float *ptr = &brows[BW*(ty*2) + (tx*2)];
    d_Result[ys*newpitch + xs] = k0*(ptr[0] + ptr[4*BW]) + k1*(ptr[1*BW] + ptr[3*BW]) + k2*ptr[2*BW];
  } 
}

__global__ void ScaleDownDense(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
#define BW (SCALEDOWN_W+4)
#define BH (SCALEDOWN_H+4)
#define W2 (SCALEDOWN_W/2)
#define H2 (SCALEDOWN_H/2)
  __shared__ float irows[BH*BW]; 
  __shared__ float brows[BH*W2];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*SCALEDOWN_W + tx;
  const int yp = blockIdx.y*SCALEDOWN_H + ty;
  const int xl = min(width-1,  max(0, xp-2));
  const int yl = min(height-1, max(0, yp-2));
  const float k0 = d_ScaleDownKernel[0];
  const float k1 = d_ScaleDownKernel[1];
  const float k2 = d_ScaleDownKernel[2];
  if (xp<(width+4) && yp<(height+4))
    irows[BW*ty + tx] = d_Data[yl*pitch + xl];
  __syncthreads();
  if (yp<(height+4) && tx<W2) {
    float *ptr = &irows[BW*ty + 2*tx];
    brows[W2*ty + tx] = k0*(ptr[0] + ptr[4]) + k1*(ptr[1] + ptr[3]) + k2*ptr[2];
  }
  __syncthreads();
  const int xs = blockIdx.x*W2 + tx;
  const int ys = blockIdx.y*H2 + ty;
  if (tx<W2 && ty<H2 && xs<(width/2) && ys<(height/2)) {
    float *ptr = &brows[W2*(ty*2) + tx];
    d_Result[ys*newpitch + xs] = k0*(ptr[0] + ptr[4*W2]) + k1*(ptr[1*W2] + ptr[3*W2]) + k2*ptr[2*W2];
  } 
}

__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W+4]; 
  __shared__ float brow[5*(SCALEDOWN_W/2)];
  __shared__ int yRead[SCALEDOWN_H+4];
  __shared__ int yWrite[SCALEDOWN_H+4];
  #define dx2 (SCALEDOWN_W/2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0*dx2;
  const int tx1 = tx + 1*dx2;
  const int tx2 = tx + 2*dx2;
  const int tx3 = tx + 3*dx2;
  const int tx4 = tx + 4*dx2;
  const int xStart = blockIdx.x*SCALEDOWN_W;
  const int yStart = blockIdx.y*SCALEDOWN_H;
  const int xWrite = xStart/2 + tx;
  float k0 = d_ScaleDownKernel[0];
  float k1 = d_ScaleDownKernel[1];
  float k2 = d_ScaleDownKernel[2];
  if (tx<SCALEDOWN_H+4) {
    int y = yStart + tx - 2; 
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = y*pitch;
    yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - 2;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);

  int maxtx = min(dx2, width/2 - xStart/2);
  for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
    {
      inrow[tx] = d_Data[yRead[dy+0] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx4] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=4 && !(dy&1))
	  d_Result[yWrite[dy+0] + xWrite] = k2*brow[tx2] + k0*(brow[tx0]+brow[tx4]) + k1*(brow[tx1]+brow[tx3]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+3)) {
      inrow[tx] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx0] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=3 && (dy&1))
	  d_Result[yWrite[dy+1] + xWrite] = k2*brow[tx3] + k0*(brow[tx1]+brow[tx0]) + k1*(brow[tx2]+brow[tx4]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+2)) {
      inrow[tx] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx1] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=2 && !(dy&1))
	  d_Result[yWrite[dy+2] + xWrite] = k2*brow[tx4] + k0*(brow[tx2]+brow[tx1]) + k1*(brow[tx3]+brow[tx0]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+1)) {
      inrow[tx] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx2] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=1 && (dy&1))
	  d_Result[yWrite[dy+3] + xWrite] = k2*brow[tx0] + k0*(brow[tx3]+brow[tx2]) + k1*(brow[tx4]+brow[tx1]);
      }
      __syncthreads();
    }
    if (dy<SCALEDOWN_H) {
      inrow[tx] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<dx2 && xWrite<width/2) {
	brow[tx3] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (!(dy&1))
	  d_Result[yWrite[dy+4] + xWrite] = k2*brow[tx1] + k0*(brow[tx4]+brow[tx3]) + k1*(brow[tx0]+brow[tx2]);
      }
      __syncthreads();
    }
  }
}

__global__ void ScaleUp(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int x = blockIdx.x*SCALEUP_W + 2*tx;
  int y = blockIdx.y*SCALEUP_H + 2*ty;
  if (x<2*width && y<2*height) {
    int xl = blockIdx.x*(SCALEUP_W/2) + tx;
    int yu = blockIdx.y*(SCALEUP_H/2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu*pitch + xl];
    float vur = d_Data[yu*pitch + xr];
    float vdl = d_Data[yd*pitch + xl];
    float vdr = d_Data[yd*pitch + xr];
    d_Result[(y + 0)*newpitch + x + 0] = vul;
    d_Result[(y + 0)*newpitch + x + 1] = 0.50f*(vul + vur);
    d_Result[(y + 1)*newpitch + x + 0] = 0.50f*(vul + vdl);
    d_Result[(y + 1)*newpitch + x + 1] = 0.25f*(vul + vur + vdl + vdr);
  }
}

__global__ void ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina; 
  float scosa = scale*cosa;

  for (int y=ty;y<16;y+=8) {
    float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
    float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
      tex2D<float>(texObj, xpos-cosa, ypos-sina);
    float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
      tex2D<float>(texObj, xpos+sina, ypos-cosa);
    float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
    float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;
    
    int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
    float horf = (tx - 1.5f)/4.0f - hori;
    float ihorf = 1.0f - horf;           
    int veri = (y + 2)/4 - 1;
    float verf = (y - 1.5f)/4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi<7 ? angi+1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;
    
    int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
    int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx>=2) { 
      float grad1 = ihorf*grad;
      if (y>=2) {   // Upper left
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1, iangf*grad2);
	atomicAdd(buffer + p2,  angf*grad2);
      }
      if (y<=13) {  // Lower left
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+32, iangf*grad2); 
	atomicAdd(buffer + p2+32,  angf*grad2);
      }
    }
    if (tx<=13) { 
      float grad1 = horf*grad;
      if (y>=2) {    // Upper right
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1+8, iangf*grad2);
	atomicAdd(buffer + p2+8,  angf*grad2);
      }
      if (y<=13) {   // Lower right
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+40, iangf*grad2);
	atomicAdd(buffer + p2+40,  angf*grad2);
      }
    }
  }
  __syncthreads();

  // Normalize twice and suppress peaks first time
  float sum = buffer[idx]*buffer[idx];
  for (int i=16;i>0;i/=2)
    sum += ShiftDown(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
  tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);
  
  sum = tsum1*tsum1; 
  for (int i=16;i>0;i/=2)
    sum += ShiftDown(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();

  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
  float *desc = d_sift[bx].data;
  desc[idx] = tsum1 * rsqrtf(tsum2);
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}

__device__ float FastAtan2(float y, float x)
{
  float absx = abs(x);
  float absy = abs(y);
  float a = __fdiv_rn(min(absx, absy),  max(absx, absy));
  float s = a*a;
  float r = ((-0.0464964749f*s + 0.15931422f)*s - 0.327622764f)*s*a + a;
  r = (absy>absx ? 1.57079637f - r : r);
  r = (x<0 ? 3.14159274f - r : r);
  r = (y<0 ? -r : r);
  return r;
}
       
__global__ void ExtractSiftDescriptorsCONSTNew(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  if (ty==0)
    gauss[tx] = __expf(-(tx-7.5f)*(tx-7.5f)/128.0f);

  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+1], d_MaxNumPoints);
  //if (tx==0 && ty==0)
  //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts); 
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
    
    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
    float sina = __sinf(theta);           // cosa -sina
    float cosa = __cosf(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_sift[bx].scale;
    float ssina = scale*sina; 
    float scosa = scale*cosa;
    
    for (int y=ty;y<16;y+=8) {
      float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f; 
      float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
      float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
	tex2D<float>(texObj, xpos-cosa, ypos-sina);
      float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
	tex2D<float>(texObj, xpos+sina, ypos-cosa);
      float grad = gauss[y]*gauss[tx] * __fsqrt_rn(dx*dx + dy*dy);
      float angf = 4.0f/3.1415f*FastAtan2(dy, dx) + 4.0f;
      
      int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
      float horf = (tx - 1.5f)/4.0f - hori;
      float ihorf = 1.0f - horf;           
      int veri = (y + 2)/4 - 1;
      float verf = (y - 1.5f)/4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi<7 ? angi+1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;
      
      int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
      int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx>=2) { 
	float grad1 = ihorf*grad;
	if (y>=2) {   // Upper left
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1, iangf*grad2);
	  atomicAdd(buffer + p2,  angf*grad2);
	}
	if (y<=13) {  // Lower left
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+32, iangf*grad2); 
	  atomicAdd(buffer + p2+32,  angf*grad2);
	}
      }
      if (tx<=13) { 
	float grad1 = horf*grad;
	if (y>=2) {    // Upper right
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1+8, iangf*grad2);
	  atomicAdd(buffer + p2+8,  angf*grad2);
	}
	if (y<=13) {   // Lower right
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+40, iangf*grad2);
	  atomicAdd(buffer + p2+40,  angf*grad2);
	}
      }
    }
    __syncthreads();
    
    // Normalize twice and suppress peaks first time
    float sum = buffer[idx]*buffer[idx];
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
    tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);
     
    sum = tsum1*tsum1; 
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    
    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * rsqrtf(tsum2);
    if (idx==0) {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    __syncthreads();
  }
}
 

__global__ void ExtractSiftDescriptorsCONST(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);

  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+1], d_MaxNumPoints);
  //if (tx==0 && ty==0)
  //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts); 
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
    
    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
    float sina = sinf(theta);           // cosa -sina
    float cosa = cosf(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_sift[bx].scale;
    float ssina = scale*sina; 
    float scosa = scale*cosa;
    
    for (int y=ty;y<16;y+=8) {
      float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f; 
      float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
      float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
	tex2D<float>(texObj, xpos-cosa, ypos-sina);
      float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
	tex2D<float>(texObj, xpos+sina, ypos-cosa);
      float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
      float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;
      
      int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
      float horf = (tx - 1.5f)/4.0f - hori;
      float ihorf = 1.0f - horf;           
      int veri = (y + 2)/4 - 1;
      float verf = (y - 1.5f)/4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi<7 ? angi+1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;
      
      int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
      int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx>=2) { 
	float grad1 = ihorf*grad;
	if (y>=2) {   // Upper left
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1, iangf*grad2);
	  atomicAdd(buffer + p2,  angf*grad2);
	}
	if (y<=13) {  // Lower left
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+32, iangf*grad2); 
	  atomicAdd(buffer + p2+32,  angf*grad2);
	}
      }
      if (tx<=13) { 
	float grad1 = horf*grad;
	if (y>=2) {    // Upper right
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1+8, iangf*grad2);
	  atomicAdd(buffer + p2+8,  angf*grad2);
	}
	if (y<=13) {   // Lower right
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+40, iangf*grad2);
	  atomicAdd(buffer + p2+40,  angf*grad2);
	}
      }
    }
    __syncthreads();
    
    // Normalize twice and suppress peaks first time
    float sum = buffer[idx]*buffer[idx];
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
    tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);
     
    sum = tsum1*tsum1; 
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    
    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * rsqrtf(tsum2);
    if (idx==0) {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    __syncthreads();
  }
}
 

__global__ void ExtractSiftDescriptorsOld(cudaTextureObject_t texObj, SiftPoint *d_sift, int fstPts, float subsampling)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[128];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  const int bx = blockIdx.x + fstPts;  // 0 -> numPts
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina; 
  float scosa = scale*cosa;

  for (int y=ty;y<16;y+=8) {
    float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
    float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
      tex2D<float>(texObj, xpos-cosa, ypos-sina);
    float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
      tex2D<float>(texObj, xpos+sina, ypos-cosa);
    float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
    float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;
    
    int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
    float horf = (tx - 1.5f)/4.0f - hori;  
    float ihorf = 1.0f - horf;           
    int veri = (y + 2)/4 - 1;
    float verf = (y - 1.5f)/4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi<7 ? angi+1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;
    
    int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
    int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx>=2) { 
      float grad1 = ihorf*grad;
      if (y>=2) {   // Upper left
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1, iangf*grad2);
	atomicAdd(buffer + p2,  angf*grad2);
      }
      if (y<=13) {  // Lower left
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+32, iangf*grad2); 
	atomicAdd(buffer + p2+32,  angf*grad2);
      }
    }
    if (tx<=13) { 
      float grad1 = horf*grad;
      if (y>=2) {    // Upper right
        float grad2 = iverf*grad1;
	atomicAdd(buffer + p1+8, iangf*grad2);
	atomicAdd(buffer + p2+8,  angf*grad2);
      }
      if (y<=13) {   // Lower right
        float grad2 = verf*grad1;
	atomicAdd(buffer + p1+40, iangf*grad2);
	atomicAdd(buffer + p2+40,  angf*grad2);
      }
    }
  }
  __syncthreads();

  // Normalize twice and suppress peaks first time
  if (idx<64)
    sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
  __syncthreads();      
  if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
  __syncthreads();      
  if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
  __syncthreads();      
  if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
  __syncthreads();      
  if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
  __syncthreads();      
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
  buffer[idx] = buffer[idx] * rsqrtf(tsum1);

  if (buffer[idx]>0.2f)
    buffer[idx] = 0.2f;
  __syncthreads();
  if (idx<64)
    sums[idx] = buffer[idx]*buffer[idx] + buffer[idx+64]*buffer[idx+64];
  __syncthreads();      
  if (idx<32) sums[idx] = sums[idx] + sums[idx+32];
  __syncthreads();      
  if (idx<16) sums[idx] = sums[idx] + sums[idx+16];
  __syncthreads();      
  if (idx<8)  sums[idx] = sums[idx] + sums[idx+8];
  __syncthreads();      
  if (idx<4)  sums[idx] = sums[idx] + sums[idx+4];
  __syncthreads();      
  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3]; 

  float *desc = d_sift[bx].data;
  desc[idx] = buffer[idx] * rsqrtf(tsum2);
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
}


__device__ void ExtractSiftDescriptor(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave, int bx)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int idx = threadIdx.x;
  const int tx = idx & 15; // 0 -> 16
  const int ty = idx / 16; // 0 -> 8
  if (ty==0)
    gauss[tx] = exp(-(tx-7.5f)*(tx-7.5f)/128.0f);
  buffer[idx] = 0.0f;
  __syncthreads();

  // Compute angles and gradients
  float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
  float sina = sinf(theta);           // cosa -sina
  float cosa = cosf(theta);           // sina  cosa
  float scale = 12.0f/16.0f*d_sift[bx].scale;
  float ssina = scale*sina; 
  float scosa = scale*cosa;
  
  for (int y=ty;y<16;y+=8) {
    float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f;
    float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
    float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
      tex2D<float>(texObj, xpos-cosa, ypos-sina);
    float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
      tex2D<float>(texObj, xpos+sina, ypos-cosa);
    float grad = gauss[y]*gauss[tx] * sqrtf(dx*dx + dy*dy);
    float angf = 4.0f/3.1415f*atan2f(dy, dx) + 4.0f;
    
    int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
    float horf = (tx - 1.5f)/4.0f - hori;
    float ihorf = 1.0f - horf;           
    int veri = (y + 2)/4 - 1;
    float verf = (y - 1.5f)/4.0f - veri;
    float iverf = 1.0f - verf;
    int angi = angf;
    int angp = (angi<7 ? angi+1 : 0);
    angf -= angi;
    float iangf = 1.0f - angf;
    
    int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
    int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
    int p2 = angp + hist;
    if (tx>=2) { 
      float grad1 = ihorf*grad;
      if (y>=2) {   // Upper left
	float grad2 = iverf*grad1;
	atomicAdd(buffer + p1, iangf*grad2);
	atomicAdd(buffer + p2,  angf*grad2);
      }
      if (y<=13) {  // Lower left
	float grad2 = verf*grad1;
	atomicAdd(buffer + p1+32, iangf*grad2); 
	atomicAdd(buffer + p2+32,  angf*grad2);
      }
    }
    if (tx<=13) { 
      float grad1 = horf*grad;
      if (y>=2) {    // Upper right
	float grad2 = iverf*grad1;
	atomicAdd(buffer + p1+8, iangf*grad2);
	atomicAdd(buffer + p2+8,  angf*grad2);
      }
      if (y<=13) {   // Lower right
	float grad2 = verf*grad1;
	atomicAdd(buffer + p1+40, iangf*grad2);
	atomicAdd(buffer + p2+40,  angf*grad2);
      }
    }
  }
  __syncthreads();
    
  // Normalize twice and suppress peaks first time
  float sum = buffer[idx]*buffer[idx];
  for (int i=16;i>0;i/=2)
    sum += ShiftDown(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();
  float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
  tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);
  
  sum = tsum1*tsum1; 
  for (int i=16;i>0;i/=2)
    sum += ShiftDown(sum, i);
  if ((idx&31)==0)
    sums[idx/32] = sum;
  __syncthreads();
  
  float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
  float *desc = d_sift[bx].data;
  desc[idx] = tsum1 * rsqrtf(tsum2);
  if (idx==0) {
    d_sift[bx].xpos *= subsampling;
    d_sift[bx].ypos *= subsampling;
    d_sift[bx].scale *= subsampling;
  }
  __syncthreads();
}


__global__ void RescalePositions(SiftPoint *d_sift, int numPts, float scale)
{
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  if (num<numPts) {
    d_sift[num].xpos *= scale;
    d_sift[num].ypos *= scale;
    d_sift[num].scale *= scale;
  }
}


__global__ void ComputeOrientations(cudaTextureObject_t texObj, SiftPoint *d_Sift, int fstPts)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;
  const int bx = blockIdx.x + fstPts;
  float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
  if (tx<11) 
    gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
  if (tx<64)
    hist[tx] = 0.0f;
  __syncthreads();
  float xp = d_Sift[bx].xpos - 4.5f;
  float yp = d_Sift[bx].ypos - 4.5f;
  int yd = tx/11;
  int xd = tx - yd*11;
  float xf = xp + xd;
  float yf = yp + yd;
  if (yd<11) {
    float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf); 
    float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0); 
    int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
    if (bin>31)
      bin = 0;
    float grad = sqrtf(dx*dx + dy*dy);
    atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
  }
  __syncthreads();
  int x1m = (tx>=1 ? tx-1 : tx+31);
  int x1p = (tx<=30 ? tx+1 : tx-31);
  if (tx<32) {
    int x2m = (tx>=2 ? tx-2 : tx+30);
    int x2p = (tx<=29 ? tx+2 : tx-30);
    hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
  }
  __syncthreads();
  if (tx<32) {
    float v = hist[32+tx];
    hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
  }
  __syncthreads();
  if (tx==0) {
    float maxval1 = 0.0;
    float maxval2 = 0.0;
    int i1 = -1;
    int i2 = -1;
    for (int i=0;i<32;i++) {
      float v = hist[i];
      if (v>maxval1) {
	maxval2 = maxval1;
	maxval1 = v;
	i2 = i1;
	i1 = i;
      } else if (v>maxval2) {
	maxval2 = v;
	i2 = i;
      }
    }
    float val1 = hist[32+((i1+1)&31)];
    float val2 = hist[32+((i1+31)&31)];
    float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
    d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
    if (maxval2>0.8f*maxval1) {
      float val1 = hist[32+((i2+1)&31)];
      float val2 = hist[32+((i2+31)&31)];
      float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
      unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
      if (idx<d_MaxNumPoints) {
	d_Sift[idx].xpos = d_Sift[bx].xpos;
	d_Sift[idx].ypos = d_Sift[bx].ypos;
	d_Sift[idx].scale = d_Sift[bx].scale;
	d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
	d_Sift[idx].subsampling = d_Sift[bx].subsampling;
      }
    } 
  }
} 

// With constant number of blocks
__global__ void ComputeOrientationsCONSTNew(float *image, int w, int p, int h, SiftPoint *d_Sift, int octave)
{
#define RAD 9
#define WID (2*RAD + 1)
#define LEN 32                                   //%%%% Note: Lowe suggests 36, not 32
  __shared__ float img[WID][WID], tmp[WID][WID];
  __shared__ float hist[2*LEN];
  __shared__ float gaussx[WID], gaussy[WID];
  const int tx = threadIdx.x;
  
  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+0], d_MaxNumPoints);  
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

    float sc = d_Sift[bx].scale;
    for (int i=tx;i<2*LEN;i+=blockDim.x)
      hist[i] = 0.0f;
    float xp = d_Sift[bx].xpos;
    float yp = d_Sift[bx].ypos;
    int xi = (int)xp;
    int yi = (int)yp;
    float xf = xp - xi;
    float yf = yp - yi;
    for (int i=tx;i<WID*WID;i+=blockDim.x) {
      int y = i/WID;
      int x = i - y*WID;
      int xp = max(min(x - RAD + xi, w - 1), 0);
      int yp = max(min(y - RAD + yi, h - 1), 0);
      img[y][x] = image[yp*p + xp];
    }
    float fac[5];
    fac[1] = fac[3] = (sc>0.5f ? __expf(-1.0f/(2.0f*(sc*sc - 0.25f))) : 0.0f);
    fac[0] = fac[4] = (sc>0.5f ? __expf(-4.0f/(2.0f*(sc*sc - 0.25f))) : 0.0f);
    fac[2] = 1.0f;
    float i2sigma2 = -1.0f/(2.0f*2.0f*2.0f*sc*sc); //%%%% Note: Lowe suggests 1.5, not 2.0
    if (tx<WID) {
      gaussx[tx] = __expf(i2sigma2*(tx-RAD-xf)*(tx-RAD-xf));
      gaussy[tx] = __expf(i2sigma2*(tx-RAD-yf)*(tx-RAD-yf));
    }
    __syncthreads();
    for (int i=tx;i<(WID-4)*WID;i+=blockDim.x) {
      int y = i/WID;
      int x = i - y*WID;
      y += 2;
      tmp[y][x] = img[y][x] + fac[1]*(img[y-1][x] + img[y+1][x]) +
	fac[0]*(img[y-2][x] + img[y+2][x]);
    }
    __syncthreads();
    for (int i=tx;i<(WID-4)*(WID-4);i+=blockDim.x) {
      int y = i/(WID-4);
      int x = i - y*(WID-4);
      x += 2;
      y += 2;
      img[y][x] = tmp[y][x] + fac[1]*(tmp[y][x-1] + tmp[y][x+1]) +
	fac[0]*(tmp[y][x-2] + tmp[y][x+2]);
    }
    __syncthreads();
    for (int i=tx;i<(WID-6)*(WID-6);i+=blockDim.x) {
      int y = i/(WID-6);
      int x = i - y*(WID-6);
      x += 3;
      y += 3;
      float dx = img[y][x+1] - img[y][x-1];
      float dy = img[y+1][x] - img[y-1][x];
      int bin = (int)((LEN/2)*atan2f(dy, dx)/3.1416f + (LEN/2) + 0.5f)%LEN;
      float grad = __fsqrt_rn(dx*dx + dy*dy);
      atomicAdd(&hist[LEN + bin], grad*gaussx[x]*gaussy[y]); 
    }
    __syncthreads();
    int x1m = (tx>=1 ? tx-1 : tx+LEN-1);
    int x1p = (tx<(LEN-1) ? tx+1 : tx-LEN+1);
    int x2m = (tx>=2 ? tx-2 : tx+LEN-2);
    int x2p = (tx<(LEN-2) ? tx+2 : tx-LEN+2);
    if (tx<LEN) {
      hist[tx] = 6.0f*hist[tx + LEN] + 4.0f*(hist[x1m + LEN] + hist[x1p + LEN]) +
	1.0f*(hist[x2m + LEN] + hist[x2p + LEN]);
      hist[tx + LEN] = 8.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) +
	0.0f*(hist[x2m] + hist[x2p]);
      float val = hist[tx + LEN];
      hist[tx] = (val>hist[x1m + LEN] && val>=hist[x1p + LEN] ? val : 0.0f);
    }
    __syncthreads();
    if (tx==0) {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i=0;i<LEN;i++) {
	float v = hist[i];
	if (v>maxval1) {
	  maxval2 = maxval1;
	  maxval1 = v;
	  i2 = i1;
	  i1 = i;
	} else if (v>maxval2) {
	  maxval2 = v;
	  i2 = i;
	}
      }
      float val1 = hist[LEN + ((i1 + 1)%LEN)];
      float val2 = hist[LEN + ((i1 + LEN - 1)%LEN)];
      float peak = i1 + 0.5f*(val1 - val2) / (2.0f*maxval1 - val1 - val2);
      d_Sift[bx].orientation = 360.0f*(peak<0.0f ? peak + LEN : peak)/LEN;
      atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave+0]); 
      if (maxval2>0.8f*maxval1 && true) {
	float val1 = hist[LEN + ((i2 + 1)%LEN)];
	float val2 = hist[LEN + ((i2 + LEN - 1)%LEN)];
	float peak = i2 + 0.5f*(val1 - val2) / (2.0f*maxval2 - val1 - val2);
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+1], 0x7fffffff);
	if (idx<d_MaxNumPoints) {
	  d_Sift[idx].xpos = d_Sift[bx].xpos;
	  d_Sift[idx].ypos = d_Sift[bx].ypos;
	  d_Sift[idx].scale = sc;
	  d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	  d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	  d_Sift[idx].orientation = 360.0f*(peak<0.0f ? peak + LEN : peak)/LEN;
	  d_Sift[idx].subsampling = d_Sift[bx].subsampling;
	}
      }
    }
  }
#undef RAD
#undef WID
#undef LEN
} 

// With constant number of blocks
__global__ void ComputeOrientationsCONST(cudaTextureObject_t texObj, SiftPoint *d_Sift, int octave)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;
  
  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+0], d_MaxNumPoints);  
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
 
    float i2sigma2 = -1.0f/(2.0f*1.5f*1.5f*d_Sift[bx].scale*d_Sift[bx].scale);
    if (tx<11) 
      gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
    if (tx<64)
      hist[tx] = 0.0f;
    __syncthreads();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx/11;
    int xd = tx - yd*11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd<11) {
      float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf); 
      float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0); 
      int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
      if (bin>31)
	bin = 0;
      float grad = sqrtf(dx*dx + dy*dy);
      atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
    }
    __syncthreads();
    int x1m = (tx>=1 ? tx-1 : tx+31);
    int x1p = (tx<=30 ? tx+1 : tx-31);
    if (tx<32) {
      int x2m = (tx>=2 ? tx-2 : tx+30);
      int x2p = (tx<=29 ? tx+2 : tx-30);
      hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    __syncthreads();
    if (tx<32) {
      float v = hist[32+tx];
      hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
    }
    __syncthreads();
    if (tx==0) {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i=0;i<32;i++) {
	float v = hist[i];
	if (v>maxval1) {
	  maxval2 = maxval1;
	  maxval1 = v;
	  i2 = i1;
	  i1 = i;
	} else if (v>maxval2) {
	  maxval2 = v;
	  i2 = i;
	}
      }
      float val1 = hist[32+((i1+1)&31)];
      float val2 = hist[32+((i1+31)&31)];
      float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
      d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
      atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave+0]); 
      if (maxval2>0.8f*maxval1 && true) {
	float val1 = hist[32+((i2+1)&31)];
	float val2 = hist[32+((i2+31)&31)];
	float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+1], 0x7fffffff);
	if (idx<d_MaxNumPoints) {
	  d_Sift[idx].xpos = d_Sift[bx].xpos;
	  d_Sift[idx].ypos = d_Sift[bx].ypos;
	  d_Sift[idx].scale = d_Sift[bx].scale;
	  d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	  d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	  d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
	  d_Sift[idx].subsampling = d_Sift[bx].subsampling;
	}
      }
    }
    __syncthreads();
  }
} 

// With constant number of blocks
__global__ void OrientAndExtractCONST(cudaTextureObject_t texObj, SiftPoint *d_Sift, float subsampling, int octave)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  __shared__ unsigned int idx; //%%%%
  const int tx = threadIdx.x;
  
  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+0], d_MaxNumPoints);  
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
 
    float i2sigma2 = -1.0f/(4.5f*d_Sift[bx].scale*d_Sift[bx].scale);
    if (tx<11) 
      gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
    if (tx<64)
      hist[tx] = 0.0f;
    __syncthreads();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx/11;
    int xd = tx - yd*11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd<11) {
      float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf); 
      float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0); 
      int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
      if (bin>31)
	bin = 0;
      float grad = sqrtf(dx*dx + dy*dy);
      atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
    }
    __syncthreads();
    int x1m = (tx>=1 ? tx-1 : tx+31);
    int x1p = (tx<=30 ? tx+1 : tx-31);
    if (tx<32) {
      int x2m = (tx>=2 ? tx-2 : tx+30);
      int x2p = (tx<=29 ? tx+2 : tx-30);
      hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    __syncthreads();
    if (tx<32) {
      float v = hist[32+tx];
      hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
    }
    __syncthreads();
    if (tx==0) {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i=0;i<32;i++) {
	float v = hist[i];
	if (v>maxval1) {
	  maxval2 = maxval1;
	  maxval1 = v;
	  i2 = i1;
	  i1 = i;
	} else if (v>maxval2) {
	  maxval2 = v;
	  i2 = i;
	}
      }
      float val1 = hist[32+((i1+1)&31)];
      float val2 = hist[32+((i1+31)&31)];
      float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
      d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
      idx = 0xffffffff; //%%%%
      atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave+0]); 
      if (maxval2>0.8f*maxval1) {
	float val1 = hist[32+((i2+1)&31)];
	float val2 = hist[32+((i2+31)&31)];
	float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
	idx = atomicInc(&d_PointCounter[2*octave+1], 0x7fffffff); //%%%%
	if (idx<d_MaxNumPoints) {
	  d_Sift[idx].xpos = d_Sift[bx].xpos;
	  d_Sift[idx].ypos = d_Sift[bx].ypos;
	  d_Sift[idx].scale = d_Sift[bx].scale;
	  d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	  d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	  d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
	  d_Sift[idx].subsampling = d_Sift[bx].subsampling;
	}
      }
    }
    __syncthreads();
    ExtractSiftDescriptor(texObj, d_Sift, subsampling, octave, bx); //%%%%
    if (idx<d_MaxNumPoints) //%%%%
      ExtractSiftDescriptor(texObj, d_Sift, subsampling, octave, idx); //%%%%
  }
} 


///////////////////////////////////////////////////////////////////////////////
// Subtract two images (multi-scale version)
///////////////////////////////////////////////////////////////////////////////
  
__global__ void FindPointsMultiTest(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[3*MEMWID];

  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0) {
    atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]); 
    atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave-1]);
  }
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (tx==0 && ty==0)
    cnt = 0; 
  __syncthreads();

  int ypos = MINMAX_H*blockIdx.y + ty;
  if (ypos>=height)
    return;
  int block = blockIdx.x/NUM_SCALES; 
  int scale = blockIdx.x - NUM_SCALES*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  float maxv = fabs(d_Data0[ptr + ypos*pitch + 1*size]);
  maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));
    
  if (Shuffle(maxv, 0)>thresh) {
    int yptr1 = ptr + ypos*pitch;
    int yptr0 = ptr + max(0,ypos-1)*pitch;
    int yptr2 = ptr + min(height-1,ypos+1)*pitch;
    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    float d31 = d_Data0[yptr1 + 2*size];
    float d11 = d_Data0[yptr1];
    
    float d10 = d_Data0[yptr0];
    float d12 = d_Data0[yptr2];
    float ymin1 = fminf(fminf(d10, d11), d12);
    float ymax1 = fmaxf(fmaxf(d10, d11), d12);
    float d30 = d_Data0[yptr0 + 2*size];
    float d32 = d_Data0[yptr2 + 2*size]; 
    float ymin3 = fminf(fminf(d30, d31), d32);
    float ymax3 = fmaxf(fmaxf(d30, d31), d32);
    float ymin2 = fminf(fminf(ymin1, fminf(fminf(d20, d22), d21)), ymin3);
    float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d20, d22), d21)), ymax3);
    
    float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
    float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
    if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
      if (d21<-thresh) {
	float minv = fminf(fminf(nmin2, ymin1), ymin3);
	minv = fminf(fminf(minv, d20), d22);
	if (d21<minv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      } 
      if (d21>thresh) {
	float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
	maxv = fmaxf(fmaxf(maxv, d20), d22);
	if (d21>maxv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
    }
  }
  __syncthreads();
  if (ty==0 && tx<cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<edgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
      if (sc>=lowestScale) {
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
	d_Sift[idx].xpos = xpos + pdx;
	d_Sift[idx].ypos = ypos + pdy;
	d_Sift[idx].scale = sc;
	d_Sift[idx].sharpness = val + dval;
	d_Sift[idx].edgeness = edge;
	d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}

__global__ void FindPointsMultiNew(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ unsigned short points[2*MEMWID];
  
  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) {
    atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]);
    atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave-1]);
  }
  int tx = threadIdx.x;
  int block = blockIdx.x/NUM_SCALES; 
  int scale = blockIdx.x - NUM_SCALES*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
  float maxv = 0.0f;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float val = d_Data0[yptr1 + 1*size];
    maxv = fmaxf(maxv, fabs(val));
  }
  //if (tx==0) printf("XXX1\n");
  if (!__any_sync(0xffffffff, maxv>thresh))
    return;
  //if (tx==0) printf("XXX2\n");
  
  int ptbits = 0;
  for (int y=0;y<yloops;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float d11 = d_Data0[yptr1 + 1*size];
    if (__any_sync(0xffffffff, fabs(d11)>thresh)) {
    
      int yptr0 = ptr + max(0,ypos-1)*pitch;
      int yptr2 = ptr + min(height-1,ypos+1)*pitch;
      float d01 = d_Data0[yptr1];
      float d10 = d_Data0[yptr0 + 1*size];
      float d12 = d_Data0[yptr2 + 1*size];
      float d21 = d_Data0[yptr1 + 2*size];
      
      float d00 = d_Data0[yptr0];
      float d02 = d_Data0[yptr2];
      float ymin1 = fminf(fminf(d00, d01), d02);
      float ymax1 = fmaxf(fmaxf(d00, d01), d02);
      float d20 = d_Data0[yptr0 + 2*size];
      float d22 = d_Data0[yptr2 + 2*size]; 
      float ymin3 = fminf(fminf(d20, d21), d22);
      float ymax3 = fmaxf(fmaxf(d20, d21), d22);
      float ymin2 = fminf(fminf(ymin1, fminf(fminf(d10, d12), d11)), ymin3);
      float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d10, d12), d11)), ymax3);
      
      float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
      float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
      float minv = fminf(fminf(nmin2, ymin1), ymin3);
      minv = fminf(fminf(minv, d10), d12);
      float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
      maxv = fmaxf(fmaxf(maxv, d10), d12);
      
      if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) 
	ptbits |= ((d11 < fminf(-thresh, minv)) | (d11 > fmaxf(thresh, maxv))) << y;
    }
  }
  
  unsigned int totbits = __popc(ptbits);
  unsigned int numbits = totbits;
  for (int d=1;d<32;d<<=1) {
    unsigned int num = ShiftUp(totbits, d);
    if (tx >= d)
      totbits += num;
  }
  int pos = totbits - numbits;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    if (ptbits & (1 << y) && pos<MEMWID) {
      points[2*pos + 0] = xpos - 1;
      points[2*pos + 1] = ypos;
      pos ++;
    }
  } 

  totbits = Shuffle(totbits, 31);
  if (tx<totbits) {
    int xpos = points[2*tx + 0];
    int ypos = points[2*tx + 1];
    int ptr = xpos + (ypos + (scale + 1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<edgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
      if (sc>=lowestScale) {
	atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]); 
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
	d_Sift[idx].xpos = xpos + pdx;
	d_Sift[idx].ypos = ypos + pdy;
	d_Sift[idx].scale = sc;
	d_Sift[idx].sharpness = val + dval;
	d_Sift[idx].edgeness = edge;
	d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}

__global__ void FindPointsMulti(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[3*MEMWID];

  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) {
    atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]);
    atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave-1]);
  }
  int tx = threadIdx.x;
  int block = blockIdx.x/NUM_SCALES; 
  int scale = blockIdx.x - NUM_SCALES*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
  float maxv = 0.0f;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float val = d_Data0[yptr1 + 1*size];
    maxv = fmaxf(maxv, fabs(val));
  }
  maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));
  if (Shuffle(maxv, 0)<=thresh)
    return;
  
  if (tx==0)
    cnt = 0; 
  __syncthreads();

  for (int y=0;y<yloops;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    int yptr0 = ptr + max(0,ypos-1)*pitch;
    int yptr2 = ptr + min(height-1,ypos+1)*pitch;
    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    float d31 = d_Data0[yptr1 + 2*size];
    float d11 = d_Data0[yptr1];
    
    float d10 = d_Data0[yptr0];
    float d12 = d_Data0[yptr2];
    float ymin1 = fminf(fminf(d10, d11), d12);
    float ymax1 = fmaxf(fmaxf(d10, d11), d12);
    float d30 = d_Data0[yptr0 + 2*size];
    float d32 = d_Data0[yptr2 + 2*size]; 
    float ymin3 = fminf(fminf(d30, d31), d32);
    float ymax3 = fmaxf(fmaxf(d30, d31), d32);
    float ymin2 = fminf(fminf(ymin1, fminf(fminf(d20, d22), d21)), ymin3);
    float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d20, d22), d21)), ymax3);
    
    float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
    float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
    if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
      if (d21<-thresh) {
	float minv = fminf(fminf(nmin2, ymin1), ymin3);
	minv = fminf(fminf(minv, d20), d22);
	if (d21<minv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      } 
      if (d21>thresh) {
	float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
	maxv = fmaxf(fmaxf(maxv, d20), d22);
	if (d21>maxv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
    }
  }
  if (tx<cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<edgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
      if (sc>=lowestScale) {
	atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]); 
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
	d_Sift[idx].xpos = xpos + pdx;
	d_Sift[idx].ypos = ypos + pdy;
	d_Sift[idx].scale = sc;
	d_Sift[idx].sharpness = val + dval;
	d_Sift[idx].edgeness = edge;
	d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}


__global__ void FindPointsMultiOld(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ float ymin1[MEMWID], ymin2[MEMWID], ymin3[MEMWID];
  __shared__ float ymax1[MEMWID], ymax2[MEMWID], ymax3[MEMWID];
  __shared__ unsigned int cnt;
  __shared__ unsigned short points[3*MEMWID];

  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) {
    atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]); 
    atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave-1]);
  }
  int tx = threadIdx.x;
  int block = blockIdx.x/NUM_SCALES; 
  int scale = blockIdx.x - NUM_SCALES*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
  float maxv = 0.0f;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float val = d_Data0[yptr1 + 1*size];
    maxv = fmaxf(maxv, fabs(val));
  }
  maxv = fmaxf(maxv, ShiftDown(maxv, 16, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 8, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 4, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 2, MINMAX_W));
  maxv = fmaxf(maxv, ShiftDown(maxv, 1, MINMAX_W));
  if (Shuffle(maxv, 0)<=thresh)
    return;
  
  if (tx==0)
    cnt = 0; 
  __syncthreads();

  for (int y=0;y<yloops;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    int yptr0 = ptr + max(0,ypos-1)*pitch;
    int yptr2 = ptr + min(height-1,ypos+1)*pitch;
    float d20 = d_Data0[yptr0 + 1*size];
    float d21 = d_Data0[yptr1 + 1*size];
    float d22 = d_Data0[yptr2 + 1*size];
    float d31 = d_Data0[yptr1 + 2*size];
    float d11 = d_Data0[yptr1];

    float d10 = d_Data0[yptr0];
    float d12 = d_Data0[yptr2];
    ymin1[tx] = fminf(fminf(d10, d11), d12);
    ymax1[tx] = fmaxf(fmaxf(d10, d11), d12);
    float d30 = d_Data0[yptr0 + 2*size];
    float d32 = d_Data0[yptr2 + 2*size]; 
    ymin3[tx] = fminf(fminf(d30, d31), d32);
    ymax3[tx] = fmaxf(fmaxf(d30, d31), d32);
    ymin2[tx] = fminf(fminf(ymin1[tx], fminf(fminf(d20, d22), d21)), ymin3[tx]);
    ymax2[tx] = fmaxf(fmaxf(ymax1[tx], fmaxf(fmaxf(d20, d22), d21)), ymax3[tx]);
    
    __syncthreads(); 

    if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) {
      if (d21<-thresh) {
	float minv = fminf(fminf(fminf(ymin2[tx-1], ymin2[tx+1]), ymin1[tx]), ymin3[tx]);
	minv = fminf(fminf(minv, d20), d22);
	if (d21<minv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      } 
      if (d21>thresh) {
	float maxv = fmaxf(fmaxf(fmaxf(ymax2[tx-1], ymax2[tx+1]), ymax1[tx]), ymax3[tx]);
	maxv = fmaxf(fmaxf(maxv, d20), d22);
	if (d21>maxv) { 
	  int pos = atomicInc(&cnt, MEMWID-1);
	  points[3*pos+0] = xpos - 1;
	  points[3*pos+1] = ypos;
	  points[3*pos+2] = scale;
	}
      }
    }
    __syncthreads();
  }
  if (tx<cnt) {
    int xpos = points[3*tx+0];
    int ypos = points[3*tx+1];
    int scale = points[3*tx+2];
    int ptr = xpos + (ypos + (scale+1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<edgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
      if (sc>=lowestScale) {
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
	d_Sift[idx].xpos = xpos + pdx;
	d_Sift[idx].ypos = ypos + pdy;
	d_Sift[idx].scale = sc;
	d_Sift[idx].sharpness = val + dval;
	d_Sift[idx].edgeness = edge;
	d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}


__global__ void LaplaceMultiTex(cudaTextureObject_t texObj, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_LaplaceKernel + octave*12*16 + scale*16;
  float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale; 
  float x = xp-3.5;
  float y = yp+0.5;
  sdata1[tx] = kernel[0]*tex2D<float>(texObj, x, y) + 
    kernel[1]*(tex2D<float>(texObj, x, y-1.0) + tex2D<float>(texObj, x, y+1.0)) + 
    kernel[2]*(tex2D<float>(texObj, x, y-2.0) + tex2D<float>(texObj, x, y+2.0)) + 
    kernel[3]*(tex2D<float>(texObj, x, y-3.0) + tex2D<float>(texObj, x, y+3.0)) + 
    kernel[4]*(tex2D<float>(texObj, x, y-4.0) + tex2D<float>(texObj, x, y+4.0));
  __syncthreads();
  float *sdata2 = data2 + LAPLACE_W*scale; 
  if (tx<LAPLACE_W) {
    sdata2[tx] = kernel[0]*sdata1[tx+4] + 
      kernel[1]*(sdata1[tx+3] + sdata1[tx+5]) + 
      kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) + 
      kernel[3]*(sdata1[tx+1] + sdata1[tx+7]) + 
      kernel[4]*(sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads(); 
  if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width) 
    d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}


__global__ void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float buff[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  float *data = d_Image + max(min(xp - LAPLACE_R, width-1), 0);
  float temp[2*LAPLACE_R + 1], kern[LAPLACE_S][LAPLACE_R + 1];
  if (xp<(width + 2*LAPLACE_R)) {
    for (int i=0;i<=2*LAPLACE_R;i++)
      temp[i] = data[max(0, min(yp + i - LAPLACE_R, height - 1))*pitch];
    for (int scale=0;scale<LAPLACE_S;scale++) {
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      float *kernel = d_LaplaceKernel + octave*12*16 + scale*16; 
      for (int i=0;i<=LAPLACE_R;i++)
	kern[scale][i] = kernel[i];
      float sum = kern[scale][0]*temp[LAPLACE_R];
#pragma unroll      
      for (int j=1;j<=LAPLACE_R;j++)
	sum += kern[scale][j]*(temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
      buf[tx] = sum;
    }
  }
  __syncthreads();
  if (tx<LAPLACE_W && xp<width) {
    int scale = 0;
    float oldRes = kern[scale][0]*buff[tx + LAPLACE_R];
#pragma unroll
    for (int j=1;j<=LAPLACE_R;j++)
      oldRes += kern[scale][j]*(buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]); 
    for (int scale=1;scale<LAPLACE_S;scale++) {
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      float res = kern[scale][0]*buf[tx + LAPLACE_R];
#pragma unroll
      for (int j=1;j<=LAPLACE_R;j++)
	res += kern[scale][j]*(buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]); 
      d_Result[(scale-1)*height*pitch + yp*pitch + xp] = res - oldRes;
      oldRes = res;
    }
  }
}

__global__ void LaplaceMultiMemWide(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float buff[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int xp4 = blockIdx.x*LAPLACE_W + 4*tx;
  const int yp = blockIdx.y;
  float kern[LAPLACE_S][LAPLACE_R+1];
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  float temp[9]; 
  if (xp<(width + 2*LAPLACE_R)) {
    for (int i=0;i<4;i++)
      temp[i] = data[max(0, min(yp+i-4, height-1))*pitch];
    for (int i=4;i<8+1;i++)
      temp[i] = data[min(yp+i-4, height-1)*pitch];
    for (int scale=0;scale<LAPLACE_S;scale++) {
      float *kernel = d_LaplaceKernel + octave*12*16 + scale*16; 
      for (int i=0;i<=LAPLACE_R;i++)
	kern[scale][i] = kernel[LAPLACE_R - i];
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      buf[tx] = kern[scale][4]*temp[4] +
	kern[scale][3]*(temp[3] + temp[5]) + kern[scale][2]*(temp[2] + temp[6]) + 
	kern[scale][1]*(temp[1] + temp[7]) + kern[scale][0]*(temp[0] + temp[8]);
    }
  }
  __syncthreads();
  if (tx<LAPLACE_W/4 && xp4<width) {
    float4 b0 = reinterpret_cast<float4*>(buff)[tx+0];
    float4 b1 = reinterpret_cast<float4*>(buff)[tx+1];
    float4 b2 = reinterpret_cast<float4*>(buff)[tx+2];
    float4 old4, new4, dif4;
    old4.x = kern[0][4]*b1.x + kern[0][3]*(b0.w + b1.y) + kern[0][2]*(b0.z + b1.z) +
      kern[0][1]*(b0.y + b1.w) + kern[0][0]*(b0.x + b2.x);
    old4.y = kern[0][4]*b1.y + kern[0][3]*(b1.x + b1.z) + kern[0][2]*(b0.w + b1.w) +
      kern[0][1]*(b0.z + b2.x) + kern[0][0]*(b0.y + b2.y);
    old4.z = kern[0][4]*b1.z + kern[0][3]*(b1.y + b1.w) + kern[0][2]*(b1.x + b2.x) +
      kern[0][1]*(b0.w + b2.y) + kern[0][0]*(b0.z + b2.z);
    old4.w = kern[0][4]*b1.w + kern[0][3]*(b1.z + b2.x) + kern[0][2]*(b1.y + b2.y) +
      kern[0][1]*(b1.x + b2.z) + kern[0][0]*(b0.w + b2.w);
    for (int scale=1;scale<LAPLACE_S;scale++) {
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      float4 b0 = reinterpret_cast<float4*>(buf)[tx+0];
      float4 b1 = reinterpret_cast<float4*>(buf)[tx+1];
      float4 b2 = reinterpret_cast<float4*>(buf)[tx+2];
      new4.x = kern[scale][4]*b1.x + kern[scale][3]*(b0.w + b1.y) +
	kern[scale][2]*(b0.z + b1.z) + kern[scale][1]*(b0.y + b1.w) +
	kern[scale][0]*(b0.x + b2.x);
      new4.y = kern[scale][4]*b1.y + kern[scale][3]*(b1.x + b1.z) +
	kern[scale][2]*(b0.w + b1.w) + kern[scale][1]*(b0.z + b2.x) +
	kern[scale][0]*(b0.y + b2.y);
      new4.z = kern[scale][4]*b1.z + kern[scale][3]*(b1.y + b1.w) +
	kern[scale][2]*(b1.x + b2.x) + kern[scale][1]*(b0.w + b2.y) +
	kern[scale][0]*(b0.z + b2.z);
      new4.w = kern[scale][4]*b1.w + kern[scale][3]*(b1.z + b2.x) +
	kern[scale][2]*(b1.y + b2.y) + kern[scale][1]*(b1.x + b2.z) +
	kern[scale][0]*(b0.w + b2.w);
      dif4.x = new4.x - old4.x;
      dif4.y = new4.y - old4.y;
      dif4.z = new4.z - old4.z;
      dif4.w = new4.w - old4.w;
      reinterpret_cast<float4*>(&d_Result[(scale-1)*height*pitch + yp*pitch + xp4])[0] = dif4;
      old4 = new4;
    }
  }
}

__global__ void LaplaceMultiMemTest(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = LAPLACE_H*blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_LaplaceKernel + octave*12*16 + scale*16; 
  float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale; 
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  int h = height-1;
  float temp[8+LAPLACE_H], kern[LAPLACE_R+1];
  for (int i=0;i<4;i++)
    temp[i] = data[max(0, min(yp+i-4, h))*pitch];
  for (int i=4;i<8+LAPLACE_H;i++)
    temp[i] = data[min(yp+i-4, h)*pitch];
  for (int i=0;i<=LAPLACE_R;i++)
    kern[i] = kernel[LAPLACE_R - i];
  for (int j=0;j<LAPLACE_H;j++) {
    sdata1[tx] = kern[4]*temp[4+j] +
      kern[3]*(temp[3+j] + temp[5+j]) + kern[2]*(temp[2+j] + temp[6+j]) + 
      kern[1]*(temp[1+j] + temp[7+j]) + kern[0]*(temp[0+j] + temp[8+j]);
    __syncthreads();
    float *sdata2 = data2 + LAPLACE_W*scale; 
    if (tx<LAPLACE_W) {
      sdata2[tx] = kern[4]*sdata1[tx+4] + 
	kern[3]*(sdata1[tx+3] + sdata1[tx+5]) + kern[2]*(sdata1[tx+2] + sdata1[tx+6]) + 
	kern[1]*(sdata1[tx+1] + sdata1[tx+7]) + kern[0]*(sdata1[tx+0] + sdata1[tx+8]);
    }
    __syncthreads(); 
    if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width && (yp+j)<height) 
      d_Result[scale*height*pitch + (yp+j)*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
  }
}

__global__ void LaplaceMultiMemOld(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float data1[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  __shared__ float data2[LAPLACE_W*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  const int scale = threadIdx.y;
  float *kernel = d_LaplaceKernel + octave*12*16 + scale*16; 
  float *sdata1 = data1 + (LAPLACE_W + 2*LAPLACE_R)*scale; 
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  int h = height-1;
  sdata1[tx] = kernel[0]*data[min(yp, h)*pitch] +
    kernel[1]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) + 
    kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) + 
    kernel[3]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) + 
    kernel[4]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
  __syncthreads();
  float *sdata2 = data2 + LAPLACE_W*scale; 
  if (tx<LAPLACE_W) {
    sdata2[tx] = kernel[0]*sdata1[tx+4] + 
      kernel[1]*(sdata1[tx+3] + sdata1[tx+5]) +
      kernel[2]*(sdata1[tx+2] + sdata1[tx+6]) + 
      kernel[3]*(sdata1[tx+1] + sdata1[tx+7]) +
      kernel[4]*(sdata1[tx+0] + sdata1[tx+8]);
  }
  __syncthreads(); 
  if (tx<LAPLACE_W && scale<LAPLACE_S-1 && xp<width) 
    d_Result[scale*height*pitch + yp*pitch + xp] = sdata2[tx] - sdata2[tx+LAPLACE_W];
}

__global__ void LowPass(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float buffer[(LOWPASS_W + 2*LOWPASS_R)*LOWPASS_H];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  float *kernel = d_LowPassKernel;
  float *data = d_Image + max(min(xp - 4, width-1), 0);
  float *buff = buffer + ty*(LOWPASS_W + 2*LOWPASS_R);
  int h = height-1;
  if (yp<height) 
    buff[tx] = kernel[4]*data[min(yp, h)*pitch] +
      kernel[3]*(data[max(0, min(yp-1, h))*pitch] + data[min(yp+1, h)*pitch]) + 
      kernel[2]*(data[max(0, min(yp-2, h))*pitch] + data[min(yp+2, h)*pitch]) + 
      kernel[1]*(data[max(0, min(yp-3, h))*pitch] + data[min(yp+3, h)*pitch]) + 
      kernel[0]*(data[max(0, min(yp-4, h))*pitch] + data[min(yp+4, h)*pitch]);
  __syncthreads();
  if (tx<LOWPASS_W && xp<width && yp<height)
    d_Result[yp*pitch + xp] = kernel[4]*buff[tx+4] + 
      kernel[3]*(buff[tx+3] + buff[tx+5]) + kernel[2]*(buff[tx+2] + buff[tx+6]) + 
      kernel[1]*(buff[tx+1] + buff[tx+7]) + kernel[0]*(buff[tx+0] + buff[tx+8]);
}

__global__ void LowPassBlockOld(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];          
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width-1), 0);
  for (int l=-8;l<=LOWPASS_H;l+=4) {
    if (l<LOWPASS_H) {
      int yl = max(min(yp + l + 4, height-1), 0);
      float val = d_Image[yl*pitch + xl];
      xrows[(l + 8 + ty)%N][tx] = k[4]*ShiftDown(val, 4) +
	k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
	k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
	k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
	k[0]*(ShiftDown(val, 8) + val);
    }
    if (l>=4) {
      int ys = yp + l - 4;
      if (xp<width && ys<height && tx<LOWPASS_W)
	d_Result[ys*pitch + xp] = k[4]*xrows[(l + 0 + ty)%N][tx] +
	     k[3]*(xrows[(l - 1 + ty)%N][tx] + xrows[(l + 1 + ty)%N][tx]) +
	     k[2]*(xrows[(l - 2 + ty)%N][tx] + xrows[(l + 2 + ty)%N][tx]) +
	     k[1]*(xrows[(l - 3 + ty)%N][tx] + xrows[(l + 3 + ty)%N][tx]) +
	     k[0]*(xrows[(l - 4 + ty)%N][tx] + xrows[(l + 4 + ty)%N][tx]);
    }
    if (l>=0)
      __syncthreads();
  }
}

__global__ void LowPassBlock(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];          
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width-1), 0);
#pragma unroll
  for (int l=-8;l<4;l+=4) {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height-1), 0);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  __syncthreads();
#pragma unroll
  for (int l=4;l<LOWPASS_H;l+=4) {
    int ly = l + ty;
    int yl = min(yp + l + 4, height-1);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[(ly + 8)%N][tx] = val;
    int ys = yp + l - 4;
    if (xp<width && ys<height && tx<LOWPASS_W)
      d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		       k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		       k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		       k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		       k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
    __syncthreads();
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp<width && ys<height && tx<LOWPASS_W)
    d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		     k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		     k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		     k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		     k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
}

//======= Part 2b': VLFeat-schedule pyramid kernels (clean-room from =======//
//======= colmap/src/thirdparty/VLFeat/sift.c, BSD; not SiftGPU)     =======//

// Pure 2x decimation (no anti-alias blur) -- matches VLFeat's
// copy_and_downsample exactly (sift.c:836-851): dst(y,x) = src(2y,2x).
// NOT the same as cusift's own ScaleDown, which bakes in a 5-tap Gaussian
// blur; VLFeat's octave transition assumes a blur-free decimation because
// the correct blur is already accounted for by the per-level sigma schedule.
__global__ void PureDecimate2xKernel(float *d_Result, float *d_Data,
                                     int dstWidth, int dstPitch, int dstHeight,
                                     int srcPitch)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=dstWidth || y>=dstHeight) return;
  d_Result[y*dstPitch + x] = d_Data[(2*y)*srcPitch + 2*x];
}

// Exact replica of vl_sift_detect's refinement step (sift.c:1267-1427):
// up to 5 Newton iterations with pivoted 3x3 Gauss elimination, relocation
// when |b[0]| or |b[1]| > 0.6, final threshold+edge test on the refined
// position.
//
// docs/PERFORMANCE.md Tier 2.1-B: ONE implementation, compiled for both host and
// device. The device instantiation (RefineKeypointsKernel below) is the
// production path -- it removes the ~221 MB/image DoG download that existed
// only to feed this function on the CPU, plus the serial CPU pass itself
// (together ~40% of extraction time). The host instantiation is still used by
// the --debug-dump-prefix path, which needs per-candidate accept/reject
// logging. Keeping one body is deliberate: two copies of this math would drift.
//
// `yo`/`so` are explicit row/plane strides so the same code can index either a
// tightly packed host cube (yo=width, so=width*height) or the pitched device
// DoG buffer (yo=pitch, so=height*pitch). Element values are identical either
// way -- only addressing differs.
//
// nv_fmad off is DEFENSIVE, not load-bearing today (measured): nvcc may contract
// a*b+c into a fused FMA in device code, which rounds differently from the host's
// separate multiply and add, and every accept/reject decision below is a
// threshold comparison on the result of this arithmetic. With the pragma, this
// function's device instantiation matches its host instantiation to all 17
// significant digits on every one of 4605 keypoints; removing it was also
// bit-identical on that image, so the pragma is insurance against a future
// compiler or refactor making the contraction visible, not the reason the two
// agree. Scoped to this function -- the fp32 kernels' fast-math/FMA behavior is
// untouched (and --use_fast_math does not affect fp64 anyway).
#pragma nv_fmad off
__host__ __device__ static bool RefineOneKeypoint(const float *dog,
                              int width, int height, int yo, int so,
                              int nDogLevels, int s_min, int s_max,
                              int ix0, int iy0, int is0,
                              double peakThresh, double edgeThresh,
                              DetectedKeypoint &out)
{
  const int xo = 1;
  int x = ix0, y = iy0;
  const int s = is0;  // level never changes during refinement (verified)
  double Dx=0, Dy=0, Ds=0, Dxx=0, Dyy=0, Dxy=0, Dxs=0, Dys=0;
  double A[9], b[3] = {0,0,0};
  int dx = 0, dy = 0;

  auto at = [&](int ddx, int ddy, int dds) -> double {
    return dog[(size_t)(s - s_min + dds)*so + (size_t)(y+ddy)*yo + (x+ddx)*xo];
  };

  for (int iter = 0; iter < 5; iter++) {
    x += dx; y += dy;
    if (x < 1 || x > width-2 || y < 1 || y > height-2) return false;

    Dx = 0.5*(at(1,0,0) - at(-1,0,0));
    Dy = 0.5*(at(0,1,0) - at(0,-1,0));
    Ds = 0.5*(at(0,0,1) - at(0,0,-1));
    double Dss_ = at(0,0,1) + at(0,0,-1) - 2.0*at(0,0,0);
    Dxx = at(1,0,0) + at(-1,0,0) - 2.0*at(0,0,0);
    Dyy = at(0,1,0) + at(0,-1,0) - 2.0*at(0,0,0);
    Dxy = 0.25*(at(1,1,0) + at(-1,-1,0) - at(-1,1,0) - at(1,-1,0));
    Dxs = 0.25*(at(1,0,1) + at(-1,0,-1) - at(-1,0,1) - at(1,0,-1));
    Dys = 0.25*(at(0,1,1) + at(0,-1,-1) - at(0,-1,1) - at(0,1,-1));

    A[0]=Dxx; A[4]=Dyy; A[8]=Dss_;
    A[1]=A[3]=Dxy; A[2]=A[6]=Dxs; A[5]=A[7]=Dys;
    b[0]=-Dx; b[1]=-Dy; b[2]=-Ds;

    for (int j = 0; j < 3; j++) {
      double maxabsa = 0, maxa = 0; int maxi = -1;
      for (int i = j; i < 3; i++) {
        double a = A[i+j*3], absa = fabs(a);
        if (absa > maxabsa) { maxa = a; maxabsa = absa; maxi = i; }
      }
      if (maxabsa < 1e-10) { b[0]=b[1]=b[2]=0; break; }
      int i = maxi;
      for (int jj = j; jj < 3; jj++) {
        double tmp = A[i+jj*3]; A[i+jj*3] = A[j+jj*3]; A[j+jj*3] = tmp;
        A[j+jj*3] /= maxa;
      }
      double tmp = b[j]; b[j] = b[i]; b[i] = tmp;
      b[j] /= maxa;
      for (int ii = j+1; ii < 3; ii++) {
        double xf = A[ii+j*3];
        for (int jj = j; jj < 3; jj++) A[ii+jj*3] -= xf*A[j+jj*3];
        b[ii] -= xf*b[j];
      }
    }
    for (int i = 2; i > 0; i--) {
      double xf = b[i];
      for (int ii = i-1; ii >= 0; ii--) b[ii] -= xf*A[ii+i*3];
    }

    dx = (b[0] >  0.6 && x < width-2)  ?  1 : 0;
    dx += (b[0] < -0.6 && x > 1)       ? -1 : 0;
    dy = (b[1] >  0.6 && y < height-2) ?  1 : 0;
    dy += (b[1] < -0.6 && y > 1)       ? -1 : 0;
    if (dx == 0 && dy == 0) break;
  }

  double val = at(0,0,0) + 0.5*(Dx*b[0] + Dy*b[1] + Ds*b[2]);
  double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy);
  double xn = x + b[0], yn = y + b[1], sn = s + b[2];

  bool good = fabs(val) > peakThresh
      && score < (edgeThresh+1)*(edgeThresh+1)/edgeThresh && score >= 0
      && fabs(b[0]) < 1.5 && fabs(b[1]) < 1.5 && fabs(b[2]) < 1.5
      && xn >= 0 && xn <= width-1 && yn >= 0 && yn <= height-1
      && sn >= s_min && sn <= s_max;
  if (!good) return false;

  out.ix = x; out.iy = y; out.is = s; out.s = (float)sn;
  out.x = (float)xn; out.y = (float)yn;
  return true;
}

#pragma nv_fmad on

// Tier 2.1-B: one thread per coarse candidate. Accepted keypoints are appended
// with an atomic counter, so only the survivors (~10 k x 32 B) cross PCIe.
// Output ORDER differs from the CPU path's candidate order; the extractor's row
// order was already nondeterministic (FindDogExtremaKernel appends the same
// way), and nothing downstream depends on it.
__global__ void RefineKeypointsKernel(const float *dog, int width, int height,
                                      int yo, int so, int nDogLevels,
                                      int s_min, int s_max,
                                      const int *candX, const int *candY, const int *candS,
                                      unsigned int candCount,
                                      double peakThresh, double edgeThresh,
                                      DetectedKeypoint *out, unsigned int *outCount)
{
  unsigned int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= candCount) return;
  DetectedKeypoint kp;
  if (!RefineOneKeypoint(dog, width, height, yo, so, nDogLevels, s_min, s_max,
                         candX[c], candY[c], candS[c], peakThresh, edgeThresh, kp))
    return;
  out[atomicAdd(outCount, 1u)] = kp;
}

// Adaptive-radius separable Gaussian blur -- VLFeat truncates its smoothing
// kernels at radius=ceil(4*sigma) (not cusift's own LowPass, which hardcodes
// LOWPASS_R=4 regardless of sigma -- confirmed by direct inspection to
// under-blur essentially every level the VLFeat schedule needs: e.g. the
// octave-0 pre-blur alone needs sigma~1.5, i.e. radius~6-7, and the largest
// per-level increment needs sigma~3+, i.e. radius~13+, both far beyond the
// fixed radius-4 support LowPass provides). Both passes take EXPLICIT
// separate src/dst pitches -- no same-pitch assumption between independently
// allocated buffers, learned the hard way earlier this session.
// docs/PERFORMANCE.md Tier 1.3: the sigma schedule is image-independent and only
// ~8 distinct sigmas ever occur (two base blurs + the per-level increments,
// which repeat identically in every octave), so weights live in a BANK of
// kernels uploaded once per process instead of one synchronous
// cudaMemcpyToSymbol per blur (24 per image). Each blur is handed the base
// offset of its own kernel; weights are computed by the same host code as
// before, so the values are bit-identical.
#define MAX_GAUSS_RADIUS 64
#define GAUSS_BANK_FLOATS 4096
__constant__ float d_AdaptiveGaussKernel[GAUSS_BANK_FLOATS];

__global__ void GaussianBlurHorizontalKernel(float * __restrict__ d_Result,
                                             const float * __restrict__ d_Data,
                                             int width, int dstPitch, int srcPitch,
                                             int height, int radius, int kOffset)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  float sum = 0.0f;
  for (int j=-radius;j<=radius;j++) {
    int xx = min(max(x+j,0), width-1);
    sum += d_AdaptiveGaussKernel[kOffset+j+radius] * d_Data[y*srcPitch+xx];
  }
  d_Result[y*dstPitch+x] = sum;
}

__global__ void GaussianBlurVerticalKernel(float * __restrict__ d_Result,
                                           const float * __restrict__ d_Data,
                                           int width, int dstPitch, int srcPitch,
                                           int height, int radius, int kOffset)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  float sum = 0.0f;
  for (int j=-radius;j<=radius;j++) {
    int yy = min(max(y+j,0), height-1);
    sum += d_AdaptiveGaussKernel[kOffset+j+radius] * d_Data[yy*srcPitch+x];
  }
  d_Result[y*dstPitch+x] = sum;
}

// 2x linear upsample, one dimension at a time -- matches VLFeat's
// copy_and_upsample_rows exactly (sift.c:738-758): even output samples copy
// the input directly, odd samples are the average of the two neighboring
// input samples, with the last input sample repeated at the far edge (no
// extrapolation). Applying this once horizontally then once vertically
// reproduces VLFeat's actual o_min<0 upsampling (which does it via two
// transposed passes -- mathematically identical result, done here without
// the transpose since CUDA addressing doesn't need it).
__global__ void Upsample2xHorizontalKernel(float *d_Result, const float *d_Data,
                                           int srcWidth, int dstPitch, int srcPitch, int height)
{
  int xh = blockIdx.x*blockDim.x + threadIdx.x;  // output column, 0..2*srcWidth-1
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int dstWidth = 2*srcWidth;
  if (xh>=dstWidth || y>=height) return;
  int x0 = xh/2;
  if ((xh & 1) == 0) {
    d_Result[y*dstPitch+xh] = d_Data[y*srcPitch+x0];
  } else {
    int x1 = min(x0+1, srcWidth-1);
    d_Result[y*dstPitch+xh] = 0.5f*(d_Data[y*srcPitch+x0] + d_Data[y*srcPitch+x1]);
  }
}

__global__ void Upsample2xVerticalKernel(float *d_Result, const float *d_Data,
                                         int width, int dstPitch, int srcPitch, int srcHeight)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int yh = blockIdx.y*blockDim.y + threadIdx.y;  // output row, 0..2*srcHeight-1
  int dstHeight = 2*srcHeight;
  if (x>=width || yh>=dstHeight) return;
  int y0 = yh/2;
  if ((yh & 1) == 0) {
    d_Result[yh*dstPitch+x] = d_Data[y0*srcPitch+x];
  } else {
    int y1 = min(y0+1, srcHeight-1);
    d_Result[yh*dstPitch+x] = 0.5f*(d_Data[y0*srcPitch+x] + d_Data[y1*srcPitch+x]);
  }
}

// Elementwise scale (used to normalize [0,255] pixel input to [0,1] for the
// VLFeat-schedule pyramid -- COLMAP does the same before calling VLFeat,
// colmap/src/colmap/feature/sift.cc:186-188).
__global__ void ScaleImageKernel(float *d_Result, float *d_Data,
                                 int width, int dstPitch, int srcPitch, int height, float factor)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  d_Result[y*dstPitch+x] = d_Data[y*srcPitch+x] * factor;
}

// Gradient magnitude+angle plane for one Gaussian level. Central differences,
// one-sided at borders -- matches VLFeat's update_gradient exactly
// (sift.c:1447-1484). Angle wrapped to [0, 2*pi).
// VLFeat's update_gradient does NOT use exact sqrt/atan2 -- it uses
// vl_fast_sqrt_f (mathop.h:479-497, the classic Carmack/Tarolli fast inverse
// sqrt: magic constant 0x5f3759df + TWO Newton iterations) and
// vl_fast_atan2_f (mathop.h:406-423, a polynomial approximation, max error
// ~0.0061 rad). Replicated verbatim from VLFeat, which is BSD 2-clause --
// see LICENSE.VLFeat, whose notice must ship with any redistribution of
// this file -- using exact sqrtf/atan2f here would silently diverge
// from VLFeat's actual gradient field by up to ~0.3 deg per sample, exactly
// the kind of systematic, position-dependent bias that flips borderline
// 0.8*max orientation peaks.
__device__ float vl_fast_resqrt_f_cuda(float x)
{
  union { float f; int i; } u;
  float xhalf = 0.5f * x;
  u.f = x;
  u.i = 0x5f3759df - (u.i >> 1);
  u.f = u.f * (1.5f - xhalf*u.f*u.f);
  u.f = u.f * (1.5f - xhalf*u.f*u.f);
  return u.f;
}

__device__ float vl_fast_sqrt_f_cuda(float x)
{
  return (x < 1e-8f) ? 0.0f : x * vl_fast_resqrt_f_cuda(x);
}

__device__ float vl_fast_atan2_f_cuda(float y, float x)
{
  const float c3 = 0.1821f;
  const float c1 = 0.9675f;
  float abs_y = fabsf(y) + 1.19209290e-07f;  // VL_EPSILON_F
  float angle, r;
  if (x >= 0.0f) {
    r = (x - abs_y) / (x + abs_y);
    angle = (float)(M_PI / 4.0);
  } else {
    r = (x + abs_y) / (abs_y - x);
    angle = (float)(3.0 * M_PI / 4.0);
  }
  angle += (c3*r*r - c1) * r;
  return (y < 0.0f) ? -angle : angle;
}

// Debug-build guard for Tier 2.3: the gradient planes that are deliberately not
// computed (level indices outside 1..numScales) are filled with SIGNALING NaN
// instead of being left as whatever the allocator handed back. Garbage fails
// quietly -- if a future consumer breaks the `is - s_min` invariant (someone
// widens the level range, or adds a path that does not go through
// AssignOctaveLevel) it would read plausible-looking floats and produce subtly
// wrong descriptors. NaN propagates through the orientation histogram and the
// descriptor normalization, so the accuracy gate fails immediately and visibly
// instead. Note this is a gate-level trap, not a hardware one: the GPU does not
// raise FP exceptions on signaling NaN.
//
// Debug builds only (guarded by NDEBUG at the call site), so production keeps the
// bandwidth that skipping those planes saves -- which is the whole point of 2.3.
__global__ void FillPlaneKernel(float *d_Data, int width, int pitch, int height, float value)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  d_Data[y*pitch+x] = value;
}

// Tier 3.1 (interleaved float2 (mag, ang) plane) was implemented and MEASURED
// on the 4060: 4% SLOWER than these two separate planes (31.6 vs 30.3 ms/image),
// so it was reverted. The plan's premise -- two transactions per sample instead
// of one -- is true but not what limits these kernels: both layouts fetch the
// same bytes per useful sample, and the interleaved rows are twice as long,
// which costs 2D-window locality in the orientation/descriptor access pattern.
// Do not re-try it without a profile showing these kernels are
// transaction-limited. See docs/perf_tier1_rtx4060.md.
__global__ void ComputeGradientPlaneKernel(const float *d_Level, int width, int pitch, int height,
                                           float *d_GradMag, float *d_GradAng)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  int xm = max(x-1,0), xp = min(x+1,width-1);
  int ym = max(y-1,0), yp = min(y+1,height-1);
  float l = d_Level[y*pitch+xm], r = d_Level[y*pitch+xp];
  float u = d_Level[ym*pitch+x], d = d_Level[yp*pitch+x];
  float gx = (x==0 || x==width-1) ? (r-l) : 0.5f*(r-l);
  float gy = (y==0 || y==height-1) ? (d-u) : 0.5f*(d-u);
  float mag = vl_fast_sqrt_f_cuda(gx*gx+gy*gy);
  float ang = vl_fast_atan2_f_cuda(gy,gx);
  const float twoPi = 6.283185307179586f;
  ang = ang + twoPi;
  ang = ang - twoPi*floorf(ang/twoPi);  // vl_mod_2pi_f(ang + 2*pi) equivalent
  d_GradMag[y*pitch+x] = mag;
  d_GradAng[y*pitch+x] = ang;
}

// VLFeat-parity orientation kernel (Part 2c). One block per candidate
// keypoint (x,y,scale only -- orientation not yet known). Mirrors
// vl_sift_calc_keypoint_orientations exactly (sift.c:1559-1699): 36-bin
// histogram, Gaussian window sigmaw=1.5*scale, radius W=max(floor(3*sigmaw),1),
// circular cutoff r^2<W^2+0.6, BILINEAR (soft) binning across 2 adjacent bins
// -- VL_SIFT_BILINEAR_ORIENTATIONS is unconditionally #define'd at
// sift.c:669, not gated behind any COLMAP option (an earlier, wrong
// assumption on my part that a ceiling-comparison investigation caught --
// see report/plan for the diagnostic trail), 6-pass 3-tap box smoothing,
// peaks where h0>0.8*max && local-max, parabolic sub-bin interpolation, up
// to 4 angles returned (VLFeat's own cap; COLMAP's max_num_orientations=2
// cap is applied on the host afterward, matching sift.cc:250-251).
// Clean-room from that BSD source + Lowe's papers; no SiftGPU code
// referenced.
//
// I/O is PACKED (VLFeatKp in, VLFeatOri out, 20 B each) rather than whole
// 576 B SiftPoints -- see cudaSift.h. The keypoint range is now an explicit
// `numKps` instead of the d_PointCounter[2*octave+/-1] window, since the packed
// arrays are already per-octave batches; callers that used the counter protocol
// pass their own count. Output angles are radians in VLFeat's convention (NOT
// degrees) and count is 0-4, exactly as before -- only the container changed.
__global__ void ComputeOrientationsVLFeat(
    const float * __restrict__ d_GradMag, const float * __restrict__ d_GradAng,
    int width, int pitch, int height, int levelStrideElems,
    const VLFeatKp * __restrict__ d_kps, VLFeatOri * __restrict__ d_out, int numKps)
{
  __shared__ float hist[36];
  __shared__ float histSmooth[36];
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float twoPi = 6.283185307179586f;

  for (int bx = blockIdx.x; bx < numKps; bx += gridDim.x) {
    for (int i = tid; i < 36; i += nthreads) hist[i] = 0.0f;
    __syncthreads();

    float x = d_kps[bx].x;
    float y = d_kps[bx].y;
    float sigma = d_kps[bx].sigma;
    int level = d_kps[bx].level;

    float sigmaw = 1.5f * sigma;
    int W = max((int)floorf(3.0f * sigmaw), 1);

    int xi = (int)(x + 0.5f);
    int yi = (int)(y + 0.5f);

    const float *gmag = d_GradMag + level * levelStrideElems;
    const float *gang = d_GradAng + level * levelStrideElems;

    int dyMin = max(-W, -yi), dyMax = min(W, height - 1 - yi);
    int dxMin = max(-W, -xi), dxMax = min(W, width - 1 - xi);
    int nxRange = dxMax - dxMin + 1;
    int nyRange = dyMax - dyMin + 1;
    int total = (nxRange > 0 && nyRange > 0) ? nxRange * nyRange : 0;

    for (int idx = tid; idx < total; idx += nthreads) {
      int ys = dyMin + idx / nxRange;
      int xs = dxMin + idx % nxRange;

      float dx = (float)(xi + xs) - x;
      float dy = (float)(yi + ys) - y;
      float r2 = dx*dx + dy*dy;
      if (r2 >= (float)W*W + 0.6f) continue;

      int px = xi + xs, py = yi + ys;
      float mod = gmag[py*pitch + px];
      float ang = gang[py*pitch + px];
      float wgt = __expf(-r2 / (2.0f * sigmaw * sigmaw));

      // Bilinear (soft) binning across 2 adjacent bins -- VLFeat's actual
      // code path (VL_SIFT_BILINEAR_ORIENTATIONS is unconditionally #define'd
      // at sift.c:669, not gated behind any COLMAP option as I'd wrongly
      // assumed earlier). Matches sift.c:1636-1644 exactly.
      float fbin = 36.0f * ang / twoPi;
      int bin = (int)floorf(fbin - 0.5f);
      float rbin = fbin - (float)bin - 0.5f;
      int bin0 = ((bin % 36) + 36) % 36;
      int bin1 = ((bin + 1) % 36 + 36) % 36;
      atomicAdd(hist + bin0, (1.0f - rbin) * mod * wgt);
      atomicAdd(hist + bin1, rbin * mod * wgt);
    }
    __syncthreads();

    // 6-pass 3-tap box smoothing + peak detection: sequential, single thread
    // (36 elements x 6 passes is trivial next to the accumulation above).
    if (tid == 0) {
      for (int i = 0; i < 36; i++) histSmooth[i] = hist[i];
      for (int iter = 0; iter < 6; iter++) {
        float prev = histSmooth[35];
        float first = histSmooth[0];
        int i;
        for (i = 0; i < 35; i++) {
          float newh = (prev + histSmooth[i] + histSmooth[(i+1) % 36]) / 3.0f;
          prev = histSmooth[i];
          histSmooth[i] = newh;
        }
        histSmooth[i] = (prev + histSmooth[i] + first) / 3.0f;
      }

      float maxh = 0.0f;
      for (int i = 0; i < 36; i++) maxh = fmaxf(maxh, histSmooth[i]);

      int nangles = 0;
      float angles[4];
      for (int i = 0; i < 36 && nangles < 4; i++) {
        float h0 = histSmooth[i];
        float hm = histSmooth[(i - 1 + 36) % 36];
        float hp = histSmooth[(i + 1) % 36];
        if (h0 > 0.8f*maxh && h0 > hm && h0 > hp) {
          float di = -0.5f * (hp - hm) / (hp + hm - 2.0f*h0);
          float th = twoPi * ((float)i + di + 0.5f) / 36.0f;
          angles[nangles++] = th;
        }
      }
      // (Previously these went into SiftPoint's unused non-contiguous fields --
      // sharpness/edgeness/score/ambiguity/match -- which needed by-name writes
      // to avoid clobbering the real orientation. VLFeatOri makes that moot.)
      for (int k = 0; k < 4; k++) d_out[bx].angles[k] = (nangles > k) ? angles[k] : 0.0f;
      d_out[bx].count = nangles;
    }
    __syncthreads();
  }
}

// VLFeat-parity descriptor kernel (Part 2b'). One block per keypoint; window
// size is data-dependent (grows with the keypoint's own sigma), so threads
// grid-stride over the (2W+1)x(2W+1) pixel window. Samples read from the
// PRECOMPUTED per-level gradient buffer for this octave, indexed by the
// keypoint's own assigned level (SiftPoint.empty[0], set on host via
// vl_sift_keypoint_init's exact inversion formula) -- not a shared per-octave
// texture. Mirrors vl_sift_calc_keypoint_descriptor (sift.c:1923-2100):
// magnif=3.0, NBP=4, NBO=8, window sigma=NBP/2, centered 2x2x2 trilinear
// binning, L2->clamp(0.2)->L2 normalization. Clean-room from that BSD source
// + Lowe's papers; no SiftGPU code referenced.
#define VLDESC_NBP 4
#define VLDESC_NBO 8
__global__ void ExtractSiftDescriptorsVLFeat(
    const float * __restrict__ d_GradMag,   // octave-combined, level-stacked planes
    const float * __restrict__ d_GradAng,
    int width, int pitch, int height, int levelStrideElems,
    VLFeatKp * __restrict__ d_kps, float * __restrict__ d_desc, int numKps,
    float subsampling, float magnif)
{
  __shared__ float buffer[VLDESC_NBP*VLDESC_NBP*VLDESC_NBO];  // 128
  __shared__ float partial[256];
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float twoPi = 6.283185307179586f;

  for (int bx = blockIdx.x; bx < numKps; bx += gridDim.x) {
    for (int i = tid; i < 128; i += nthreads) buffer[i] = 0.0f;
    __syncthreads();

    float x = d_kps[bx].x;
    float y = d_kps[bx].y;
    float sigma = d_kps[bx].sigma;
    int level = d_kps[bx].level;
    float angle0 = d_kps[bx].ori;   // already radians in VLFeatKp

    float SBP = magnif*sigma + 1e-10f;
    int W = (int)floorf(sqrtf(2.0f)*SBP*(VLDESC_NBP+1)/2.0f + 0.5f);

    int xi = (int)(x + 0.5f);
    int yi = (int)(y + 0.5f);
    float st0 = sinf(angle0), ct0 = cosf(angle0);

    const float *gmag = d_GradMag + level*levelStrideElems;
    const float *gang = d_GradAng + level*levelStrideElems;

    int dyMin = max(-W, 1-yi), dyMax = min(W, height-yi-2);
    int dxMin = max(-W, 1-xi), dxMax = min(W, width-xi-2);
    int nxRange = dxMax - dxMin + 1;
    int nyRange = dyMax - dyMin + 1;
    int total = (nxRange>0 && nyRange>0) ? nxRange*nyRange : 0;

    for (int idx = tid; idx < total; idx += nthreads) {
      int dyi = dyMin + idx/nxRange;
      int dxi = dxMin + idx%nxRange;

      int px = xi+dxi, py = yi+dyi;
      float mod = gmag[py*pitch+px];
      float angle = gang[py*pitch+px];
      float theta = angle - angle0;
      theta = theta - twoPi*floorf(theta/twoPi);  // wrap to [0,2pi)

      float dx = (float)(xi+dxi) - x;
      float dy = (float)(yi+dyi) - y;
      float nx = ( ct0*dx + st0*dy) / SBP;
      float ny = (-st0*dx + ct0*dy) / SBP;
      float nt = VLDESC_NBO * theta / twoPi;

      float wsigma = VLDESC_NBP / 2.0f;
      float win = __expf(-(nx*nx+ny*ny) / (2.0f*wsigma*wsigma));

      int binx = (int)floorf(nx - 0.5f);
      int biny = (int)floorf(ny - 0.5f);
      int bint = (int)floorf(nt);
      float rbinx = nx - (binx + 0.5f);
      float rbiny = ny - (biny + 0.5f);
      float rbint = nt - bint;

      for (int dbinx = 0; dbinx < 2; dbinx++) {
        for (int dbiny = 0; dbiny < 2; dbiny++) {
          for (int dbint = 0; dbint < 2; dbint++) {
            int bx_ = binx + dbinx, by_ = biny + dbiny;
            if (bx_ >= -(VLDESC_NBP/2) && bx_ < (VLDESC_NBP/2) &&
                by_ >= -(VLDESC_NBP/2) && by_ < (VLDESC_NBP/2)) {
              float weight = win * mod
                  * fabsf(1.0f - dbinx - rbinx)
                  * fabsf(1.0f - dbiny - rbiny)
                  * fabsf(1.0f - dbint - rbint);
              int bt_ = (bint + dbint) % VLDESC_NBO;
              if (bt_ < 0) bt_ += VLDESC_NBO;
              // Write directly in COLMAP's "UBC" orientation-bin order (the
              // convention SiftGPU/Lowe binaries and existing COLMAP DBs
              // use), not VLFeat's native order -- matches
              // TransformVLFeatToUBCFeatureDescriptors exactly
              // (colmap/src/colmap/feature/sift.cc:118-131): q={0,7,6,5,4,3,2,1},
              // i.e. bin 0 fixed, bins 1..7 reversed, per spatial cell.
              // Confirmed empirically: writing native VLFeat order and
              // comparing against a COLMAP-sourced oracle gave cosine ~0.76;
              // this permutation alone brings it to ~0.997.
              int bt_ubc = (VLDESC_NBO - bt_) % VLDESC_NBO;
              int arrIdx = (by_ + VLDESC_NBP/2)*(VLDESC_NBO*VLDESC_NBP)
                         + (bx_ + VLDESC_NBP/2)*VLDESC_NBO + bt_ubc;
              atomicAdd(buffer + arrIdx, weight);
            }
          }
        }
      }
    }
    __syncthreads();

    // Normalize: L2 -> clamp(0.2) -> L2 (same scheme as cusift's existing kernel).
    float val = (tid < 128) ? buffer[tid] : 0.0f;
    partial[tid] = (tid < 128) ? val*val : 0.0f;
    __syncthreads();
    for (int s = nthreads/2; s > 0; s >>= 1) { if (tid<s) partial[tid] += partial[tid+s]; __syncthreads(); }
    float norm1 = sqrtf(partial[0]) + 1e-12f;
    float clamped = (tid < 128) ? fminf(val/norm1, 0.2f) : 0.0f;
    __syncthreads();
    partial[tid] = (tid < 128) ? clamped*clamped : 0.0f;
    __syncthreads();
    for (int s = nthreads/2; s > 0; s >>= 1) { if (tid<s) partial[tid] += partial[tid+s]; __syncthreads(); }
    float norm2 = sqrtf(partial[0]) + 1e-12f;
    if (tid < 128) d_desc[(size_t)bx*128 + tid] = clamped / norm2;
    __syncthreads();

    if (tid == 0) {
      d_kps[bx].x *= subsampling;
      d_kps[bx].y *= subsampling;
      d_kps[bx].sigma *= subsampling;
    }
    __syncthreads();
  }
}

//======= Part 2d: detection (clean-room from colmap/src/thirdparty/VLFeat/ =//
//======= sift.c, BSD; not SiftGPU)                                        =//

// DoG plane = gaussB - gaussA, matches vl_sift_detect's own DoG computation
// (sift.c:1176-1185) exactly -- plain pointwise difference of two adjacent
// Gaussian levels, no separate blur.
__global__ void ComputeDogPlaneKernel(const float *d_GaussA, const float *d_GaussB,
                                      int width, int pitch, int height, float *d_Dog)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  int idx = y*pitch+x;
  d_Dog[idx] = d_GaussB[idx] - d_GaussA[idx];
}

// Coarse (pre-refinement) 26-neighbor DoG extrema scan -- matches
// vl_sift_detect's CHECK_NEIGHBORS macro exactly (sift.c:1199-1232):
// threshold |v| >= 0.8*peak_thresh, strict extremum over all 26 neighbors in
// the 3x3x3 cube (x,y,s). Only the interior is scanned (x,y in [1,w-2],
// s-plane-index in [1, nDogLevels-2]) -- same border margin as VLFeat.
// Candidates are appended (ix, iy, s) to a flat output buffer via an atomic
// counter; s here is the TRUE VLFeat level value (dogPlaneIdx - 1, since
// s_min=-1 and dogPlaneIdx = s - s_min). Refinement (iterative Newton,
// relocation, edge test) is intentionally NOT done on the GPU -- it has a
// small, highly data-dependent iteration count and can relocate to a
// neighboring pixel, which is a poor fit for a parallel kernel; VLFeat
// itself runs this step serially on the CPU. Candidates are cheap (typically
// a few thousand per octave with peak_thresh=0), so the coarse scan (the
// only genuinely parallel-friendly part) runs on the GPU and refinement runs
// on the host in cudaSiftH.cu's DetectKeypointsVLFeat.
__global__ void FindDogExtremaKernel(const float *d_Dog, int width, int pitch, int height,
                                     int nDogLevels, float peakThresh,
                                     int *d_CandX, int *d_CandY, int *d_CandS,
                                     int maxCandidates, unsigned int *d_CandCount)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int sIdx = blockIdx.z + 1;  // dog-plane index, interior range [1, nDogLevels-2]
  if (x >= width-1 || y >= height-1 || sIdx >= nDogLevels-1) return;

  int so = pitch*height;  // s-stride (plane stride) in the DoG cube
  const float *pt = d_Dog + sIdx*so + y*pitch + x;
  float v = *pt;
  float th = 0.8f*peakThresh;

  bool isMax = v >= th;
  bool isMin = v <= -th;
  if (!isMax && !isMin) return;

  // All 26 combinations of (dx,dy,ds) in {-1,0,1}^3 except (0,0,0). Generated
  // programmatically (not hand-listed) after an earlier hand-written version
  // was found to have 4 duplicate entries and 4 missing ones -- the missing
  // (0,+-1,+-1) diagonal neighbors let some non-extremal points pass the
  // strict 26-neighbor test, producing spurious detections (found via a
  // direct numpy ground-truth diff, see plan).
  const int dxs[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};
  const int dys[26] = {-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1};
  const int dss[26] = {-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1};
  #pragma unroll
  for (int k = 0; k < 26; k++) {
    float nv = *(pt + dxs[k] + dys[k]*pitch + dss[k]*so);
    if (isMax && v <= nv) { isMax = false; }
    if (isMin && v >= nv) { isMin = false; }
  }
  if (!isMax && !isMin) return;

  unsigned int idx = atomicAdd(d_CandCount, 1);
  if ((int)idx < maxCandidates) {
    d_CandX[idx] = x;
    d_CandY[idx] = y;
    d_CandS[idx] = sIdx - 1;  // true VLFeat level s = dogPlaneIdx - 1 (s_min=-1)
  }
}

