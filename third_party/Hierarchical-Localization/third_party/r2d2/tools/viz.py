# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import numpy as np
import matplotlib.pyplot as pl


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    
    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    
    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    
    Copied from https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
    Copyright (c) 2018 Tom Runia
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)



def show_flow( img0, img1, flow, mask=None ):
    img0 = np.asarray(img0)
    img1 = np.asarray(img1)
    if mask is None: mask = 1
    mask = np.asarray(mask)
    if mask.ndim == 2: mask = mask[:,:,None]
    assert flow.ndim == 3
    assert flow.shape[:2] == img0.shape[:2] and flow.shape[2] == 2
    
    def noticks():
      pl.xticks([])
      pl.yticks([])
    fig = pl.figure("showing correspondences")
    ax1 = pl.subplot(221)
    ax1.numaxis = 0
    pl.imshow(img0*mask)
    noticks()
    ax2 = pl.subplot(222)
    ax2.numaxis = 1
    pl.imshow(img1)
    noticks()
    
    ax = pl.subplot(212)
    ax.numaxis = 0
    flow_img = flow_to_color(np.where(np.isnan(flow), 0, flow))
    pl.imshow(flow_img * mask)
    noticks()
    
    pl.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.02, hspace=0.02)
    
    def motion_notify_callback(event):
      if event.inaxes is None: return
      x,y = event.xdata, event.ydata
      ax1.lines = []
      ax2.lines = []
      try:
        x,y = int(x+0.5), int(y+0.5)
        ax1.plot(x,y,'+',ms=10,mew=2,color='blue',scalex=False,scaley=False)
        x,y = flow[y,x] + (x,y)
        ax2.plot(x,y,'+',ms=10,mew=2,color='red',scalex=False,scaley=False)
        # we redraw only the concerned axes
        renderer = fig.canvas.get_renderer()
        ax1.draw(renderer)
        ax2.draw(renderer)
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)
      except IndexError:
        return
  
    cid_move = fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)
    print("Move your mouse over the images to show matches (ctrl-C to quit)")
    pl.show()


