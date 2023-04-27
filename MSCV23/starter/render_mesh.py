"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --cow_path data/cow.obj
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio
import pdb
import cv2

from pytorch3d.renderer import look_at_view_transform
from starter.utils import get_device, get_mesh_renderer
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection


def render_cow2(ratio,
    cow_path="data/cow.obj", image_size=256, device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()
    
    mesh = pytorch3d.io.load_objs_as_meshes([cow_path])
    mesh = mesh.to(device)
    mesh.scale_verts_(5)

    verts = mesh.verts_list()[0]

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    Rs = [None] * 5
    Ts = [None] * 5

    # azim: from (0, 0, 1) to projected vector, counter clockwise
    # the return is camera view coordinate https://pytorch3d.org/docs/cameras
    # also need to count pytorch3D convention: X'=XR + T, not OpenCV RX + T
    Rs[0], Ts[0]=look_at_view_transform(10, 0, 270, up=((0, 0, 1),), device=device)
    Rs[1], Ts[1]=look_at_view_transform(10, -90, 0, up=((0, 0, 1),), device=device)
    Rs[2], Ts[2]=look_at_view_transform(10, 0, 90, up=((0, 0, 1),), device=device)
    Rs[3], Ts[3]=look_at_view_transform(10, 90, 0, up=((0, 0, 1),), device=device)
    Rs[4], Ts[4]=look_at_view_transform(10, 0, 0, up=((0, 1, 0),), device=device)

    Rs[0], Ts[0]=look_at_view_transform(10, 0, 0, up=((0, 1, 0),), device=device)
    Rs[1], Ts[1]=look_at_view_transform(10, 0, 50, up=((0, 1, 0),), device=device)
    Rs[2], Ts[2]=look_at_view_transform(10, 30, 30, up=((0, 1, 0),), device=device)
    Rs[3], Ts[3]=look_at_view_transform(10, 10, 40, up=((0, 1, 0),), device=device)
    Rs[4], Ts[4]=look_at_view_transform(10, 20, 20, up=((0, 1, 0),), device=device)

    # height, width
    image_size=torch.tensor([[2048, 4096]], device=device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0

    # unit: physical
    focal_length=torch.tensor([[1000, 1050]], device=device) / scale
    principal_point=image_size.flip(1) / scale / 2.0

    rends = []
    KRTs = []
    vmaps = []
    for i in range(5):
        scalepp = (1.0 - ratio[i * 3 + 1: i*3 + 3]).unsqueeze(0)
        
        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=Rs[i], T=Ts[i], focal_length=focal_length * ratio[i * 3], principal_point=scalepp * principal_point, image_size=image_size,
            device=device
        )

        vmap = checkVisible(cameras, image_size[0], mesh)
        vmaps.append(vmap)

        R, T, K = opencv_from_cameras_projection(cameras, image_size=image_size)
        KRTs.append((K[0].type(torch.double), R[0].type(torch.double), T.T.type(torch.double)))

        rend = renderer(mesh, cameras=cameras, lights=lights)
        # convert from [0,1] to [0, 255]
        rend = (rend * 255).type(torch.uint8)

        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to CPU (if needed).
        rends.append(rend)
    return rends, KRTs, verts, vmaps


def skew(t):
    """ t is 3x1
    """
    return torch.tensor([[0, -t[2, 0], t[1, 0]],
                     [t[2, 0], 0, -t[0, 0]],
                     [-t[1, 0], t[0, 0], 0]], device=t.device)

# left is identity
def RightRelativeToLeft(RlTl, RrTr):
    Rl, Tl = RlTl
    Rr, Tr = RrTr
    # pad = torch.tensor([[0, 0, 0, 1]], dtype=torch.double, device=Rl.device)
    # T1w=torch.concatenate([Rl, Tl], axis=1)
    # T1w=torch.concatenate([T1w, pad], axis=0)
    
    # T2w=torch.concatenate([Rr, Tr], axis=1)
    # T2w=torch.concatenate([T2w, pad], axis=0)

    #2 to 1
    # T12 = T1w @ torch.linalg.pinv(T2w)
    # return T12[:3, :3], T12[:3, 3:4]

    Rnew=Rl@(Rr.T)
    Tnew=Tl-Rnew@Tr
    return Rnew, Tnew

def Essential(RT):
    """
    Args:
        RT: a pair
    """
    return skew(RT[1])@RT[0]

# K0 is reference
def Fundamental(E, K0, K1):
    return torch.linalg.pinv(K1).T @ E @ torch.linalg.pinv(K0)


def checkVisible(cameras, img_size, meshes):
    raster_settings = pytorch3d.renderer.mesh.rasterizer.RasterizationSettings(
        image_size=(2048, 4096), 
        blur_radius=0.0,
        faces_per_pixel=1, 
    )
    rasterizer = pytorch3d.renderer.mesh.rasterizer.MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )

    # Get the output from rasterization
    fragments = rasterizer(meshes)

    # pix_to_face is of shape (N, H, W, 1)
    pix_to_face = fragments.pix_to_face  

    # (F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = meshes.faces_packed() 
    # (V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = meshes.verts_packed() 
    vertex_visibility_map = torch.zeros(packed_verts.shape[0])   # (V,)

    # Indices of unique visible faces
    visible_faces = pix_to_face.unique()   # (num_visible_faces )

    # Get Indices of unique visible verts using the vertex indices in the faces
    visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
    unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )

    # Update visibility indicator to 1 for all visible vertices 
    vertex_visibility_map[unique_visible_verts_idx] = 1.0
    return vertex_visibility_map


def checkFundamental(pt1, pt2, F):
    ones = torch.ones([1, pt1.shape[1]], device=get_device())
    pt1s = torch.cat([pt1, ones], axis=0)
    pt2s = torch.cat([pt2, ones], axis=0)

    # WRONG, FIXME: transpose
    res = torch.sum((F @ pt2s) * pt1s, axis=0)
    # pdb.set_trace()
    print(torch.mean(torch.abs(res)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/Q6.gif")
    parser.add_argument("--image_size", type=int, default=1024)
    args = parser.parse_args()

    # These are the 15 variables we are going to optimize
    ratio = torch.tensor([1, 1, 1, 
            0.95, 0.9, 0.9, 
            1.05, 0.95, 0.95, 
            0.9, 1.05, 1.05,
            1.1, 1.15, 1.15], device=get_device())

    # ratio = torch.tensor([1, 1.1, 1.1, 
    #         1, 1., 1., 
    #         1., 1., 1., 
    #         1, 1., 1.,
    #         1., 1., 1.], device=get_device())
    
    images, KRTs, verts, vmaps = render_cow2(ratio, cow_path=args.cow_path, image_size=args.image_size)
    for i, im in enumerate(images):
        plt.imsave(str(i) + '.jpg', im)

    E = torch.zeros((4, 5, 3, 3), dtype=torch.double,      device = get_device())
    F = torch.zeros((4, 5, 3, 3), dtype=torch.double,       device = get_device())
    for i in range(4):
        for j in range(i + 1, 5):
            RtoL = RightRelativeToLeft(KRTs[i][1:], KRTs[j][1:])
            E[i, j] = Essential(RtoL)
            # BUG!!!
            F[i, j] = Fundamental(E[i, j], KRTs[j][0], KRTs[i][0])
            # F[i, j] = Fundamental(E[i, j], KRTs[i][0], KRTs[j][0])

            # compute F from openCV, only assume 2d correspondences are known
            vi = vmaps[i]
            vj = vmaps[j]
            vij = torch.logical_and(vi, vj)
            vv = verts[vij].T.type(torch.double)
            # vv = verts.T
            vj2d = KRTs[j][0] @ (KRTs[j][1] @ vv + KRTs[j][2])
            vj2d = vj2d[:2, :] / vj2d[2:, :]

            vi2d = KRTs[i][0] @ (KRTs[i][1] @ vv + KRTs[i][2])
            vi2d = vi2d[:2, :] / vi2d[2:, :]

            vicv = vi2d.T.cpu().numpy()
            vjcv = vj2d.T.cpu().numpy()
            # Fcv, mask = cv2.findFundamentalMat(vjcv, vicv, cv2.FM_RANSAC)
            Fcv, mask = cv2.findFundamentalMat(vjcv, vicv, cv2.FM_LMEDS)
            Fcv = torch.tensor(Fcv, device=get_device(), dtype=torch.double)
            # print(Fcv, i, j)
            # print("Ground Truth:", F[i,j]/F[i,j][2,2])

            # checkFundamental(vi2d, vj2d, F[i, j])
            # checkFundamental(vi2d, vj2d, Fcv)
            # pdb.set_trace()
            # scale = F[i,j][2,2]
            # print(scale)
            # F[i,j] = torch.tensor(Fcv, device=get_device())*scale
            F[i,j] = torch.tensor(Fcv, device=get_device())

    E = E.cpu().numpy()
    F = F.cpu().numpy()
    def cost(f, E, F, i, j, id):
        #BIG!!!
        K1 = np.array([[1000*f[3*i], 0, 2048*f[3*i+1]], [0, 1050*f[3*i], 1024*f[3*i+2]], [0, 0, 1]])
        K0 = np.array([[1000*f[3*j], 0, 2048*f[3*j+1]], [0, 1050*f[3*j], 1024*f[3*j+2]], [0, 0, 1]])

        ff = K1.T @ F @ K0
        def costxx(scale):
            delta = ff * scale - E 
            return np.sum(delta ** 2)
        ret = minimize(costxx, 1.0)
        delta = ff * ret.x - E 
        return np.sum(delta ** 2) 

    def cost2(f):
        c = 0
        id = 15
        for i in range(4):
            for j in range(i + 1, 5):
                c += cost(f, E[i, j], F[i, j], i, j, id)
                id+=1
        return c
    from scipy.optimize import minimize    
    ret = minimize(cost2, np.ones(15))
    print(ret.x[:15].reshape(-1, 3))
