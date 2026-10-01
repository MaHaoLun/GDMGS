#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import sys
import cv2
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    
    def process_frame(idx, key):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        return CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)

    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每个帧到执行器进行处理
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        # 使用as_completed()获取已完成的任务
        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            cam_infos.append(cam_info)
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_extrinsics)}"+Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()

    cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]

        # Helper: get relative image path from multiple possible keys
        def _get_frame_relpath(frame):
            for key in [
                "file_path", "image_path", "img_path", "file", "filename", "name"
            ]:
                if key in frame and isinstance(frame[key], str) and len(frame[key]) > 0:
                    return frame[key]
            return None

        # Helper: get transform matrix from frame (supports 'transform_matrix' or 'rot_mat')
        def _get_transform_matrix(frame):
            if "transform_matrix" in frame:
                return np.array(frame["transform_matrix"])  # camera-to-world
            if "rot_mat" in frame:
                return np.array(frame["rot_mat"])  # assume same convention as c2w
            raise KeyError("Neither 'transform_matrix' nor 'rot_mat' found in frame")

        # Decide how to resolve image paths
        sample_rel = _get_frame_relpath(frames[0])
        images_by_index = None
        use_index_mapping = False
        if sample_rel is not None:
            # If relative path already includes extension, do not append
            if sample_rel.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png', 'bmp', 'tif', 'tiff', 'webp']:
                extension = ""

            def resolve_image_path(frame):
                cam_name = _get_frame_relpath(frame) + extension
                return os.path.join(path, cam_name)
        else:
            # No explicit image path in frames: scan for images and map by frame_index or order
            candidate_dirs = [path, os.path.join(path, "images"), os.path.join(path, "rgb"), os.path.join(path, "imgs")]
            exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.bmp", "*.tif", "*.tiff", "*.webp"]
            found = []
            for d in candidate_dirs:
                if os.path.isdir(d):
                    for e in exts:
                        found.extend(glob.glob(os.path.join(d, e)))
                if len(found) > 0:
                    break
            if len(found) == 0:
                raise ValueError("Could not find any images under scene folder. Provide file paths in JSON or place images under scene/images|rgb|imgs.")

            # Natural sort: try numeric stem, fallback to lexicographic
            def _stem_key(p):
                s = Path(p).stem
                try:
                    return int(s)
                except:
                    return s
            found = sorted(found, key=_stem_key)

            # Build index mapping if frame_index provided
            all_have_index = all("frame_index" in fr for fr in frames)
            if all_have_index:
                max_idx = max(int(fr["frame_index"]) for fr in frames)
                if max_idx >= len(found):
                    raise ValueError(f"frame_index max {max_idx} exceeds available images {len(found)}")
                images_by_index = {int(i): found[int(i)] for i in range(len(found))}
                use_index_mapping = True
            else:
                # Use order mapping
                if len(found) < len(frames):
                    raise ValueError(f"Found {len(found)} images but {len(frames)} frames in JSON")
                images_by_index = {i: found[i] for i in range(len(frames))}
                use_index_mapping = True

            def resolve_image_path(frame):
                if use_index_mapping:
                    idx = int(frame.get("frame_index", -1))
                    if idx == -1:
                        # fallback to position if index missing for this frame
                        # note: this path requires us to pass enumerate index; handled below
                        raise RuntimeError("Index mapping requires frame_index; internal usage will provide enumerate index.")
                    return images_by_index[idx]
                raise RuntimeError("Unexpected resolver state")

        def process_frame(idx, frame):
            # Resolve image path (handle resolver that expects index mapping)
            try:
                image_path = resolve_image_path(frame)
            except RuntimeError:
                # Fallback for order mapping without frame_index
                image_path = images_by_index[idx]

            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist!")

            # Camera-to-world transform
            c2w = _get_transform_matrix(frame)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(image_path).stem
            image = Image.open(image_path)

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            return CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
    ct = 0
    progress_bar = tqdm(frames, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]
        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            cam_infos.append(cam_info)
            ct += 1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)

        progress_bar.close()
    
    cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8, warmup_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = images
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir))
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if warmup_ply_path is not None:
        print(warmup_ply_path)
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        # try:
        print(f'start fetching data from ply file')
        pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfo(path, eval, extension=".png", warmup_ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if warmup_ply_path is not None:
        print(f'fetching data from warmup ply file')
        pcd = fetchPly(warmup_ply_path)
    else:
        ply_paths = glob.glob(os.path.join(path, "*.ply"))
        if len(ply_paths)==0:
            ply_path = os.path.join(path, "points3d.ply")
            # Since this data set has no colmap data, we start with random points
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            colors = np.random.random((num_pts, 3))
            normals=np.zeros((num_pts, 3))
            pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

            storePly(ply_path, xyz, colors*255)
        else:
            ply_path = ply_paths[0]
            pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCityInfo(path, eval, llffhold=8, extension=".png", warmup_ply_path=None):
    
    json_path = glob.glob(os.path.join(path, f"transforms.json"))[0].split('/')[-1]
    print("Reading Training Transforms from {}".format(json_path))
    
    # load ply
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        las_paths = glob.glob(os.path.join(path, "LAS/*.las"))
        las_path = las_paths[0]
        print(f'las_path: {las_path}')
        try:
            pcd = read_multiple_las_files(las_paths, ply_path)
        except:
            raise ValueError("Load LAS failed!")
    
    # load camera
    cam_infos = readCamerasFromTransforms(path, json_path, extension)
    
    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []
    test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(cam_infos)
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo
}