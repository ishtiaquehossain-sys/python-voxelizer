import os
import sys
import shutil
import argparse
import numpy as np
import scipy as sp
import tqdm
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = kwargs.get('disable', True)
        super().__init__(*argv, **kwargs)
tqdm.tqdm = _TQDM
import mcubes
import trimesh
from trimesh.voxel.creation import voxelize
from voxelfuse.voxel_model import VoxelModel
from voxelfuse.mesh import Mesh
from voxelfuse.primitives import generateMaterials
import concurrent.futures
from npytar import NpyTarWriter


def create_directories(output_dir, obj_dir, stl_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if obj_dir is not None:
        os.makedirs(obj_dir)
    if stl_dir is not None:
        os.makedirs(stl_dir)


def load_meshes(pml_dir, obj_file_list):
    files = None
    meshes = []
    extents = []
    with open(os.path.join(pml_dir, obj_file_list), "r") as ins:
        files = ins.read().split("\n")
        files = list(filter(None, files))
    print("Found {} files.".format(len(files)))
    for file in files:
        obj_file = os.path.join(pml_dir, file)
        mesh = trimesh.load(obj_file)
        mesh.apply_translation(-mesh.centroid)
        meshes.append(mesh)
        extents.append(mesh.extents)
    max_extent = np.array(extents).max()
    return meshes, max_extent


def voxelize_mesh(mesh, idx, pitch, resolution, obj_dir, stl_dir):
    voxel = voxelize(mesh=mesh, pitch=pitch)
    max_side = np.max(voxel.matrix.shape)
    if max_side > resolution:
        scale_factor =  resolution / float(max_side)
        dim0 = int(voxel.matrix.shape[0] * scale_factor)
        dim1 = int(voxel.matrix.shape[1] * scale_factor)
        dim2 = int(voxel.matrix.shape[2] * scale_factor)
        voxel = voxel.revoxelized((dim0, dim1, dim2))
    voxel_arr = voxel.matrix.astype(int)
    x_1 = (resolution - voxel_arr.shape[0]) // 2
    x_2 = resolution - voxel_arr.shape[0] - x_1
    y_1 = (resolution - voxel_arr.shape[1]) // 2
    y_2 = resolution - voxel_arr.shape[1] - y_1
    z_1 = (resolution - voxel_arr.shape[2]) // 2
    z_2 = resolution - voxel_arr.shape[2] - z_1
    voxel_arr = np.pad(voxel_arr, ((x_1, x_2), (y_1, y_2), (z_1, z_2)), 'constant', constant_values=0)
    voxel_arr = sp.ndimage.binary_fill_holes(voxel_arr).astype(int)
    if obj_dir is not None:
        v, f = mcubes.marching_cubes(voxel_arr, 0.5)
        mcubes.export_obj(v, f, os.path.join(obj_dir, 'voxelized_mesh_{}.obj'.format(idx)))
    if stl_dir is not None:
        model = VoxelModel(voxel_arr, generateMaterials(4))  # 4 is aluminium.
        mesh = Mesh.fromVoxelModel(model)
        mesh.export(os.path.join(stl_dir, 'voxelized_mesh_{}.stl'.format(idx)))
    return voxel_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64, help='voxel grid size')
    parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
    parser.add_argument("--output_dir", type=str, default='outputs', help="voxel save directory")
    parser.add_argument("--output_obj", type=bool, default=False, help="obj files are generated")
    parser.add_argument("--output_stl", type=bool, default=False, help="stl files are generated")
    args = parser.parse_args()

    pml_dir = os.path.join('..','Procedural-Modeling-Library', 'code', 'Release', 'Outputs')
    obj_file_list = 'mesh_data.txt'
    output_dir = args.output_dir
    resolution = args.resolution
    num_workers = args.num_workers
    tar_file_name = 'shapenet10_test.tar'
    obj_dir = os.path.join(output_dir, 'obj') if args.output_obj else None
    stl_dir = os.path.join(output_dir, 'stl') if args.output_stl else None

    create_directories(output_dir, obj_dir, stl_dir)
    meshes, max_extent = load_meshes(pml_dir, obj_file_list)
    pitch = float(max_extent) / args.resolution

    writer = NpyTarWriter(os.path.join(output_dir, tar_file_name))
    print('Voxelizing...')
    with tqdm.tqdm(total=len(meshes), disable=False, file=sys.stdout) as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, mesh in enumerate(meshes):
                futures = []
                future = executor.submit(voxelize_mesh, mesh, i, pitch, resolution, obj_dir, stl_dir)
                name = '{:03d}.{}.{:03d}'.format(3, i, 1)
                writer.add(future.result(), name)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
    writer.close()
