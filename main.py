import numpy as np
use_cuda = False
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

import trimesh
import os
import open3d as o3d
import pyvista as pv
import trimesh.transformations as tra

from tqdm import tqdm
from pathlib import Path
from tkinter import filedialog, Tk
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from trimesh import util
from probreg import cpd

def split_path(file_path):
    """
    Splits a file path into its directory, name, and extension.

    Parameters:
    file_path (str): The full path to the file.

    Returns:
    A tuple containing the directory, name, and extension of the file.
    """
    directory, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    return directory, name, ext

def load_stl(file_path):
    """
    Loads an STL file into a Trimesh object recentered around the origin.

    Args:
        file_path (str): The full path to the STL file.

    Returns:
        A Trimesh object.
    """
    mesh = trimesh.load_mesh(file_path)
    mesh.vertices -= mesh.centroid
    return mesh

def decimate_mesh(mesh, fraction, n_jobs=8, max_iterations=100):
    def create_lightweight_mesh(vertices, faces):
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def simplify_quadric_decimation_chunk(faces_chunk):
        mesh_copy = create_lightweight_mesh(mesh.vertices, faces_chunk)
        return mesh_copy.simplify_quadric_decimation(int(faces_chunk.shape[0] * 0.9))

    target_faces = int(mesh.faces.shape[0] * fraction)

    with tqdm(total=target_faces, desc="Decimating mesh", ncols=80) as pbar:
        iteration = 0
        while mesh.faces.shape[0] > target_faces:
            iteration += 1
            if iteration > max_iterations:
                break

            faces_chunks = np.array_split(mesh.faces, n_jobs)

            new_meshes = Parallel(n_jobs=n_jobs)(
                delayed(simplify_quadric_decimation_chunk)(chunk)
                for chunk in faces_chunks)

            mesh = util.concatenate(new_meshes)
            pbar.update(int(mesh.faces.shape[0] - target_faces))

    return mesh

def plot_meshes_and_points(mesh_list, points_list=None, 
                           mesh_colors=None, point_colors=None,
                           opacities=None, title=None):
    # Create a Pyvista plotter
    plotter = pv.Plotter()

    # Set default colors and opacities if not provided
    if mesh_colors is None:
        mesh_colors = ['red', 'blue'] * len(mesh_list)
    if opacities is None:
        opacities = [0.9] * len(mesh_list)

    # Convert Trimesh objects to Pyvista objects and add them to the plotter
    for mesh, color, opacity in zip(mesh_list, mesh_colors, opacities):
        pv_mesh = pv.wrap(mesh)
        plotter.add_mesh(pv_mesh, color=color, opacity=opacity)

    # Add the points to the plotter if points_list is provided
    if points_list is not None:
        if point_colors is None:
            point_colors = ['green', 'yellow'] * len(points_list)
        
        for points, color in zip(points_list, point_colors):
            plotter.add_mesh(
                pv.PolyData(points), color=color, point_size=5,
                render_points_as_spheres=True
            )

    # Set the title
    if title is not None:
        plotter.add_text(title, font_size=20, name='title')

    # Show the plot
    plotter.show()

def pre_rotate(mesh, angle):
    """
    Rotate a mesh around the z-axis by a given angle (in degrees).
    """
    T = tra.rotation_matrix(np.radians(angle), [0, 0, 1])
    mesh.apply_transform(T)

def sample_points_in_cylinders(mesh, cylinders, num_points, sampling_factor=3):
    """
    Sample points from the surface of the mesh within multiple cylinders.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        cylinders (list of tuples): A list of cylinders, each defined by a tuple
                                     (center, height, radius) where center is a
                                     list or numpy array representing the center
                                     of the cylinder [x, y, z], height is the
                                     height of the cylinder, and radius is the
                                     radius of the cylinder.
        num_points (int): The total number of points to sample, distributed evenly
                          among the cylinders.
        sampling_factor (int): A multiplier for the number of points to sample
                               initially from the mesh. A higher value increases
                               the chance of finding points within cylinders.

    Returns:
        A numpy array of shape (num_points, 3) containing the sampled points
        from the surface of the mesh within the specified cylinders.
    """
    points = []

    # Get all points on the surface of the mesh
    all_points, _ = trimesh.sample.sample_surface(mesh, num_points * sampling_factor)
    combined_points = np.empty((0, 3))

    # get all points within cylinders, not less that num_points
    while combined_points.shape[0] < num_points:
        for c in cylinders:
            # Get the center, height, and radius of the cylinder
            center, height, radius = c

            # Select points that are within the cylinder
            distances_xy = np.linalg.norm(all_points[:, :2] - center[:2], axis=1)
            z_differences = np.abs(all_points[:, 2] - center[2])
            points_in_cylinder = all_points[(distances_xy <= radius) & 
                                            (z_differences <= height / 2)]

            points.append(points_in_cylinder)

        # Combine all points from all cylinders
        combined_points = np.vstack(points)
        sampling_factor += 1

    # Randomly select a subset of points if there are more points than needed
    if combined_points.shape[0] > num_points:
        indices = np.random.choice(
            combined_points.shape[0], num_points, replace=False
        )

        combined_points = combined_points[indices]

    return combined_points

    
def cpd_registration(source_mesh, target_mesh, cylinders,
                     samples, max_iterations=50, tolerance=1e-5,
                     use_cuda=use_cuda):
    """
    Perform Coherent Point Drift (CPD) registration on two point clouds.

    Args:
        source_mesh (trimesh.mesh): Source mesh
        target_mesh (trimesh.mesh): Target mesh
        max_iterations (int, optional): Maximum number of iterations. Default is 50.
        tolerance (float, optional): Tolerance for convergence. Default is 1e-5.

    Returns:
        numpy.ndarray: The transformation matrix between the source_mesh and target_mesh point clouds.
    """
    target_points_all = sample_points_in_cylinders(target_mesh, cylinders, samples)
    source_points_all = sample_points_in_cylinders(source_mesh, cylinders, samples)

    # Convert numpy arrays to open3d point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points_all)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points_all)

    # Perform CPD registration
    tf_param, _, _ = cpd.registration_cpd(
        source_pcd, target_pcd,
        maxiter=max_iterations,
        tol=tolerance,
        use_cuda=use_cuda
    )

    # Extract the rotation and translation components
    rotation_matrix = tf_param.rot
    translation_vector = tf_param.t

    # Compose the transformation matrix
    transformation_matrix = np.eye(4)
    if use_cuda:
        transformation_matrix[:3, :3] = rotation_matrix.get()
        transformation_matrix[:3, 3] = translation_vector.get()
    else:
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

    # Return the transformation matrix
    return transformation_matrix

def save_stl(mesh, file_path):
    mesh.export(file_path)


if __name__ == "__main__":
    root_path = Path.cwd()
    raw_file_path = root_path / "raw_scans"
    desampled_scans_path = root_path / "desampled_scans"
    
    # ask user what they want to do 
    print("Desampling a model or performing a best fit of two models?")
    print("1. Desampling a model")
    print("2. Performing a best fit of two models")
    print("3. Pre-rotate a model")
    user_input = input("Enter 1, 2, or 3: ")

    root = Tk()
    root.withdraw()
    root.attributes("-top", 1)

    if user_input == "1":
        # ask user to select folder path using filedialog
        input_folder_path = filedialog.askdirectory(
            initialdir=raw_file_path,
            title="Select folder"
        )

        output_folder_path = desampled_scans_path

        # Set the fraction of faces to keep (e.g., 0.5 to keep half of the faces)
        FRACTION_FACES = 0.5

        # Get all STL files in the folder
        stl_files = [f for f in Path(input_folder_path).glob("*.stl")]

        for stl_file in stl_files:
            # get file name from stl_file
            file_name = stl_file.name
            output_file_path = output_folder_path / f'desamp_{file_name}'
            mesh = load_stl(stl_file)
            decimated_mesh = decimate_mesh(mesh, FRACTION_FACES)
            save_stl(decimated_mesh, output_file_path)

    elif user_input == "2":
        # Ask user to select two STL files using filedialog
        input_file_paths = filedialog.askopenfilenames(
            title="Select two STL files",
            initialdir=desampled_scans_path,
            filetypes=(("stl files", "*.stl"),),
            multiple=True
        )

        if len(input_file_paths) < 2:
            print(("Error: You selected less than 2 STL files."
                   " Please select exactly 2 STL files."))
        elif len(input_file_paths) > 2:
            print(("Error: You selected more than 2 STL files."
                   " Please select exactly 2 STL files."))
        else:
            target_file_path = input_file_paths[0]
            target_name = split_path(target_file_path)[1]
            model2_file_path = input_file_paths[1]
            model2_name = split_path(model2_file_path)[1]

            # load and visualized raw models, post-centering
            target_mesh = load_stl(target_file_path)
            source_mesh = load_stl(model2_file_path)
            
            # pre-rotate source_mesh and visualize
            ROT = -30.0
            pre_rotate(source_mesh, ROT)
            plot_meshes_and_points(
                mesh_list=[target_mesh, source_mesh],
                title=(
                    f"Pre-rotated source_mesh model by {ROT}"
                    " degrees"
                )
            )

            # create boundary cylinder for targeted alignment and visualize
            cylinder_coords = np.array([4, 17, 0])
            height = 12
            radius = 6.5
            cylinder_1 = [cylinder_coords, height, radius]
            cylinder_2 = [-cylinder_coords, height, radius]
            cylinders = (cylinder_1, cylinder_2)
            NUM_POINTS = 10000
            
            # Sample points within the cylinders
            target_points_all = sample_points_in_cylinders(target_mesh, cylinders, NUM_POINTS)
            source_points_all = sample_points_in_cylinders(source_mesh, cylinders, NUM_POINTS)

            plot_meshes_and_points(
                points_list=[target_points_all, source_points_all],
                mesh_list=[target_mesh, source_mesh],
                title=(
                    f"AOI cylinders, 1: {cylinder_coords}, "
                    f"2:{-cylinder_coords}\nheight: {height}, "
                    f"radius: {radius}"
                )
            )

            # perform CPD registration and visualize
            tran_matrix = cpd_registration(source_mesh, target_mesh, cylinders, samples=NUM_POINTS)
            source_mesh.apply_transform(tran_matrix)
            plot_meshes_and_points(
                mesh_list=[target_mesh, source_mesh],
                title= "Aligned Models"
            )
            
            # combine and save models
            combined_model = trimesh.util.concatenate(target_mesh, source_mesh)
    
            output_file_path = (
                root_path / "aligned_scans" / 
                f"{target_name}_{model2_name}_aligned.stl"
            )

            save_stl(combined_model, output_file_path)

    elif user_input == "3":
        # ask user to select file path using filedialog
        input_file_path = filedialog.askopenfilename(
            initialdir=desampled_scans_path,
            title="Select file", 
            filetypes=(("stl files", "*.stl"),)
        )

        # Load the mesh from file
        mesh = trimesh.load(input_file_path)

        # Pre-rotate the mesh around the z-axis by 45 degrees
        pre_rotate(mesh, -120)

        # Save the rotated mesh to file
        export_path = root_path / "pre-rotated_scans" / "rotated.stl"
        mesh.export(export_path)

    else:
        print("Invalid input")
        print("Rerun script and make a valid selection")
