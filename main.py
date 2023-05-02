import trimesh
import os
import numpy as np
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
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

def plot_meshes(meshes, colors=None, opacities=None, title=None):
    # Create a Pyvista plotter
    plotter = pv.Plotter()

    # Set default colors and opacities if not provided
    if colors is None:
        colors = ['red', 'blue'] * len(meshes)
    if opacities is None:
        opacities = [0.9] * len(meshes)

    # Convert Trimesh objects to Pyvista objects and add them to the plotter
    for mesh, color, opacity in zip(meshes, colors, opacities):
        pv_mesh = pv.wrap(mesh)
        plotter.add_mesh(pv_mesh, color=color, opacity=opacity)

    # Set the title
    if title is not None:
        plotter.add_text(title, font_size=20, name='title')

    # Show the plot
    plotter.show()

def plot_points_and_meshes(points_list, meshes, point_colors=None, mesh_colors=None, opacities=None, title=None):
    # Create a Pyvista plotter
    plotter = pv.Plotter()

    # Set default colors and opacities if not provided
    if point_colors is None:
        point_colors = ['green', 'yellow'] * len(points_list)
    if mesh_colors is None:
        mesh_colors = ['red', 'blue'] * len(meshes)
    if opacities is None:
        opacities = [0.9] * len(meshes)

    # Add the points to the plotter
    for points, color in zip(points_list, point_colors):
        plotter.add_mesh(pv.PolyData(points), color=color, point_size=5, render_points_as_spheres=True)

    # Convert Trimesh objects to Pyvista objects and add them to the plotter
    for mesh, color, opacity in zip(meshes, mesh_colors, opacities):
        pv_mesh = pv.wrap(mesh)
        plotter.add_mesh(pv_mesh, color=color, opacity=opacity)

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

def create_cylinder(center, height, radius, num_segments=36):
    """
    Create a cylinder mesh centered at the given point.

    Args:
        center (numpy array): The center of the cylinder [x, y, z].
        height (float): The height of the cylinder.
        radius (float): The radius of the cylinder.
        num_segments (int): The number of segments in the cylinder.

    Returns:
        A Trimesh object representing the cylinder.
    """
    cylinder = trimesh.creation.cylinder(height=height, radius=radius, 
                                         sections=num_segments)
    cylinder.vertices += center
    return cylinder

def sample_points_in_cylinder(mesh, center, height, radius, num_points):
    """
    Sample points from the surface of the mesh within a cylinder.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        center (list or numpy array): The center of the cylinder [x, y, z].
        height (float): The height of the cylinder.
        radius (float): The radius of the cylinder.
        num_points (int): The number of points to sample.

    Returns:
        A numpy array of shape (num_points, 3) containing the sampled points.
    """
    # Get all points on the surface of the mesh
    all_points, _ = trimesh.sample.sample_surface(mesh, num_points * 10)

    # Select points that are within the cylinder
    distances_xy = np.linalg.norm(all_points[:, :2] - center[:2], axis=1)
    z_differences = np.abs(all_points[:, 2] - center[2])
    points_in_cylinder = all_points[(distances_xy <= radius) & (z_differences <= height / 2)]

    # Randomly select a subset of points if there are more points than needed
    if points_in_cylinder.shape[0] > num_points:
        indices = np.random.choice(points_in_cylinder.shape[0], num_points, replace=False)
        points_in_cylinder = points_in_cylinder[indices]

    return points_in_cylinder

def cpd_registration(source, center1, height1, radius1,
                     target, center2, height2, radius2, 
                     samples, max_iterations=50, tolerance=1e-5,
                     use_cuda=use_cuda):
    """
    Perform Coherent Point Drift (CPD) registration on two point clouds.

    Args:
        source (trimesh.mesh): Source mesh
        target (trimesh.mesh): Target mesh
        max_iterations (int, optional): Maximum number of iterations. Default is 50.
        tolerance (float, optional): Tolerance for convergence. Default is 1e-5.

    Returns:
        numpy.ndarray: The transformation matrix between the source and target point clouds.
    """
    target_points1 = sample_points_in_cylinder(target, center1, height1, radius1, samples // 2)
    target_points2 = sample_points_in_cylinder(target, center2, height2, radius2, samples // 2)
    target_points = np.vstack((target_points1, target_points2))

    source_points1 = sample_points_in_cylinder(source, center1, height1, radius1, samples // 2)
    source_points2 = sample_points_in_cylinder(source, center2, height2, radius2, samples // 2)
    source_points = np.vstack((source_points1, source_points2))
    
    # Convert numpy arrays to open3d point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

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
    transformation_matrix[:3, :3] = rotation_matrix.get()
    transformation_matrix[:3, 3] = translation_vector.get()

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

    if user_input == "1":
        # ask user to select folder path using filedialog
        root = Tk()
        root.withdraw()
        input_folder_path = filedialog.askdirectory(initialdir=raw_file_path,
                                                    title="Select folder",)
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
        root = Tk()
        root.withdraw()
        input_file_paths = filedialog.askopenfilenames(
                                            title="Select two STL files",
                                            initialdir=desampled_scans_path,
                                            filetypes=(("stl files", "*.stl"),),
                                            multiple=True)

        if len(input_file_paths) < 2:
            print(("Error: You selected less than 2 STL files."
                   " Please select exactly 2 STL files."))
        elif len(input_file_paths) > 2:
            print(("Error: You selected more than 2 STL files."
                   " Please select exactly 2 STL files."))
        else:
            model1_file_path = input_file_paths[0]
            print(f"model 1 path: {model1_file_path}")
            model1_name = split_path(model1_file_path)[1]
            print(f"model 1 name: {model1_name}")
            model2_file_path = input_file_paths[1]
            print(f"model 2 path: {model2_file_path}")
            model2_name = split_path(model2_file_path)[1]
            print(f"model 2 name: {model2_name}")

            # load and visualized raw models, post-centering
            model1 = load_stl(model1_file_path)
            model2 = load_stl(model2_file_path)
            # plot_meshes([model1, model2], colors=['red', 'blue'], 
            #             title="Raw models")
            
            # pre-rotate model2 and visualize
            ROT = -30.0
            pre_rotate(model2, ROT)
            plot_meshes([model1, model2], colors=['red', 'blue'], 
                        title=f"Pre-rotated model 2 by {ROT} degrees")

            # create boundary cylinder for targeted alignment and visualize
            cylinder_coords = np.array([4, 17, 0])
            height = 12
            radius = 6.5
            NUM_POINTS = 10000
            
            # Sample points within the cylinders
            model1_points1 = sample_points_in_cylinder(model1, cylinder_coords,
                                                       height, radius,
                                                       num_points=NUM_POINTS//2)
            model1_points2 = sample_points_in_cylinder(model1, -cylinder_coords,
                                                       height, radius,
                                                       num_points=NUM_POINTS//2)
            model2_points1 = sample_points_in_cylinder(model2, cylinder_coords,
                                                       height, radius,
                                                       num_points=NUM_POINTS//2)
            model2_points2 = sample_points_in_cylinder(model2, -cylinder_coords,
                                                       height, radius,
                                                       num_points=NUM_POINTS//2)
            model1_points = np.concatenate((model1_points1, model1_points2))
            model2_points = np.concatenate((model2_points1, model2_points2))

            plot_points_and_meshes([model1_points, model2_points],
                                   [model1, model2],
                                    point_colors=['green', 'yellow'],
                                    mesh_colors=['red', 'blue'],
                                    title=(f"AOI cylinders, 1: {cylinder_coords}, "
                                           f"2:{-cylinder_coords}\nheight: {height}, "
                                           f"radius: {radius}"))

            # perform CPD registration and visualize
            tran_matrix = cpd_registration(model2, cylinder_coords, height, radius,
                                           model1, -cylinder_coords, height, radius,
                                           samples=NUM_POINTS)
            model2.apply_transform(tran_matrix)
            plot_meshes([model1, model2], colors=['red', 'blue'],
                        title="Aligned Models")
            
            # combine and save models
            combined_model = trimesh.util.concatenate(model1, model2)
    
            output_file_path = (root_path / "aligned_scans" /
                               f"{model1_name}_{model2_name}_aligned.stl")
            save_stl(combined_model, output_file_path)

    elif user_input == "3":
        # ask user to select file path using filedialog
        root = Tk()
        root.withdraw()
        input_file_path = filedialog.askopenfilename(initialdir=desampled_scans_path,
                                                    title="Select file", 
                                                    filetypes=(("stl files", "*.stl"),))
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
