import trimesh
import os

import numpy as np
import pyvista as pv
import trimesh.transformations as tra

from tqdm import tqdm
from pathlib import Path
from tkinter import filedialog, Tk
from joblib import Parallel, delayed

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
    mesh.vertices = mesh.vertices - mesh.centroid
    return mesh

def decimate_mesh_slow(mesh, fraction):
    """
    Decimate a mesh using a slower method by iteratively applying quadric decimation.

    Args:
        mesh (trimesh.Trimesh): The input mesh to be decimated.
        fraction (float): The fraction of faces to keep in the decimated mesh (0 < fraction <= 1).

    Returns:
        trimesh.Trimesh: The decimated mesh.
    """
    target_faces = int(mesh.faces.shape[0] * fraction)

    with tqdm(total=target_faces, desc="Decimating mesh", ncols=80) as pbar:
        while mesh.faces.shape[0] > target_faces:
            mesh = mesh.simplify_quadric_decimation(int(mesh.faces.shape[0] * 0.95))
            pbar.update(int(mesh.faces.shape[0] * 0.05))

    return mesh

def decimate_mesh(mesh, fraction, n_jobs=8):
    """
    Decimate a mesh using a faster method by parallelizing quadric decimation.

    Args:
        mesh (trimesh.Trimesh): The input mesh to be decimated.
        fraction (float): The fraction of faces to keep in the decimated mesh (0 < fraction <= 1).
        n_jobs (int, optional): The number of parallel jobs to run. Default is 8.

    Returns:
        trimesh.Trimesh: The decimated mesh.
    """
    target_faces = int(mesh.faces.shape[0] * fraction)

    with tqdm(total=target_faces, desc="Decimating mesh", ncols=80) as pbar:
        while mesh.faces.shape[0] > target_faces:
            # Split the mesh faces into n_jobs chunks for parallel processing
            faces_chunks = np.array_split(mesh.faces, n_jobs)
            
            # Define the function to be parallelized
            def simplify_quadric_decimation_chunk(faces_chunk):
                mesh_copy = mesh.copy()
                mesh_copy.faces = faces_chunk
                return mesh_copy.simplify_quadric_decimation(int(faces_chunk.shape[0] * 0.95))
            
            # Apply the function to each faces chunk in parallel
            new_meshes = Parallel(n_jobs=n_jobs)(
                delayed(simplify_quadric_decimation_chunk)(chunk) 
                                                    for chunk in faces_chunks)
            
            # Combine the meshes
            mesh = trimesh.util.concatenate(new_meshes)
            pbar.update(int(mesh.faces.shape[0] - target_faces))

    return mesh

def pre_rotate(mesh, angle):
    import numpy as np
    """
    Rotate a mesh around the z-axis by a given angle (in degrees).
    """
    T = tra.rotation_matrix(np.radians(angle), [0, 0, 1])
    mesh.apply_transform(T)

def sample_points_in_sphere(mesh, center, radius, num_points):
    """
    Sample points from the surface of the mesh within a sphere.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        center (list or numpy array): The center of the sphere [x, y, z].
        radius (float): The radius of the sphere.
        num_points (int): The number of points to sample.

    Returns:
        A numpy array of shape (num_points, 3) containing the sampled points.
    """
    # Get all points on the surface of the mesh
    all_points, _ = trimesh.sample.sample_surface(mesh, num_points * 10)

    # Select points that are within the sphere
    distances = np.linalg.norm(all_points - center, axis=1)
    points_in_sphere = all_points[distances <= radius]

    # Randomly select a subset of points if there are more points than needed
    if points_in_sphere.shape[0] > num_points:
        indices = np.random.choice(points_in_sphere.shape[0], 
                                   num_points, replace=False)
        points_in_sphere = points_in_sphere[indices]

    return points_in_sphere

def icp_registration(model1, model2, center, radius, 
                     sampled=True, samples=10000):
    # Align the models using ICP
    if sampled:
        model1_points = sample_points_in_sphere(model1, center, radius, samples)
        model2_points = sample_points_in_sphere(model2, center, radius, samples)

        transformation_matrix, _, cost = trimesh.registration.icp(
            model1_points, model2_points, scale=False
        )
    else:
        transformation_matrix, _, cost = trimesh.registration.icp(
            model1.vertices, model2.vertices, scale=False
        )

    # Apply the transformation matrix to model2
    model2.apply_transform(transformation_matrix)

    return transformation_matrix, cost

def icp_registration_with_two_spheres(model1, model2, 
                                      center1, radius1, 
                                      center2, radius2, 
                                      sampled=True, samples=10000):
    # Align the models using ICP
    if sampled:
        model1_points1 = sample_points_in_sphere(model1, center1, 
                                                 radius1, samples // 2)
        model1_points2 = sample_points_in_sphere(model1, center2, 
                                                 radius2, samples // 2)
        model1_points = np.vstack((model1_points1, model1_points2))
        
        model2_points1 = sample_points_in_sphere(model2, center1, 
                                                 radius1, samples // 2)
        model2_points2 = sample_points_in_sphere(model2, center2, 
                                                 radius2, samples // 2)
        model2_points = np.vstack((model2_points1, model2_points2))

        transformation_matrix, _, cost = trimesh.registration.icp(
            model1_points, model2_points, scale=False
        )
    else:
        transformation_matrix, _, cost = trimesh.registration.icp(
            model1.vertices, model2.vertices, scale=False
        )

    # Apply the transformation matrix to model2
    model2.apply_transform(transformation_matrix)

    return transformation_matrix, cost

def plot_meshes(meshes, colors=None, opacities=None, title=None):
    # Create a Pyvista plotter
    plotter = pv.Plotter()

    # Set default colors and opacities if not provided
    if colors is None:
        colors = ['red', 'blue'] * len(meshes)
    if opacities is None:
        opacities = [0.5] * len(meshes)

    # Convert Trimesh objects to Pyvista objects and add them to the plotter
    for mesh, color, opacity in zip(meshes, colors, opacities):
        pv_mesh = pv.wrap(mesh)
        plotter.add_mesh(pv_mesh, color=color, opacity=opacity)

    # Set the title
    if title is not None:
        plotter.add_text(title, font_size=20, name='title')

    # Show the plot
    plotter.show()

def save_stl(mesh, file_path):
    mesh.export(file_path)

if __name__ == "__main__":
    root_path = Path.cwd()
    raw_file_path = root_path / "raw_scans"
    desampled_scans_path = root_path / "desampled_scans"
    
    # ask user if they are desampling a model or performing a best fit of two models
    print("Desampling a model or performing a best fit of two models?")
    print("1. Desampling a model")
    print("2. Performing a best fit of two models")
    print("3. Pre-rotate a model")
    user_input = input("Enter 1, 2, or 3: ")

    if user_input == "1":
        root = Tk()
        root.withdraw()  # Hide the root window

        # ask user to select folder path using filedialog
        input_folder_path = filedialog.askdirectory(initialdir=raw_file_path,
                                                    title="Select folder",)

        output_folder_path = desampled_scans_path

        # Set the fraction of faces to keep (e.g., 0.5 to keep half of the faces)
        fraction = 0.1

        # Get all STL files in the folder
        stl_files = [f for f in Path(input_folder_path).glob("*.stl")]

        for stl_file in stl_files:
            # get file name from stl_file
            file_name = stl_file.name
            output_file_path = output_folder_path / f'desamp_{file_name}'

            mesh = load_stl(stl_file)

            decimated_mesh = decimate_mesh(mesh, fraction)

            save_stl(decimated_mesh, output_file_path)

    elif user_input == "2":
        root = Tk()
        root.withdraw()  # Hide the root window

       # Ask user to select two STL files using filedialog
        input_file_paths = filedialog.askopenfilenames(
                                            title="Select two STL files",
                                            initialdir=desampled_scans_path,
                                            filetypes=(("stl files", "*.stl"),),
                                            multiple=True)

        if len(input_file_paths) < 2:
            print("Error: You selected less than 2 STL files. Please select exactly 2 STL files.")
        elif len(input_file_paths) > 2:
            print("Error: You selected more than 2 STL files. Please select exactly 2 STL files.")
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
            plot_meshes([model1, model2], colors=['red', 'blue'], 
                        opacities=[0.5, 0.5],
                        title="Raw models, post-centering")

            # pre-rotate model2
            ROT = -120.0
            pre_rotate(model2, ROT)
            plot_meshes([model1, model2], colors=['red', 'blue'], 
                        title=f"Pre-rotated model 2 by {ROT} degrees")

            # create sphere to select points of interest and point all three meshes
            sphere_coord = np.array([-7, 11, 0])
            radius = 4
            sphere1 = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere1.vertices += sphere_coord
            sphere2 = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere2.vertices -= sphere_coord
            plot_meshes([model1, model2, sphere1, sphere2], 
                        colors=['red', 'blue', 'green', 'yellow'], 
                        title=f"AOI spheres, 1: {sphere_coord}, 2:{-sphere_coord}\nradius: {radius}")

            # perform ICP registration
            """ transformation_matrix, _ = icp_registration(model1, model2, center,
                                                        radius, sampled=False) """
            transformation_matrix, _ = icp_registration_with_two_spheres(model1, model2,
                                                                         sphere_coord, radius,
                                                                         -sphere_coord, radius,
                                                                         sampled=False)
            plot_meshes([model1, model2], colors=['red', 'blue'],
                        title="Aligned Models")
            combined_model = trimesh.util.concatenate(model1,
                                                      model2)

            output_file_path = root_path / "aligned_scans" / f"{model1_name}_{model2_name}_aligned.stl"
            save_stl(combined_model, output_file_path)

    elif user_input == "3":

        root = Tk()
        root.withdraw()  # Hide the root window

        # ask user to select file path using filedialog
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
