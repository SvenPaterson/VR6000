import trimesh, os

from tqdm import tqdm
from pathlib import Path
from tkinter import filedialog, Tk
from joblib import Parallel, delayed

import numpy as np
import trimesh.transformations as tra

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
    target_faces = int(mesh.faces.shape[0] * fraction)

    with tqdm(total=target_faces, desc="Decimating mesh", ncols=80) as pbar:
        while mesh.faces.shape[0] > target_faces:
            mesh = mesh.simplify_quadric_decimation(int(mesh.faces.shape[0] * 0.95))
            pbar.update(int(mesh.faces.shape[0] * 0.05))

    return mesh

def decimate_mesh(mesh, fraction, n_jobs=8):
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
                delayed(simplify_quadric_decimation_chunk)(chunk) for chunk in faces_chunks)
            
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

def icp_registration(model1, model2):
    # Align the models using ICP
    transformation_matrix, _, cost = trimesh.registration.icp(
        model1.vertices, model2.vertices, scale=False  # Set scale=False to avoid scaling
    )

    # Apply the transformation matrix to model2
    model2.apply_transform(transformation_matrix)

    return transformation_matrix, cost


def apply_transformation(model, transformation_matrix):
    model.apply_transform(transformation_matrix)

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

        # Ask user to select a folder containing the STL files using filedialog
        input_folder_path = filedialog.askdirectory(title="Select folder containing the two STL files",
                                                    initialdir=desampled_scans_path)

        # Get all STL files in the folder
        stl_files = [f for f in Path(input_folder_path).glob("*.stl")]

        if len(stl_files) < 2:
            print("Error: The selected folder contains less than 2 STL files. Please select a folder with exactly 2 STL files.")
        elif len(stl_files) > 2:
            print("Error: The selected folder contains more than 2 STL files. Please select a folder with exactly 2 STL files.")
        else:
            model1_file_path = str(stl_files[0])
            print(f"model 1 path: {model1_file_path}")
            model1_name = split_path(model1_file_path)[1]
            print(f"model 1 name: {model1_name}")
            model2_file_path = str(stl_files[1])
            print(f"model 2 path: {model2_file_path}")
            model2_name = split_path(model2_file_path)[1]
            print(f"model 2 name: {model2_name}")

            model1 = load_stl(model1_file_path)
            model2 = load_stl(model2_file_path)

            ROT = -120.0
            pre_rotate(model2, ROT)
            transformation_matrix, _ = icp_registration(model1, model2)
            apply_transformation(model2, transformation_matrix)

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
