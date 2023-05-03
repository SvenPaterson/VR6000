import multiprocessing

import trimesh
import numpy as np

print(f"Number of cores: {multiprocessing.cpu_count()}")

print("Attributes of trimesh.remesh:")
print(dir(trimesh.remesh))

sphere_coord = np.array([-9, 13, 0])
print(f"Sphere coordinate: {sphere_coord}")
print(f"invers sphere coordinate: {-sphere_coord}")

import cupy as cp

# Check CuPy version
print("CuPy version:", cp.__version__)

# Check CUDA version
print("CUDA version:", cp.cuda.runtime.runtimeGetVersion())



# Test basic CuPy functionality
x = cp.array([1, 2, 3])
y = cp.array([4, 5, 6])
z = x + y
print("Result of CuPy array addition:", z)
