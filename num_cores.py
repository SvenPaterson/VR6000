import multiprocessing

import trimesh
import numpy as np

print(f"Number of cores: {multiprocessing.cpu_count()}")

print("Attributes of trimesh.remesh:")
print(dir(trimesh.remesh))

sphere_coord = np.array([-9, 13, 0])
print(f"Sphere coordinate: {sphere_coord}")
print(f"invers sphere coordinate: {-sphere_coord}")