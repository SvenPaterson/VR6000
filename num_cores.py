import multiprocessing

import trimesh

print(f"Number of cores: {multiprocessing.cpu_count()}")

print("Attributes of trimesh.remesh:")
print(dir(trimesh.remesh))
