#import stl file and plot it
from stl import mesh
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree
from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy.spatial import SphericalVoronoi

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('test-scan.stl')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
# Auto scale to the mesh size
scale = your_mesh.points.flatten('-1')
ax.auto_scale_xyz(scale, scale, scale)
# Show the plot to the screen
plt.show()

