

program_version = 'Smart Seal Scanner Software V1.01 2/3/2023'

import numpy as np
from scipy import interpolate
import open3d as o3d
import os
import tkinter 
from tkinter import filedialog
import pyvista as pv
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                                Load SDF Surface
# -----------------------------------------------------------------------------

n_x = 10000
n_y = 1280

# bound_left  = 580
# bound_right = 740

bound_left  = 470#420 # 350
bound_right = 650#570 # 520

height_bound = 0.8 # 0.72

left_eps = -0.007
right_eps = 0.008

hasrlimit = True 
iqr_scale = 1.5
lower_limit_y = 0.0
upper_limit_y = 3
lower_limit_alpha = 0
upper_limit_alpha = 25
save = True
flip_surface = False
upper_limit = 1.0
plot3d = False

def sdf2array(file_path):
    with open(file_path, mode='rb') as file: # b is important -> binary
        file.seek(81) # data before 81 bytes are header, irrelevant 
        data = np.fromfile(file, dtype = np.double) 
        return data

def save_array_at_dir(dir_path, filename, arr, mode='w'):
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, mode):
        np.save(file_path,arr)

def chooseFile():
    
    # Open a browse window and choose a file
    root = tkinter.Tk()
    root.withdraw()     # used to hide tkinter window
    
    currdir = os.getcwd()
    fileName = filedialog.askopenfilename(\
        parent = root, 
        initialdir = currdir, 
        title = 'Please select a surface data file of the type *.sdf',
        filetypes = (('Surface data files' ,'*.sdf'),
                     ('all files','*.*')))
    if len(fileName) > 0:
        print ("You chose: %s" % fileName)
    return fileName

# Choose file
absolutePathToFile = chooseFile()
dirName, fileName = os.path.split(absolutePathToFile)
fileName = fileName.replace('.sdf','')

y = np.linspace(0, 20.46, n_y)
x = np.arange(0, 2*3.14, 2*3.14/n_x)
x = np.round(x,decimals=4)
Y, X = np.meshgrid(y, x)
Z = sdf2array(absolutePathToFile)

if len(Z) > n_x*n_y:
    Z = Z[:-1]
# Z[np.isnan(Z)] = 1e5

Z = np.reshape(Z,[10000,n_y])

Z = Z[:,bound_left:bound_right]
X = X[:,bound_left:bound_right]
Y = Y[:,bound_left:bound_right]

n_y_2 = np.size(X,axis=1)
n_x_2 = np.size(X,axis=0)

# Z = Z2
# Z2 = np.reshape(Z,[10000,1280])

new_scans_folder = '\\Smart_Seal_Scan_Evaluation'
mydir = (dirName + new_scans_folder)
check_folder = os.path.isdir(mydir)

'''Check if directory exists, if not, create it'''

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(mydir)
    print("created folder: ", mydir)
else:
    print(mydir, "folder already exists.")
'''
import matplotlib as mpl
fig, ax = mpl.pyplot.subplots(figsize=(8, 5))
mpl.pyplot.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
ax.plot(Y[500,:],Z[500,:],'-',label='static pressure',color='red')
# ax.set_ylim([1e-2,1e6])
# ax.set_xlim([2.5,3.5])
ax.set(xlabel='x [mm]', ylabel='y [mm]')
# ax.legend(loc = 'upper center')
# fig.savefig("p_and_h_instroke_logarithmic.png",dpi=300)
mpl.pyplot.show()
'''

# Express the mesh with 3d coordinates.
data3d = np.zeros((np.size(X), 3))
data3d[:, 0] = np.reshape(X, -1)
data3d[:, 1] = np.reshape(Y, -1)
data3d[:, 2] = np.reshape(Z, -1)

# data3d = data3d[::10,:]


# -----------------------------------------------------------------------------
#                                Outlier Removal
# -----------------------------------------------------------------------------

# Print step
print("statistical oulier removal")

# Object of the PointCloud class. A point cloud consists of point coordinates, 
# and optionally point colors and point normals.
pcd = o3d.geometry.PointCloud()

# Convert float64 numpy array of shape (n, 3) to Open3D format.
# Line below gives points which ARE NOT nans
pcd.points = o3d.utility.Vector3dVector(data3d[~np.isnan(data3d).any(axis=1)])

# Object of the PointCloud class. A point cloud consists of point coordinates, 
# and optionally point colors and point normals.
nans = o3d.geometry.PointCloud()

# Convert float64 numpy array of shape (n, 3) to Open3D format.
# Line below gives points which ARE nans
nans.points = o3d.utility.Vector3dVector(data3d[np.isnan(data3d).any(axis=1)])

# Remove points that are further away from their neighbors compared to the 
# average for the point cloud. It takes two input parameters:
#
# 1. nb_neighbors:
#
#    allows to specify how many neighbors are taken into account 
#    in order to calculate the average distance for a given point.
#
# 2. std_ratio:
#
#    allows to set the threshold level based on the standard 
#    deviation of the average distances across the point cloud. The lower this 
#    number the more aggressive the filter will be.
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=3.0)

# Select inliners and outliers 
inlier_cloud = pcd.select_by_index(ind)
outlier_cloud = pcd.select_by_index(ind, invert=True)

# Get coordinates of inliers and outliers and 
# transfer from open3D format to float64 
# Together, the outliers, inliners and nans, make 100% of the points
xyz_inliner = np.asarray(inlier_cloud.points)
xyz_outlier = np.asarray(outlier_cloud.points)
xyz_nans = np.asarray(nans.points)

# Percentage of nans
percentage_nans = len(xyz_nans[:,0])/(n_x*n_y)*100
print(f'Percentage nans: {round(percentage_nans,2)}%')

# Percentage of outliers
percentage_outliers = len(xyz_outlier[:,0])/(n_x*n_y)*100
print(f'Percentage outliers: {round(percentage_outliers)}%')

# Percentage corrupted data
print(f'Percentage N/A data: {round(percentage_nans + percentage_outliers)}%')

# -----------------------------------------------------------------------------
#                            2-Step 2D Interpolation
# -----------------------------------------------------------------------------

# Combine coordinates of outliers + nans into a single array 
xyz_to_interpolate = np.concatenate((xyz_outlier,xyz_nans))
print('1st interpolation with linear method')

# Use inliers to interpolate the array with the outliers + nans
# Note that [:,0:2] corresponds to x and y and [:,2] corresponds to z values
z_interpolated = interpolate.griddata(xyz_inliner[:,0:2],
                                      xyz_inliner[:,2],
                                      xyz_to_interpolate[:,0:2],
                                      method='linear')
z_interpolated = np.asarray(z_interpolated)
xyz_interpolated = []

# How enumerate() is used
# values = ["a", "b", "c"]
# for count, value in enumerate(values, start=1):
# ...     print(count, value)
# ...
# 1 a
# 2 b
# 3 c

for i,arr in enumerate(xyz_to_interpolate):
    arr[2] = z_interpolated[i]
    xyz_interpolated.append(arr)
xyz_interpolated = np.asarray(xyz_interpolated)

xyz_all = np.concatenate((xyz_inliner,xyz_interpolated))

xyz_to_interpolate_second_step = xyz_all[np.isnan(xyz_all).any(axis=1)]
xyz_all_without_nan = xyz_all[~np.isnan(xyz_all).any(axis=1)]
print('2nd interpolation with nearest method')
z_interpolated_second_step = interpolate.griddata(xyz_all_without_nan[:,0:2],xyz_all_without_nan[:,2],xyz_to_interpolate_second_step[:,0:2],method='nearest')
z_interpolated_second_step = np.asarray(z_interpolated_second_step)
xyz_interpolated_second_step = []
for i,arr in enumerate(xyz_to_interpolate_second_step):
    arr[2] = z_interpolated_second_step[i]
    xyz_interpolated_second_step.append(arr)
xyz_interpolated_second_step = np.asarray(xyz_interpolated_second_step)

if xyz_interpolated_second_step.ndim == 2:
    xyz_all = np.concatenate((xyz_all_without_nan,xyz_interpolated_second_step))
elif xyz_interpolated_second_step.ndim == 1:
    xyz_all = xyz_all_without_nan

helper = np.isnan(np.min(xyz_all))
print('Kammer still contains nans? {}'.format(helper))

ind1 = np.argsort(xyz_all[:,0])
xyz_all_sorted = xyz_all[ind1]

global final_array
final_array = []
for i in range(0,n_x):
    y = xyz_all_sorted[n_y_2*i:n_y_2*(i+1),2]
    x = xyz_all_sorted[n_y_2*i:n_y_2*(i+1),1] 
    ind2 = np.argsort(x)
    y = y[ind2]
    final_array.append(y)

final_array = np.asarray(final_array, dtype = float)


data3d[:, 0] = np.reshape(X, -1)
data3d[:, 1] = np.reshape(Y, -1)
data3d[:, 2] = np.reshape(final_array, -1)

xyz_nans[:,2] = np.min(data3d[:,2])

if plot3d:
    pl = pv.Plotter(notebook = False)
    pl.add_points(data3d, point_size= 0.5, color='#D4AF37', opacity=0.1) #'#C6E2FF'
    pl.add_points(xyz_outlier, point_size= 0.5, color='#FE7E63', opacity=1.0)
    pl.show()


# -----------------------------------------------------------------------------
#                                 Wear Detection
# -----------------------------------------------------------------------------

# Packages and Functions
import os
import matplotlib.pyplot as plt
import numpy as np
# from scipy import interpolate
# from scipy.signal import savgol_filter
# import typer

def ar_zero_y(arr):
    ar_z = np.zeros(np.shape(arr))
    for i in np.arange(0,np.shape(arr)[0],1):
        ar_z[i] = arr[i] - np.nanmin(arr[i])
    return ar_z

def get_right_point_window_3d(diff_arr,y_arr,window_size = 20,delta_y = 0.002):
    slope = 0
    diff_arr = np.asarray(diff_arr)
    a = min(diff_arr.shape[0],10)
    for i in range(0,a):
        slope += diff_arr[-1-i].mean()
    slope = slope/a
    for x in range(10,y_arr.shape[0]):
        x_b = y_arr.shape[0]-(x+1)
        window_slope = y_arr[(x_b-window_size):x_b].mean()
        if window_slope < slope-delta_y:
            return round(x_b-(window_size/3)-0.5)
    return -1

def get_left_point(y_arr, threshold_x = 5, threshold_y = 0):
    count = 0
    x_ret = 0
    for x, val in enumerate(y_arr):
        if val > threshold_y:
            count += 1
            if count ==1:
                x_ret = x
        else:
            count = 0
        if count == threshold_x:
            return x_ret
    return -1

# Data to a new array
if flip_surface == True:
    arr = np.fliplr(final_array)
else:
    arr = final_array



# Set minimum to zero
arr = ar_zero_y(arr)

# Profile length coordinates for a total profile length 
x = np.linspace(0,n_y_2/n_y*20.46,n_y_2).T

print(arr.shape)

# Detect Wear Edgepoints and Caculate Distances

distances = []
orientation = []
left_points = []
right_points = []
diff_arr = []
for i in range(0,10000):
    y = arr[i]
        # Create an empty list
    filter_arr = []

    for element in y:
    # if the element is higher than 0.38, set the value to True, otherwise False:
        if element > height_bound:
            filter_arr.append(False)
        else:
            filter_arr.append(True)
    
    kernel_size = 6
    kernel = np.ones(kernel_size) / kernel_size
    y_filter = np.convolve(y, kernel, mode='same')
    y_show = y[filter_arr]
    
    y_filter = y_filter[filter_arr]
    # y_filter = y[filter_arr]
    x_filter = x[filter_arr]

    y_diff = np.gradient(y_filter,1)
    y_diff = np.convolve(y_diff, kernel, mode='same')
    diff_arr.append(y_diff[-15:])

    left_point = get_left_point(y_diff, 3, threshold_y=left_eps)
    right_point = get_right_point_window_3d(diff_arr,y_diff,window_size=10,delta_y=right_eps)

    if right_point == -1:
        print(f'Failure while trying to get right point in plot {i}')
    else:
        if left_point == -1:
            print(f'Failure while trying to get left point in plot {i}')
            print(right_point)
        else:
            left_point_coordinates = np.array((x_filter[left_point], y_filter[left_point]))
            left_point_coordinates_3d = np.array((0.000628*i,x_filter[left_point], y_filter[left_point]))
            left_points.append(left_point_coordinates_3d)
            right_point_coordinates = np.array((x_filter[right_point], y_filter[right_point]))
            right_point_coordinates_3d = np.array((0.000628*i,x_filter[right_point], y_filter[right_point]))
            right_points.append(right_point_coordinates_3d)
            eucl_dst = np.linalg.norm(left_point_coordinates-right_point_coordinates)
            distances.append(eucl_dst)
            orientation.append(abs(np.arctan((right_point_coordinates[1]-left_point_coordinates[1])/(right_point_coordinates[0]-left_point_coordinates[0]))*180/np.pi))
            if np.mod(i,1250) == 0:
                # Plot profile for one cross-section as example
                fig = plt.figure(figsize =(6.4, 4.5))
                ax = fig.gca()
                              
                ax.set_xlabel("x in mm")
                ax.set_ylabel("y in mm", color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax2 = ax.twinx()
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylabel("dy/dx", color='red')
                
                ax.set_facecolor('#EBEBEB')
                # Remove border around plot.
                [ax.spines[side].set_visible(False) for side in ax.spines]
                [ax2.spines[side].set_visible(False) for side in ax2.spines]
                # Style the grid.
                # ax.grid(which='major', color='white', linewidth=1.2)
                # ax2.grid(which='major', color='white', linewidth=1.2)
                ax.tick_params(axis=u'both', which=u'both',length=0)
                ax2.tick_params(axis=u'both', which=u'both',length=0)
                
                ax2.plot(x_filter,y_diff,'r',lw=1.5,alpha=0.5,label='gradient(y)')
                ax2.axhline(y = 0.008, color = 'r', lw=0.5, linestyle = '--', alpha=0.3)
                ax2.axhline(y = -0.0035, color = 'r', lw=0.5, linestyle = '--', alpha=0.3)
                ax2.axvline(x = left_point_coordinates[0], color = 'b', lw=0.5, linestyle = '--', alpha=0.3)
                ax2.axvline(x = right_point_coordinates[0], color = 'b', lw=0.5, linestyle = '--', alpha=0.3)
                
                ax.plot(x_filter,y_filter,color='blue',lw=3,alpha=0.5,label='y')
                ax.scatter(left_point_coordinates[0], left_point_coordinates[1], marker='o', color='black', label = 'wbw end point')#, marker='s', size=5)#, markerfacecolor='w', markeredgewidth=1.5, markeredgecolor='red')
                ax.scatter(right_point_coordinates[0], right_point_coordinates[1], marker='o', color='black')#, marker='s', size=5)#, markerfacecolor='w', markeredgewidth=1.5, markeredgecolor='red')
                ax.set_title(f'Cross Section {int(i)} of 10000', fontsize = 10)
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc= 'upper center')
                
                if save:
                    plt.savefig(dirName + new_scans_folder 
                                + '//' + fileName 
                                + '_Cross_Section_' + str(i)
                                + '.png', 
                                bbox_inches='tight',dpi=300)
                plt.show()
                

distances = np.asarray(distances)
orientation = np.asarray(orientation)
left_points = np.asarray(left_points)
right_points = np.asarray(right_points)


data3d[:, 2] = np.reshape(arr, -1)
data3d[:, 1] = data3d[:, 1] - data3d[0, 1]

if plot3d:
    pl = pv.Plotter(notebook = False)
    pl.add_points(np.column_stack((data3d[:,:2],np.max(data3d[:,2])-data3d[:,2])), point_size= 1, color='#D4AF37', opacity=0.1)
    pl.add_points(np.column_stack((left_points[:,:2],np.max(data3d[:,2])-left_points[:,2])), point_size= 2, color='#9400D3', opacity=1)
    pl.add_points(np.column_stack((right_points[:,:2],np.max(data3d[:,2])-right_points[:,2])), point_size= 2, color='#9400D3', opacity=1)
    pl.show_grid(xlabel="circumf. direction [rad]", ylabel="ax. direction [mm]", zlabel="height [mm]")
    pl.camera.position = (15, -1.5, 11)
    pl.show()

# point_cloud = pv.PolyData(np.column_stack((data3d[:,:2],np.max(data3d[:,2])-data3d[:,2])))
# left_points_cloud = pv.PolyData(np.column_stack((left_points[:,:2],np.max(data3d[:,2])-left_points[:,2])))
# right_points_cloud = pv.PolyData(np.column_stack((right_points[:,:2],np.max(data3d[:,2])-right_points[:,2])))

# point_cloud.plot(point_size= 1, color='#D4AF37', opacity=0.1)
# left_points_cloud.plot(point_size= 2, color='#9400D3', opacity=1)
# right_points_cloud.plot(point_size= 2, color='#9400D3', opacity=1)

# pl = pv.Plotter(notebook = False)
# pl.add_points(point_cloud, point_size= 1, color='#D4AF37', opacity=0.1)
# pl.add_points(left_points_cloud, point_size= 2, color='#9400D3', opacity=1)
# pl.show()

import matplotlib as mpl

x = np.reshape(data3d[:,0],[10000,bound_right-bound_left])
y = np.reshape(data3d[:,1],[10000,bound_right-bound_left])
h_c = np.reshape(data3d[:,2],[10000,bound_right-bound_left])

fig = mpl.pyplot.figure(figsize=(6.4,6.4))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x*180/np.pi,y,h_c,
                       linewidth=0.5, antialiased=True,vmin=0,vmax=0.5, color = 'lightblue', alpha = 0.6)
ax.scatter(left_points[:,0]*180/np.pi,left_points[:,1],left_points[:,2], s = 0.3, c='green')
ax.scatter(right_points[:,0]*180/np.pi,right_points[:,1],right_points[:,2], s = 0.3, c='green')
ax.zaxis.set_major_locator(mpl.ticker.LinearLocator(5))
ax.zaxis.set_major_formatter('{x:.01f}')
ax.view_init(60, 10)

[ax.spines[side].set_visible(False) for side in ax.spines]

ax.set_xlabel("circumf. [°]")
ax.set_ylabel("axial [mm]")
mpl.pyplot.gca().invert_zaxis()
ax.set_zlabel("radial [mm]")

ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,1)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,1)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,1)
if save:
    plt.savefig(dirName + new_scans_folder 
                + '/' + fileName 
                + '_detected_wear_band_' + '.png', 
                bbox_inches='tight',dpi=300)
mpl.pyplot.show()



# .save('my_mesh.ply')

# -----------------------------------------------------------------------------
#            Filter and Plot the Wear Band Width and Orientation
# -----------------------------------------------------------------------------

import pandas as pd
import bottleneck as bn

'''
# Distance and Orientation Scatter Plots

fig = plt.figure()
ax = fig.gca()
ax.set_ylabel("distance in mm")
ax.set_xlabel("radial degrees")
ax.set_ylim([0,upper_limit_y])
x_plot_1 = np.linspace(0,360,len(distances))
ax.scatter(x_plot_1, distances)
fig.set_figwidth(10)
ax.set_title('Scatter Plot of the Wear Band Width')
# if save:
#     plt.savefig('../results/figures/{hours}h/Kammer_{kammer}_distance_scatter.svg', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
ax.set_ylabel("orientation in °")
ax.set_xlabel("radial degrees")
ax.set_ylim([-10,10])
x_plot_1 = np.linspace(0,360,len(distances))
ax.scatter(x_plot_1, orientation)
fig.set_figwidth(10)
ax.set_title('Scatter Plot of the Wear Orientation')

# Distances after Outlier Removal and Smoothing
'''

def rollavg_bottlneck(a,n):
    return bn.move_mean(a, window=n,min_count = None)

distances_df = pd.DataFrame(distances, columns=["distance"])

q1 = distances_df.distance.quantile(.25)
median = distances_df.distance.median()
q3 = distances_df.distance.quantile(.75)
iq_range = q3 - q1

def is_outlier(row, scale = 1.5):
    if row > (median + (scale* iq_range)) or row < (median - (scale* iq_range)):
        return True
    else:
        return False

#apply the function to the original df:
distances_df.loc[:, 'outlier'] = distances_df.distance.apply(is_outlier, scale = iqr_scale)
outlier_mask_distances = distances_df.outlier
num_outlier = (outlier_mask_distances).sum()
print(f'find {num_outlier} outliers')
#filter to only non-outliers:
distances_filtered = distances_df[~outlier_mask_distances].distance.to_numpy()

rolling_win_size = 200
circular_distances_filtered = np.concatenate([distances_filtered, distances_filtered[:rolling_win_size]])
lagged_distances_smoothed = rollavg_bottlneck(circular_distances_filtered, rolling_win_size)[rolling_win_size:]
distances_smoothed = np.concatenate([lagged_distances_smoothed[-rolling_win_size // 2:], lagged_distances_smoothed[:-rolling_win_size // 2]])





theta_all = np.linspace(0, 360, len(distances_filtered))
fig = plt.figure(figsize =(6, 4.5))
ax = fig.gca()
ax.set_ylabel('wear band width [mm]')
ax.set_xlabel('angle along circumference [°]')
x_plot = np.linspace(0,360,len(distances_smoothed))
ax.set_ylim([lower_limit_y,upper_limit_y])
plt.xticks(np.arange(0, 361, step=60))

ax.set_facecolor('#EBEBEB')
# Remove border around plot.
[ax.spines[side].set_visible(False) for side in ax.spines]
# Style the grid.
ax.grid(which='major', color='white', linewidth=1.2)
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.scatter(theta_all, distances_filtered, marker='o', color='yellowgreen', alpha = 0.3, label = 'detected wear band width')
ax.plot(theta_all, distances_smoothed, color='blue',lw=3,alpha=0.5,label='smoothed wear band width')
ax.set_title('Wear Band Width', fontsize = 10)
ax.grid('True')
ax.set_xlim([-20,380])

ax.legend(loc = 'lower right')


if save:
    plt.savefig(dirName + new_scans_folder 
                + '/' + fileName 
                + '_wbw_' + '.png', 
                bbox_inches='tight',dpi=300)
plt.show()


# Orientation after Outlier Removal and Smoothing

orientation_df = pd.DataFrame(orientation, columns=["orientations"])

q1 = orientation_df.orientations.quantile(.25)
median = orientation_df.orientations.median()
q3 = orientation_df.orientations.quantile(.75)
iq_range = q3 - q1

#apply the function to the original df:
orientation_df.loc[:, 'outlier'] = orientation_df.orientations.apply(is_outlier, scale = iqr_scale)
outlier_mask_orientation = orientation_df.outlier
num_outlier = (outlier_mask_orientation).sum()
print(f'find {num_outlier} outliers')
#filter to only non-outliers:
orientation_filtered = orientation_df[~outlier_mask_orientation].orientations.to_numpy()

rolling_win_size = 500
circular_orientation_filtered = np.concatenate([orientation_filtered, orientation_filtered[:rolling_win_size]])
lagged_orientation_smoothed = rollavg_bottlneck(circular_orientation_filtered, rolling_win_size)[rolling_win_size:]
orientation_smoothed = np.concatenate([lagged_orientation_smoothed[-rolling_win_size // 2:], lagged_orientation_smoothed[:-rolling_win_size // 2]])




theta_all = np.linspace(0, 360, len(orientation_filtered))
fig = plt.figure(figsize =(6.4, 4.5))
ax = fig.gca()
ax.set_ylabel('absolute wear band orientation [°]')
ax.set_xlabel('angle along circumference [°]')
x_plot = np.linspace(0,360,len(orientation_smoothed))
ax.set_ylim([lower_limit_alpha,upper_limit_alpha])
plt.xticks(np.arange(0, 361, step=60))

ax.set_facecolor('#EBEBEB')
# Remove border around plot.
[ax.spines[side].set_visible(False) for side in ax.spines]
# Style the grid.
ax.grid(which='major', color='white', linewidth=1.2)
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.scatter(theta_all, orientation_filtered, marker='o', color='yellowgreen', alpha = 0.3, label = 'detected wear band width')
ax.plot(theta_all, orientation_smoothed, color='blue',lw=3,alpha=0.5,label='smoothed wear band width')
ax.set_title('Wear Orientation', fontsize = 10)
ax.grid('True')
ax.set_xlim([-20,380])

ax.legend(loc = 'lower right')


if save:
    plt.savefig(dirName + new_scans_folder 
                + '/' + fileName 
                + '_orientation_' + '.png', 
                bbox_inches='tight',dpi=300)
plt.show()


def spiderplot_(input_array, 
                input_array_smoothed, 
                lower_limit, 
                upper_limit, 
                outlier_mask, 
                identifier,
                fileName,
                percentage_na_data,
                program_version,
                save,
                spider_plot_path,
                new_scans_folder):
    
    # Distance Spider Plot
    
    if identifier == 'wbw':
        unit = 'mm'
        title = 'Wear Band Width'
    else:
        unit = '°'
        title = 'Wear Band Orientation'
    
    num_slices = len(input_array)
    theta_all = np.linspace(0, 2 * np.pi, num_slices)
    input_array_smoothed = input_array_smoothed[np.logical_not(np.isnan(input_array_smoothed))]
    theta = theta_all[~outlier_mask]
        
    minidx = np.nanargmin(input_array_smoothed)
    maxidx = np.nanargmax(input_array_smoothed)
    
    minr, mintheta = input_array_smoothed[minidx], theta[minidx]
    maxr, maxtheta = input_array_smoothed[maxidx], theta[maxidx]
    
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize =(6.4, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Give plot a gray background like ggplot.
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    # ax.grid(which='minor', color='white', linewidth=0.6)
    # Show the minor ticks and grid.
    # ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    # ax.tick_params(which='minor', bottom=False, left=False)

    
    line, = ax.plot(theta, input_array_smoothed, color='#2986cc', lw=2, 
                    label = title + ' [' + unit + ']')
    lmin, = ax.plot(theta_all, [minr]*num_slices, color='#f44336', lw=1,
                    label = 'max./min.') 
    lmax, = ax.plot(theta_all, [maxr]*num_slices, color='#f44336', lw=1) 
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(10)
    
    ax.vlines(mintheta, 0, minr, color='red', lw=1) 
    ax.vlines(maxtheta, 0, maxr, color='red', lw=1)
    
    ax.set_ylim([lower_limit,upper_limit]) 
    
    ax.plot([mintheta], [minr], 'o', markerfacecolor="None", markeredgecolor='#f44336', markersize=4, markeredgewidth=2)
    ax.plot([maxtheta], [maxr], 'o', markerfacecolor="None", markeredgecolor='#f44336', markersize=4, markeredgewidth=2)
    

    ax.text(-0.05, -0.07, 'n/a pts.:   ' + f' {round(percentage_na_data,2)}% \n' + 
            'mean:      ' + f' {round(np.mean(input_array_smoothed),2)}' + unit 
            + '\n' + 'median:   ' + f' {round(np.median(input_array_smoothed),2)}' + unit 
            + '\n' + 'min.:        ' + f' {round(np.min(input_array_smoothed),2)}' + unit 
            + '\n' + 'max.:       ' + f' {round(np.max(input_array_smoothed),2)}' + unit 
            + '\n\n      ' + program_version,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
       
    ax.legend(loc = 'lower right', bbox_to_anchor=(1.05, -0.25))
    
    ax.annotate(f'min: {round(minr,2)}'+unit,
                xy=(mintheta, minr/2),  # theta, radius
                xycoords='data',
                xytext=(-10, -10), 
                textcoords='offset points',
                va="center", ha="center",
                bbox=dict(boxstyle="round", edgecolor='grey', lw=1, fc='w', alpha=1 ),
                )
    ax.annotate(f'max: {round(maxr,2)}'+unit,
                xy=(maxtheta, maxr/2),  # theta, radius
                xycoords='data',
                xytext=(30, 10), 
                textcoords='offset points',
                va="center", ha="center",
                bbox=dict(boxstyle="round", edgecolor='grey',lw=1, fc='w', alpha=1 ),
                )
    ax.set_title('Spider Plot of the '+ title + '\n Seal: ' + fileName + '\n', fontsize = 11, fontweight='bold')
    plt.show()
    
    if save:
        plt.savefig(spider_plot_path + new_scans_folder 
                    + '/' + fileName 
                    + '_' + identifier 
                    + '.png', 
                    bbox_inches='tight',dpi=300)



  
identifier = 'wbw'
spiderplot_(distances, 
            distances_smoothed, 
            lower_limit_y, 
            upper_limit_y, 
            outlier_mask_distances, 
            identifier,
            fileName,
            percentage_nans + percentage_outliers,
            program_version,
            save,
            dirName,
            new_scans_folder)

identifier = 'orientation'
spiderplot_(orientation, 
            orientation_smoothed, 
            lower_limit_alpha, 
            upper_limit_alpha, 
            outlier_mask_orientation, 
            identifier,
            fileName,
            percentage_nans + percentage_outliers,
            program_version,
            save,
            dirName,
            new_scans_folder)

# # Distance Box-Plot

# # Creating pl
# # show plot
# fig = plt.figure(figsize =(6, 8))
# ax = fig.add_subplot(111)

# plt.ylabel('distance in mm', fontsize=15)

# plt.boxplot(distances_smoothed[199:])
# print(distances_smoothed.shape)
# ax.set_title('Box Plot of the Wear Band Width')
# # if save:
# #     plt.savefig(f'../results/figures/{hours}h/Kammer_{kammer}_smooth_distance_boxplot.svg', bbox_inches='tight')
# plt.show()





