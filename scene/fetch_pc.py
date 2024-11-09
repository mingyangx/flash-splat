import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3d

# Function to load .ply files and extract data
def load_ply(file_path):
    plydata = PlyData.read(file_path)
    data = np.vstack([plydata['vertex'][axis] for axis in ['x', 'y', 'z']]).T
    return data

def load_ply_all(file_path):
    plydata = PlyData.read(file_path)
    data = np.vstack([plydata['vertex'][axis] for axis in ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz']]).T
    return data

# Function to align two point clouds based on x and y axes only and return as numpy arrays
def align_point_clouds_shift_xy(flash_pointcloud_path, noflash_pointcloud_path):
    # Step 1: Load the point clouds from file paths as numpy arrays
    flash_points = load_ply(flash_pointcloud_path)
    noflash_points = load_ply(noflash_pointcloud_path)

    # Step 2: Compute centroids (mean values) for both point clouds in the x and y axes
    flash_centroid_xy = np.mean(flash_points[:, :2], axis=0)  # Centroid of Flash (x, y)
    noflash_centroid_xy = np.mean(noflash_points[:, :2], axis=0)  # Centroid of NoFlash (x, y)

    # Step 3: Compute the translation (shift) needed to align the centroids
    shift_xy = flash_centroid_xy - noflash_centroid_xy  # Calculate the shift (translation)

    # Step 4: Apply the shift to align NoFlash point cloud to Flash in x and y
    # aligned_noflash_points = noflash_points.copy()  # Deep Copy the NoFlash point cloud
    
    aligned_noflash_points = copy.deepcopy(noflash_points)
    aligned_noflash_points[:, :2] += shift_xy  # Apply the shift to NoFlash in x and y axes

    # Return the Flash point cloud (unchanged) and the shifted NoFlash point cloud
    return flash_points, aligned_noflash_points


def plot_and_separate_transmission_reflection_kmean(which_scene, flash_hist, noflash_hist, ori_data, ori_data_all, xedges, yedges, title, x_label, y_label, flag, proximity_threshold=2, C=1e-3):
    
    # Step 1 - Align the histograms (Placeholder for now)
    # Add alignment code here if necessary

    # Step 2: Compute the difference histogram
    diff_hist = flash_hist - noflash_hist
    diff_hist = diff_hist.T  # Transpose for proper orientation

    # Step 3: Suppress the positive peak
    # Apply absolute value and suppress peaks (works similarly to scaling positive values)
    diff_hist_woPeak = np.abs(diff_hist)**0.8 # default 0.8
    negative_part_flag = (diff_hist < 0).astype(int)
    diff_hist = diff_hist_woPeak * (1 - negative_part_flag) - diff_hist_woPeak * negative_part_flag

    # Step 4: Filter out local bias with a Gaussian Filter
    diff_hist = gaussian_filter(diff_hist, sigma=3.0) # defailt 2.5

    # Step 5: Plot the difference histogram
    plt.figure(figsize=(10, 6))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.subplot(1, 2, 1)
    plt.imshow(diff_hist, origin='lower', aspect='auto', extent=extent, cmap='bwr')
    plt.colorbar(label='Difference in Count')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Kmneans Classification
    # set up kmeans model
    kmeans = KMeans(n_clusters=2, random_state=0)
    # flatten the histogram
    values = diff_hist.ravel().reshape(-1, 1)
    # fit the model
    kmeans.fit(values)
    # predict the labels
    pred_labels = kmeans.predict(values)
    # reshape the labels
    Z = pred_labels.reshape(diff_hist.shape)
    # plot the boundary
    xx, yy = np.meshgrid(xedges[:-1], yedges[:-1])
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='yellow')  # Plot decision boundary at Z=0

    plt.subplot(1, 2, 2)
    # vis the labels
    labels = np.where(Z == 0, 0, 1)
    plt.imshow(labels.reshape(diff_hist.shape), origin='lower', aspect='auto', extent=extent, cmap='bwr')
    plt.colorbar(label='Labels')

    # Step 8: Save the plot
    # plt.savefig(f"transmission_reflection_boundary_kmean_{flag}.png")
    # plt.show()

    # Find the max_x / min_x / max_y / min_y for the label 1
    # x_range | y_range

    max_x, min_x, max_y, min_y = -np.inf, np.inf, -np.inf, np.inf
    for i in range(diff_hist.shape[0]):
        for j in range(diff_hist.shape[1]):
            if labels[i, j] == 1:
                max_x = max(max_x, j)
                min_x = min(min_x, j)
                max_y = max(max_y, i)
                min_y = min(min_y, i)
    # enlarge the range by 1
    max_x += 1
    min_x -= 1
    max_y += 1
    min_y -= 1
    x_bin_length = (xedges[-1] - xedges[0]) / diff_hist.shape[1]
    y_bin_length = (yedges[-1] - yedges[0]) / diff_hist.shape[0]
    min_x_coord = xedges[0] + min_x * x_bin_length
    max_x_coord = xedges[0] + max_x * x_bin_length
    min_y_coord = yedges[0] + min_y * y_bin_length
    max_y_coord = yedges[0] + max_y * y_bin_length

    # So if the coordination of original data is within the range, then it is transmission. We extract it and save it as trans.ply.
    # Otherwise, it is reflection. We extract it and save it as reflect.ply.

    # Extract the transmission and reflection data
    fetch_pc = []
    for i in range(ori_data.shape[0]):
        x, y = ori_data[i, 0], ori_data[i, 2]
        if which_scene == 'Transmission' and (y <= max_y_coord):
            fetch_pc.append(ori_data_all[i])
        elif which_scene == 'Reflection' and (y > max_y_coord):
            fetch_pc.append(ori_data_all[i])
        elif which_scene == 'Beta' and (y <= max_y_coord):
            fetch_pc.append(ori_data_all[i]) 

    # Convert lists to NumPy arrays
    fetch_pc = np.array(fetch_pc)
    return fetch_pc


def fetch_pc_main(ply_pth_flash, ply_pth_noflash, ply_pth_mix, which_scene):
    # Alignment
    flash_data, noflash_data = align_point_clouds_shift_xy(ply_pth_flash, ply_pth_noflash)
    _, ori_data = align_point_clouds_shift_xy(ply_pth_flash, ply_pth_mix)
    ori_data_all = load_ply_all(ply_pth_mix)

    # Compute the 2D histograms for both datasets
    hist_bin = 80
    range_bin = [[-100, 100], [0, 200]]
    hist_noflash, xedges, yedges = np.histogram2d(noflash_data[:, 0], noflash_data[:, 2], bins=hist_bin, range=range_bin)
    hist_flash, _, _ = np.histogram2d(flash_data[:, 0], flash_data[:, 2], bins=hist_bin, range=range_bin)

    # Auto Cut 
    pos_ratio = 0.01
    neg_ratio = 0.01
    # plot_and_separate_transmission_reflection_svm(hist_flash, hist_noflash, xedges, yedges, 'Difference 2D Histogram (Flash - No Flash)', 'X Axis', 'Z Axis', 'difference', proximity_threshold=4, C=1e-3)
    fetch_pc = plot_and_separate_transmission_reflection_kmean(which_scene, hist_flash, hist_noflash, ori_data, ori_data_all, xedges, yedges, 'Difference 2D Histogram (Flash - No Flash)', 'X Axis', 'Z Axis', 'difference', proximity_threshold=4, C=1e-3)

    return fetch_pc