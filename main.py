import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Real_sense_camera import RealSenseCamera
from ArUco_Detector import ArUcoDetector

# Camera calibration matrix
# camera_matrix = np.array([
#     [383.640, 0, 312.172],
#     [0, 383.640, 236.375],
#     [0, 0, 1]
# ])

def plot(dist, far, angle):
    plt.figure(figsize=(10, 5))
    
    # Plot angle vs far
    plt.subplot(1, 2, 1)
    plt.scatter(far, angle, c='blue', label='Angle vs Far')
    plt.xlabel("Far")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle vs Far")
    plt.legend()
    
    # Plot distance vs far
    plt.subplot(1, 2, 2)
    plt.scatter(far, dist, c='red', label='Distance vs Far')
    plt.xlabel("Far")
    plt.ylabel("Distance (mm)")
    plt.title("Distance vs Far")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_angle_camera(angle_cam,angle):
    # Plot angle vs angle_cam
    plt.subplot(1, 2, 1)
    plt.scatter(angle_cam, angle, c='green', label='Angle vs angle_cam')
    plt.xlabel("angle_cam")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle vs angle_cam")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_corner_distances(corner_distances):
    """
    Plots the corner distances of ArUco markers, annotating the highest and lowest values.

    Args:
        corner_distances (dict): Dictionary containing corner distances with keys as corner pairs
                                 (e.g., 'topLeft-topRight') and values as distances.
    """
    # Extract keys (corner pairs) and values (distances)
    corner_pairs = list(corner_distances.keys())
    distances = list(corner_distances.values())
    
    # Identify the highest and lowest distances
    max_distance = max(distances)
    min_distance = min(distances)
    max_index = distances.index(max_distance)
    min_index = distances.index(min_distance)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(corner_pairs, distances, color='skyblue', edgecolor='black')

    # Annotate the highest and lowest bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == max_index:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                     ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
        elif i == min_index:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                     ha='center', va='bottom', color='green', fontsize=10, fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                     ha='center', va='bottom', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Corner Pairs', fontsize=12)
    plt.ylabel('Distances (mm)', fontsize=12)
    plt.title('Distances Between ArUco Marker Corners', fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()

def calculate_angle_camera_to_marker(marker_center_3d, camera_position=np.array([0, 0, 0])):
    """
    Calculates the angle between the camera and an ArUco marker.

    Args:
        marker_center_3d (np.ndarray): The 3D coordinates of the ArUco marker center (x, y, z).
        camera_position (np.ndarray): The 3D coordinates of the camera (default is origin [0, 0, 0]).
        
    Returns:
        float: The angle in degrees between the camera and the marker.
    """
    # Define the camera's viewing direction (along the Z-axis by default)
    camera_view_direction = np.array([0, 0, 1])
    
    # Vector from camera to marker
    vector_to_marker = marker_center_3d - camera_position

    # Normalize the vectors
    vector_to_marker_norm = vector_to_marker / np.linalg.norm(vector_to_marker)
    camera_view_norm = camera_view_direction / np.linalg.norm(camera_view_direction)

    # Compute the angle using the dot product
    dot_product = np.dot(vector_to_marker_norm, camera_view_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg
 
if __name__ == '__main__':
    dict_to_use = 'DICT_4X4_50'
    aruco_detector = ArUcoDetector(dict_to_use)
    camera = RealSenseCamera()

    camera.start_streaming()

    try:
        distances, fars, angles, corner_dist, angles_cam = [], [], [], [], []

        while True:
            color_image, depth_image, depth_frame = camera.get_frames()
            if color_image is None or depth_image is None:
                continue

            # Preprocess depth image
            depth_image = cv2.medianBlur(depth_image, 5)
            depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
            depth_image = np.round(depth_image, decimals=1)

            # Mask unaligned parts of color image
            depth_image_3d = np.dstack((depth_image,) * 3)
            masked_color_image = np.where(depth_image_3d <= 0, 153, color_image)

            # Detect markers and draw them
            detection_result = aruco_detector.detect(color_image)
            color_image = ArUcoDetector.get_image_with_markers(color_image, detection_result)

            corners, ids, _ = detection_result

            if ids is not None:
                for marker_corner, marker_id in zip(corners, ids.flatten()):
                    # Calculate and log distances between corners
                    corner_distances = ArUcoDetector.calculate_corner_distances(marker_corner, depth_image, camera)
                    #print(f"Distance corner: {corner_distances}")
                    corner_dist.append(corner_distances)

            if ids is not None and len(corners) > 1:
                ids_sorted, corners_sorted = zip(*sorted(zip(ids.flatten(), corners), key=lambda x: x[0]))

                # Convert marker corners to 3D world coordinates
                marker_3d_corners = []
                for marker_corners in corners_sorted:
                    corners_3d = [camera.pixel_to_world([int(pt[0]), int(pt[1])], depth_image[int(pt[1]), int(pt[0])])
                                  for pt in marker_corners.reshape(4, 2) if depth_image[int(pt[1]), int(pt[0])] > 0]
                    if len(corners_3d) == 4:
                        marker_3d_corners.append(np.array(corners_3d))
                    else:
                        print("Invalid corner size")

                # Calculate angles and distances between markers
                for i in range(len(marker_3d_corners) - 1):
                    angle = ArUcoDetector.angle_between_two_aruco(marker_3d_corners[i], marker_3d_corners[i + 1])
                    print(f"Angle between Markers: {angle}")
                    angles.append(angle)

                for i in range(len(ids_sorted) - 1):
                    marker1_center = ArUcoDetector.get_marker_center(corners_sorted[i].reshape(4, 2), depth_image)
                    marker2_center = ArUcoDetector.get_marker_center(corners_sorted[i + 1].reshape(4, 2), depth_image)

                    avg_depth = (marker1_center[2] + marker2_center[2]) / 2
                    if avg_depth > 0:
                        far = camera.pixel_to_world([(marker1_center[0] + marker2_center[0]) // 2,
                                                     (marker1_center[1] + marker2_center[1]) // 2], avg_depth)
                        fars.append(np.linalg.norm(far))

                    marker1_3D = camera.pixel_to_world(marker1_center[:2], marker1_center[2])
                    marker2_3D = camera.pixel_to_world(marker2_center[:2], marker2_center[2])

                    angle_cam = calculate_angle_camera_to_marker(marker1_3D)
                    print(f"Angle between camera and marker: {angle_cam:.2f} degrees")
                    angles_cam.append(angle_cam)

                    distance = ArUcoDetector.calculate_distance(marker1_3D, marker2_3D)
                    distances.append(distance)
                    print(f"Distance between Markers {ids_sorted[i]} and {ids_sorted[i + 1]}: {distance:.2f} mm")

                    cv2.putText(masked_color_image,
                                f"Distance ({ids_sorted[i]}-{ids_sorted[i + 1]}): {distance:.2f} mm",
                                (10, 30 + 30 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Display results
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            masked_color_image = ArUcoDetector.get_image_with_markers(masked_color_image, detection_result)
            cv2.imshow('RealSense ArUco Detection', masked_color_image)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()

 #Plot data periodically
            # if len(distances) > 100:  # frame number
            #     plot(distances, fars, angles)
            #     #plot_corner_distances(corner_distances)
            #     #plot_angle_camera(angles_cam,angles)
            #     distances.clear()
            #     fars.clear()
            #     angles.clear()
            #     corner_dist.clear()
            #     angles_cam.clear()