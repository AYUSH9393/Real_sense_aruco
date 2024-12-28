import cv2
import numpy as np
from Real_sense_camera import RealSenseCamera

class ArUcoDetector:
    ARUCO_DICT = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    }

    def __init__(self, dict_to_use):
        self.dict_to_use = dict_to_use
        self.arucoDict = cv2.aruco.Dictionary_get(ArUcoDetector.ARUCO_DICT[dict_to_use])
        self.arucoParams = cv2.aruco.DetectorParameters_create()

    def detect(self, image):
        result = cv2.aruco.detectMarkers(image, self.arucoDict, parameters=self.arucoParams)
        return result

    @staticmethod
    def get_image_with_markers(input_image, detect_res):
        image = input_image.copy()
        corners, ids, _ = detect_res

        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                

                # Draw bounding box and center
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -2)
                cv2.putText(image, str(markerID),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image

    @staticmethod
    def get_marker_center(corner_points, depth_image):
        """Calculate the center of the marker using the depth of each corner point."""
        depths = []
        for point in corner_points:
            x, y = int(point[0]), int(point[1])
            depth = depth_image[y, x]
            if depth > 0:
                depths.append(depth)
            else:
                depths.append(0)  # Invalid depth, handle this as needed

        # Calculate the average depth for the marker center
        center_depth = np.mean(depths) if depths else 0
        center_x = np.mean([p[0] for p in corner_points])
        center_y = np.mean([p[1] for p in corner_points])
        center_x = round(center_x,1)
        center_y = round(center_y,1)
        return center_x, center_y, center_depth

    @staticmethod
    def calculate_distance(marker1, marker2):
        """Calculate the Euclidean distance between two 3D points."""
        x1, y1, z1 = marker1
        x2, y2, z2 = marker2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def calculate_corner_distances(marker_corners, depth_image, RealSenseCamera):
        """Calculate distances between adjacent corners of a single ArUco marker."""
        corner_pairs = [
            ("topLeft", "topRight"),
            ("topRight", "bottomRight"),
            ("bottomRight", "bottomLeft"),
            ("bottomLeft", "topLeft"),
        ]
        
        corners = marker_corners.reshape((4, 2))
        distances = {}

        for i, (start_label, end_label) in enumerate(corner_pairs):
            start_point = corners[i]
            end_point = corners[(i + 1) % 4]  # Use modulo for circular connectivity
            
            # Extract depths for the start and end points
            x1, y1 = int(start_point[0]), int(start_point[1])
            x2, y2 = int(end_point[0]), int(end_point[1])

            depth1 = depth_image[y1, x1]
            depth2 = depth_image[y2, x2]

            #round of
            depth1 = round(depth1, 1)  # Round to 1 decimal place
            depth2 = round(depth2, 1)

            if depth1 > 0 and depth2 > 0:
                # Convert pixels to 3D points using RealSense intrinsics
                point1 = RealSenseCamera.pixel_to_world([x1, y1], depth1)
                point2 = RealSenseCamera.pixel_to_world([x2, y2], depth2)

                # Calculate Euclidean distance
                distances[f"{start_label}-{end_label}"] = np.sqrt(
                    (point2[0] - point1[0]) ** 2 +
                    (point2[1] - point1[1]) ** 2 +
                    (point2[2] - point1[2]) ** 2
                )

        return distances

    def angle_between_two_aruco(d1, d2):
        """
        Calculate the angle (in degrees) between the planes of two ArUco markers.
        :param d1: Corner points of the first marker as a 4x3 array (X, Y, Z for each corner).
        :param d2: Corner points of the second marker as a 4x3 array (X, Y, Z for each corner).
        :return: Angle in degrees between the planes.
        """
        d1 = np.array(d1)
        d2 = np.array(d2)

        # Calculate normal vectors for both planes
        n1 = np.cross(d1[2, :] - d1[1, :], d1[2, :] - d1[3, :])    
        n2 = np.cross(d2[2, :] - d2[1, :], d2[2, :] - d2[3, :])   

        # Normalize the normal vectors
        v1_u = n1 / np.linalg.norm(n1)
        v2_u = n2 / np.linalg.norm(n2)

        # Compute the angle between the normal vectors
        angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        angle_degree = np.rad2deg(angle_rad)
        return angle_degree