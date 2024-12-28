import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        # Enable RGB and depth streams
        # self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
        self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

    def start_streaming(self):
        profile = self.pipe.start(self.cfg)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale() * 1000 # In meters, for mm multiple with 1000
        self.intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def stop_streaming(self):
        self.pipe.stop()

    def get_frames(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image,depth_frame

    def pixel_to_world(self, pixel, depth):
        """Convert a pixel (x, y) and its depth value to 3D world coordinates."""
        x, y = pixel
        depth_in_meters = depth * self.depth_scale
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth_in_meters)
        return point  # Returns [X, Y, Z] in meters