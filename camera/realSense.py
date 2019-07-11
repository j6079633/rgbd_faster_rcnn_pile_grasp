import pyrealsense2 as rs
import numpy as np
import cv2

class realSense():
    def __init__(self, RGB_size, D_size):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, D_size[0], D_size[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, RGB_size[0], RGB_size[1], rs.format.bgr8, 30)
        self.color_image = []
        self.depth_image = []
        print("init OK")
    def startStream(self):
        self.pipeline.start(self.config)
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                self.depth_image = np.uint8(depth_frame.get_data())
                self.color_image = np.uint8(color_frame.get_data())  
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                #TODO: normalize to 0-255
    def stopStream(self):
        self.pipeline.stop()
                
