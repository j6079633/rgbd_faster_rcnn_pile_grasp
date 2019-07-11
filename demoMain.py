import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ui.demoUI import Ui_MainWindow    
import pyrealsense2 as rs
import numpy as np
import cv2

from model.model import Model

from robot.controller import Controller
from robot.gripper import Gripper
import time

class RGBDThread(QThread):
    rgbd_trigger = pyqtSignal(object)
    def __init__(self):
        super(RGBDThread,self).__init__()
        self._mutex = QMutex()
        self._running = True

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        #print("depth scale: ", self.depth_scale)
        #clipping_distance_in_meters = 1 #1 meter
        #self.clipping_distance = clipping_distance_in_meters / self.depth_scale
        #print(clipping_distance)
        self.dmin = 4500.0
        self.dmax = 6000.0
        self.scale = 1
    def __del__(self):
        self.pipeline.stop()
        self.wait()
    def run(self):
        try:
            while self.running():
                # Wait for a coherent pair of frames: depth and color
                #print(self._running)
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    print("continue")
                    continue
                self.d_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                # Convert images to numpy arrays
                self.depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.uint8(color_frame.get_data()) 
                #depth_image_nobg = np.where(self.depth_image > self.clipping_distance, 0, self.depth_image)
                #print(np.amax(depth_image_nobg))
                #depth_image_nobg = np.uint8(depth_image_nobg)
                #self.depth_image_3c = cv2.convertScaleAbs(np.dstack((depth_image_nobg,depth_image_nobg,depth_image_nobg)), alpha=0.03)
                    
                depth_image_nobg = np.where(self.depth_image == 0.0, self.dmax, self.depth_image)
                depth_image_nobg = np.where(depth_image_nobg > self.dmax, self.dmax, depth_image_nobg)
                depth_image_nobg = np.where(depth_image_nobg < self.dmin, self.dmin, depth_image_nobg)
                depth_image_nobg = depth_image_nobg / (self.scale * 10)
                depth_image_nobg = self.normalize(depth_image_nobg)
                
                depth_image_3c = np.dstack((depth_image_nobg,depth_image_nobg,depth_image_nobg))
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                #self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3c, alpha=0.03), cv2.COLORMAP_JET)
                # TODO: normalize to 0-255
                self.color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
                depth_image_3c = cv2.cvtColor(depth_image_3c,cv2.COLOR_BGR2RGB)
                
                self.rgbd_pair = [self.color_image, depth_image_3c, self.depth_image]
                self.rgbd_trigger.emit(self.rgbd_pair)
                #self.depth_trigger.emit(self.depth_image_3c)
                cv2.waitKey(5)
        except NameError as e:
            print(e)
            self.pipeline.stop()
    def pause(self):
        #print("pause streaming")
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()
    def restart(self):
        #print("restart streaming")
        self._mutex.lock()
        self._running = True
        self._mutex.unlock()
        self.run()
    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()

    def get_depth_profile(self):
        return self.depth_scale, self.d_intrin
    def normalize(self, depth):
        min_val = 30
        max_val = 255
        
        d_med = (np.max(depth) + np.min(depth)) / 2.0
        d_diff = depth - d_med
        depth_rev = np.copy(depth)
        depth_rev = d_diff * (-1) + d_med
        depth_rev = depth_rev - np.min(depth_rev)
    
        depth_rev = cv2.normalize(depth_rev, None, min_val, max_val, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return depth_rev


class ModelThread(QThread):
    # control signal: 
        # model signal = true
        # arm signal = false
    # Input: image pair
    # Output: position of bounding box
    # control signal: 
        # model signal = false
        # arm signal = true
    position_trigger = pyqtSignal(object)
    rst_image_trigger = pyqtSignal(object)
    direction_trigger = pyqtSignal(object)
    def __init__(self):
        super(ModelThread,self).__init__()
        self._mutex = QMutex()
        self._running = True
        self._rgb_image = None
        self._depth_image = None
        self._model = Model()
        #self._model = model_VGG()
    def run(self):
        while self.running():
            if (type(self._rgb_image) is None) and (type(self._depth_image) is None):
                print("no input")
                continue
            position, norm_dir_img, norm_dir_cam = self._model.inference(self._rgb_image, self._depth_image)
            #position = self._model.detect(self._rgb_image)
            if len(position) > 0:
                print(len(position))
                self.rst_image_trigger.emit(self.drawDet(position, self._rgb_image,norm_dir_img))
                #self.rst_image_trigger.emit(resized_mask_image)
                #self.position_trigger.emit() # position should be the point to grasp
                self.position_trigger.emit(self.calCenter(position[0])) # position should be the point to grasp
                #TODO: draw 3 directions and center of the grasping point
                #self.direction_trigger.emit(norm_dir_img[0])
                self.stopDetect()
            else:
                print("detect nothing")
                self.stopDetect()
            self._rgb_image = None
            self._depth_image = None
            
            
    def reDetect(self):
        #print("restart streaming")
        self._mutex.lock()
        self._running = True
        self._mutex.unlock()
        self.run()

    def stopDetect(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()
    
    def drawDet(self, dets, rgb_img, norm_dir_img):
        # ordered box list
        rst_image = rgb_img
        
        #change type to integer
        if(norm_dir_img != None):
            norm_dir_img = np.array(norm_dir_img).astype(np.int16)

        for i in range(len(dets)):
            bbox = dets[i, :4]
            #bbox = dets[:4]
            #score = dets[0, -1]
            if i == 0:            
                cv2.rectangle(rst_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [255,0,0], 2)
            else: 
                cv2.rectangle(rst_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0,255,0], 2)
        
        #draw PCA result
        if(norm_dir_img != None):
            cv2.arrowedLine(rst_image,tuple(norm_dir_img[0]),tuple(norm_dir_img[1]),(255,140,0),5)
            cv2.arrowedLine(rst_image,tuple(norm_dir_img[0]),tuple(norm_dir_img[2]),(0,255,0),5)
            cv2.arrowedLine(rst_image,tuple(norm_dir_img[0]),tuple(norm_dir_img[3]),(0,0,255),5)
            cv2.circle(rst_image,tuple(norm_dir_img[0]),10,(130,0,75),-1)
        return rst_image

    def calCenter(self, det):
        center_x = (det[0] + det[2]) / 2
        center_y = (det[1] + det[3]) / 2        
        return [center_x, center_y, det[5]]

class examplePopup(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.lblName = QLabel(self)

class ArmThread(QThread):
    # control signal: working signal of robotarm
    # Input: position of bounding box
    # output: finish signal
    release_trigger = pyqtSignal(object)
    def __init__(self):
        super(ArmThread,self).__init__()
        self._mutex = QMutex()
        self.cntrl = Controller()
        self.grip = Gripper()
        self.trans_mat = np.array([
            [0.6219, -0.0021, 0.],
            [-0.0028, -0.6218, 0.],
            [-337.3547, -163.6015, 1.0]
        ])
        self.baseline_depth = 5850
        self.pose = None
        self._running = True
    def initArmGripper(self):
        # to init arm position
        # init gripper
        self.grip.gripper_reset()
        self.cntrl.power_on()
    
    def run(self):
        while self.running():
            if self.pose == None:
                #print("don't move")
                continue
            else:
                self.pick(self.pose)
                self.pose = None
    def testArm(self):
        rcv = self.cntrl.get_robot_pos()
        pos = self.cntrl.robot_info
        print("pos=",pos)
    def calPosXY(self, camera_point):
        #print(camera_point)
        camera_xy = [camera_point[0], camera_point[1]]
        camera_xy.append(1.0)
        #print(camera_point)
        arm_point = np.array(np.array(camera_xy)) @ self.trans_mat
        print("arm_point", arm_point)
        return arm_point
    def pick(self, camera_point):
        print(camera_point, type(camera_point))
        arm_point = self.calPosXY(camera_point)
        pos = self.cntrl.get_robot_pos()
        #pos = self.cntrl.robot_info
        new_point = arm_point 
        #print(int(new_point[0]*1000), int(new_point[1]*1000), pos[2], pos[3], pos[4], pos[5])
        self.cntrl.move_robot_pos(int(new_point[0]*1000), int(new_point[1]*1000), pos[2], pos[3], pos[4], pos[5], 2000)
        #self.cntrl.move_robot_pos(-339197, -264430, 156320, -1668426, -24203, -74088, 1000)
        self.cntrl.wait_move_end()
        depth_diff = (self.baseline_depth - camera_point[2])
        arm_z = 10000+round(40*depth_diff)
        if arm_z > 156320:
            arm_z = str(156320)
        elif arm_z < 10000:
            arm_z = str(10000)
        else:
            arm_z = str(arm_z)
        #print(10000+round(40*depth_diff))
        
        self.cntrl.move_robot_pos(int(new_point[0]*1000), int(new_point[1]*1000), arm_z, pos[3], pos[4], pos[5], 2000)
        self.cntrl.wait_move_end()
        self.grip.gripper_off()
        time.sleep(0.5)
        self.cntrl.move_robot_pos(int(new_point[0]*1000), int(new_point[1]*1000), pos[2], pos[3], pos[4], pos[5], 2000)
        self.cntrl.wait_move_end()
        self.cntrl.move_robot_pos('-271077', '-415768', '156320', '-1709954', '-1907', '-104123', 2000)
        self.cntrl.wait_move_end()
        self.grip.gripper_on()
        time.sleep(0.3)
        self.release_trigger.emit(True)
        print("here")
        #go home
        #self.cntrl.move_robot_pos('2883', '-246016', '166040', '-1709973', '-1929', '-104740', 2000)
        #self.cntrl.wait_move_end()
        print("fin")
    def goHome(self):
        self.cntrl.move_robot_pos('2883', '-246016', '166040', '-1709973', '-1929', '-104740', 2000)
        self.cntrl.wait_move_end()
    def reGrasp(self):
        #print("restart streaming")
        self._mutex.lock()
        self._running = True
        self._mutex.unlock()
        self.run()

    def stopGrasp(self):
        self.goHome()
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()
    
    

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)        
        self._stream_thread = RGBDThread()
        self._stream_thread.rgbd_trigger.connect(self.updateRGBDFrame)
        self._stream_thread.rgbd_trigger.connect(self.bBoxDetection)

        self._arm_thread = ArmThread()
        self._arm_thread.initArmGripper()
        self._arm_thread.start()
        self._arm_thread.release_trigger.connect(self.Detection)       

        self._model_thread = ModelThread()
        self._model_thread.rst_image_trigger.connect(self.updateResultFrame)
        #self._model_thread.rst_image_trigger.connect(self.buildPopup)
        self._model_thread.position_trigger.connect(self.updateResultPosition)
        self._model_thread.position_trigger.connect(self.pickupObject)        
        self.Stream_PushBt.clicked.connect(self.Start2Stop)
        self.Init_PushBt.clicked.connect(self.InitGripper)
        self.detect_PushBt.clicked.connect(self.Detection)
        self.Set_PushBt.clicked.connect(self.SetDepthVis)

    def Start2Stop(self):
        if(self.Stream_PushBt.text() == "Start Stream"):
            self.Stream_PushBt.setText("Pause Stream")
            if(self._stream_thread._running == True):
                self._stream_thread.start()
            else:
                self._stream_thread.restart()
        else:
            self.Stream_PushBt.setText("Start Stream")
            #print("Pause Streamming")
            self._stream_thread.pause()
    
    def updateRGBDFrame(self, rgbd_image):        
        self._rgb_image = QImage(rgbd_image[0][:], rgbd_image[0].shape[1], rgbd_image[0].shape[0], rgbd_image[0].shape[1] * 3, QImage.Format_RGB888)        
        self.RGBFrame.setPixmap(QPixmap.fromImage(self._rgb_image))
        self._d_image = QImage(rgbd_image[1][:], rgbd_image[1].shape[1], rgbd_image[1].shape[0], rgbd_image[1].shape[1] * 3, QImage.Format_RGB888)
        self.DepthFrame.setPixmap(QPixmap.fromImage(self._d_image))
        QApplication.processEvents()
        
        
    def updateResultFrame(self, result_image):
        self._rst_image = QImage(result_image[:], result_image.shape[1], result_image.shape[0],result_image.shape[1] * 3, QImage.Format_RGB888)
        self.ResultFrame.setPixmap(QPixmap.fromImage(self._rst_image))
        QApplication.processEvents()

    def updateResultPosition(self, position):
        """depth_scale, depth_intrin = self._stream_thread.get_depth_profile()
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [position[1], position[0]], depth_scale)
        self.x_text.setText(str(depth_point[0]))
        self.y_text.setText(str(depth_point[1]))
        self.z_text.setText(str(depth_point[2]))
        """
        self.x_text.setText(str(int(position[0])))
        self.y_text.setText(str(int(position[1])))
        self.z_text.setText(str(int(position[2]/10)))

    def Detection(self):
        
        if(self._model_thread._running == True):
            self._model_thread.start()
            #self._arm_thread.start()
        else:
            self._model_thread.reDetect()

    def bBoxDetection(self, rgbd_image):
        self._model_thread._rgb_image = rgbd_image[0]
        self._model_thread._depth_image = rgbd_image[2]
    
    def buildPopup(self, item):
        self.exPopup = examplePopup()
        self.exPopup.setGeometry(100, 200, item.shape[1], item.shape[0])
        img = QImage(item[:],item.shape[1], item.shape[0],item.shape[1] * 3, QImage.Format_RGB888)        
        self.exPopup.lblName.setPixmap(QPixmap.fromImage(img))
        self.exPopup.show()


    def InitGripper(self):
        #self._arm_thread.initArmGripper()
        self._arm_thread.goHome()
        #self._arm_thread.start()
        #self._arm_thread.testArm()
    
    def pickupObject(self, position):
        if len(position) > 0:
            print("detect ", position)
            #self._arm_thread.pick(position)
            self._model_thread.stopDetect()
            self._arm_thread.pose = position
            self._model_thread._rgb_image = self._stream_thread.color_image
            self._model_thread._depth_image = self._stream_thread.depth_image
            self._model_thread.reDetect()
        else:
            print("arm detect nothing")
            self._arm_thread.goHome()

    def SetDepthVis(self):
        self._stream_thread.dmin = float(self.Mind_Edit.text())
        self._stream_thread.dmax = float(self.Maxd_Edit.text())
        self._stream_thread.dscale = float(self.Scaled_Edit.text())

if __name__ == "__main__":
    print("in main")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
