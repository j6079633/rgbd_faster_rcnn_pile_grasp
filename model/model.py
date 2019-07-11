import tensorflow as tf
from model.fast_rcnn.config import cfg
from model.fast_rcnn.test import im_detect
from model.fast_rcnn.nms_wrapper import nms
import model.networks.VGGnet_test as VGGnet_test

import numpy as np
import os, sys, cv2
import argparse

import sd_mask_rcnn_util

CLASSES = ('__background__', 'block')

class Model():
    def __init__(self):
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.__detect_sess = tf.Session(config=config) # init session
        self.__detect_network = VGGnet_test() # load network
        self.__detect_saver = tf.train.Saver(write_version = tf.train.SaverDef.V1) # load model
        self.__detect_saver.restore(self.__detect_sess, "weights/workpiece_2/VGGnet_fast_rcnn_iter_10000.ckpt")

        #initialize for sd mask rcnn
        #__sd_mask_rcnn_model: loaded model
        #yaml_file_path = "cfg/benchmark_one_image_PCA.yaml"
        #self.__sd_mask_rcnn_model, self.__sd_mask_rcnn_graph = sd_mask_rcnn_util.init_sd_mask_rcnn(yaml_file_path)

        #self.__sd_mask_rcnn_model.keras_model.summary()
        #print("Init\n",type(self.__sd_mask_rcnn_model))
        """
        self.__seg_sess = tf.Session(config=config)   
        self.__seg_network = Mask_test()
        self.__seg_saver = tf.train.Saver() # load model
        self.__seg_saver.restore(self._seg_sess, "path to the mask rcnn weights")
        """

    def inference(self, rgb_image, depth_image):
        bboxes = self.detect(rgb_image, depth_image)
        #print("1. ", depth_image.shape)
        #print("2. ", np.rint(bboxes[0]).astype(np.int32))
        #norm_dir_img, norm_dir_cam = self.seg_and_norm(np.rint(bboxes[0]).astype(np.int32), depth_image)
        #img_fill  = self.seg_and_norm((((np.rint(bboxes[0])).astype(np.int32))), depth_image)
        #return np.rint(bboxes).astype(np.int32), norm_dir_img, norm_dir_cam #, resized_mask_image
        return np.rint(bboxes).astype(np.int32), None, None       

    def detect(self, rgb_image, depth_image):
        dets = []
        scores, boxes = im_detect(self.__detect_sess, self.__detect_network, rgb_image)#, depth_image)
        NMS_THRESH = 0.05
        for cls_ind,cls in enumerate(CLASSES[1:]):
            #print("cls_inds: ", cls_ind)
            print(cls)
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        # reorder by the median depth in 0.3 scaled down region
        #print(dets.shape, self._cal_z_median(dets, depth_image, 0.3)[:, np.newaxis].shape)
        dets = np.hstack((dets, self._cal_z_median(dets, depth_image, 0.05)[:, np.newaxis]))
        non_zero_depth = np.where(dets[:, 5]!=0.0)
        dets = dets[non_zero_depth[0], :]
        in_range = np.where(dets[:, 3]>=100)
        dets = dets[in_range[0], :]
        not_bg = np.where(dets[:, 4] > 0.5)
        dets = dets[not_bg[0], :]
        #print(non_zero_depth)
        dets = dets[np.argsort(dets[:, 5])]
        #dets = dets[np.lexsort((dets[:,5],dets[:, 4]))][::-1]
        print(dets)
        #np.lexsort((a[:,2], a[:,1],a[:,0]))
        return dets

    def _cal_z_median(self, rois, z_img, scale_factor):
        """
            rois: (x1, y1, x2, y2)
            z_img: z value
            scale_factor: scale down region
        """
        z_meds = []
        print(z_img.shape)
        for i in range(rois.shape[0]):
            x = rois[i][2] - rois[i][0]
            scale_x = x * scale_factor
            y = rois[i][3] - rois[i][1]
            scale_y = y * scale_factor

            start_x = int(rois[i][0] + (x//2) - (scale_x//2))
            start_y = int(rois[i][1] + (y//2) - (scale_y//2))
            scaled_z_img = z_img[start_y: int(start_y + scale_y), start_x: int(start_x + scale_x)]
            if(np.isnan(np.median(scaled_z_img))):
                z_meds.append(0.0)
            else:
                print(np.median(scaled_z_img))
                print(z_img[int(rois[i][1] + (y//2)), int(rois[i][0] + (x//2))])
                z_meds.append(np.median(scaled_z_img))
            #print(np.median(scaled_z_img))
        return np.array(z_meds)

    def seg_and_norm(self, rois, depth_image):
        '''
        input:
        rois : Faster RCNN reogion of interest ((x1,y1): left top point (x2,y2):right down point)
        depth_image: Full depth raw 16-bits image (1280*720)

        output:
        norm_dir_on_img_coor: [center point (start point),principle axis vector(end point),secondary axis vector(end point),normal vector(end point)] (2D)
        norm_dir_on_cam_coor: [center point,principle axis vector,secondary axis vector,normal vector] (3D)
        '''
        norm_dir_on_cam_coor = []

        #print("seg_and_norm\n")
        #self.__sd_mask_rcnn_model.keras_model.summary()
        #print("Seg_and_norm\n",type(self.__sd_mask_rcnn_model))

        #instrinstic parameter
        fx = 952.067383 
        fy = 952.067383 
        tu = 614.694763 
        tv = 354.57666 
        int_mat = np.array([[fx,0,tu],[0,fy,tv],[0,0,1]])

        #whole image roi
        x1 = 149
        y1 = 157
        x2 = 1253
        y2 = 656
        ori_roi = (x1,y1,x2,y2)

        depth_image = depth_image//10
        depth_image_8bit = sd_mask_rcnn_util.norm_depth_image(depth_image) #normalize 16-bit depth image
        depth_image_8bit_crop = sd_mask_rcnn_util.crop_image(depth_image_8bit,ori_roi) #crop ROI
        depth_image_filled = sd_mask_rcnn_util.inpaint_image(depth_image_8bit_crop) #Fill the hole

        norm_dir_on_cam_coor = []
        norm_dir_on_img_coor = []

        det_masks,_,_,no_padding_range = sd_mask_rcnn_util.calculate_mask(self.__sd_mask_rcnn_model,self.__sd_mask_rcnn_graph,depth_image_filled,ori_roi)

        #depth_img_w_seg_res = sd_mask_rcnn_util.proj_back_to_original_image(depth_image_8bit,resized_mask_image,ori_roi)

        #mask = sd_mask_rcnn_util.gen_obj_mask(depth_img_w_seg_res)

        roi_masks = sd_mask_rcnn_util.get_roi_obj(det_masks,rois,no_padding_range,ori_roi)
        print("ROI object numbers: ",len(roi_masks))
    
        top_mask = sd_mask_rcnn_util.get_top_obj(depth_image,roi_masks)

        depth_16bit_target_facet = sd_mask_rcnn_util.map_to_raw_depth(depth_image,top_mask)

        cam_coor_pts = sd_mask_rcnn_util.map_to_cam_coordinate(depth_16bit_target_facet,int_mat)
    
        pca_on_cam_coor = sd_mask_rcnn_util.PCA_on_cam_coor(cam_coor_pts)
        norm_dir_on_cam_coor.append(pca_on_cam_coor.mean_) #3D center point
        norm_dir_on_cam_coor.append(pca_on_cam_coor.components_[0]) #principle axis vector
        norm_dir_on_cam_coor.append(pca_on_cam_coor.components_[1]) #secondary axis vector
        norm_dir_on_cam_coor.append(pca_on_cam_coor.components_[2]) #normal vector
        
        norm_dir_on_img_coor = sd_mask_rcnn_util.PCA_proj_to_image_coor(pca_on_cam_coor,int_mat)

        print("norm_dir_on_img_coor",norm_dir_on_img_coor)
        print("norm_dir_on_cam_coor",norm_dir_on_cam_coor)
	

        return norm_dir_on_img_coor,norm_dir_on_cam_coor
   
