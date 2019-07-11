import numpy as np
import os, sys, cv2
import argparse
import tensorflow as tf

import skimage.io as io
import random
import itertools
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import patches,  lines

#3D plot library
'''
from mpl_toolkits.mplot3d import Axes3D
'''

from sklearn.decomposition import PCA

#from mrcnn.visualize import display_images
from mrcnn.model import log
from mrcnn import utils as m_utils

from sd_maskrcnn import utils
#from mrcnn import model as modellib, utils as utilslib, visualize
from mrcnn.utils import resize_image

from autolab_core import YamlConfig
from sd_maskrcnn.config import MaskConfig
from mrcnn import model as modellib

def init_sd_mask_rcnn(yaml_file_path):
    '''
    input: yaml file path
    returns: loaded model
    '''

    # Initialization for sd mask RCNN
    # parse the provided configuration file, set tf settings, and benchmark
    # Change add_argument default for yaml file path
    conf_parser = argparse.ArgumentParser(description="Benchmark SD Mask RCNN model")
    conf_parser.add_argument("--config", action="store", default=yaml_file_path,
                            dest="conf_file", type=str, help="path to the configuration file")
    
    conf_args = conf_parser.parse_args(args=[])
    
    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)

    inference_config = MaskConfig(config['model']['settings'])
    inference_config.GPU_COUNT = 1
    inference_config.IMAGES_PER_GPU = 1
    
    model_dir, _ = os.path.split(config['model']['path'])
    model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config,
                            model_dir=model_dir)

    # Load trained weights
    print("Loading weights from ", config['model']['path'])
    model.load_weights(config['model']['path'], by_name=True)
    graph = tf.get_default_graph()
    return model,graph

def norm_depth_image(depth_image,norm_max_distance=900):
    '''
    inputs:
    depth_image: 16-bit raw depth image
    norm_max_distance: normalize distance range

    return:
    depth_image_8bit: normalized depth image
    '''
    #image width and height, normalize option, save image count
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    #create 8 bit image
    depth_image_8bit = np.zeros((height,width),dtype = np.uint8)

    #normalize image
    depth_image_modify = np.where(depth_image > norm_max_distance, norm_max_distance,depth_image)
    depth_image_8bit = cv2.normalize(depth_image_modify,depth_image_8bit,0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return depth_image_8bit

def crop_image(image,rois):
    '''
    inputs:
    image: image to be cropped
    rois: crop ROI ((x1,y1): left top point (x2,y2):right down point)

    return:
    image: cropped image
    '''
    return image[rois[1]:rois[3],rois[0]:rois[2]]

def inpaint_image(depth_img_8bit_crop):
    '''
    inputs:
    image: image to be inpainted (fill holes)

    return:
    inpainted_image: image which has no hole
    '''
    w =  depth_img_8bit_crop.shape[1]
    h =  depth_img_8bit_crop.shape[0]

    #mask initialize
    mask_zero = np.zeros((h,w),dtype=np.uint8)
    mask = np.zeros((h,w),dtype=np.uint8)

    """
    print(w,h)

    hole = np.count_nonzero(depth_img == 0)
    print("hole:"+str(hole))
    """

    mask_zero = (depth_img_8bit_crop==0) #Find zero pixel value position
    print(mask.shape)
    print(mask_zero.shape)
    mask[mask_zero] = 255 #Use white to mark zero pixel value position

    inpainted_image = cv2.inpaint(depth_img_8bit_crop,mask,3,cv2.INPAINT_TELEA)

    return inpainted_image

def detect_show_instance(image,model,graph,class_names,rois):    
    '''
    Detect facet, remove padding zero and resize image to get ROI size
    ''' 
    # Run object detection
    with graph.as_default():
        results = model.detect([image], verbose=1)

    # Display results
    r = results[0]
    #scores = r['scores']
    resized_mask_image,no_padding_range = display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names = class_names,rois = rois, scores=None, show_bbox=False, show_class=False)
    return resized_mask_image, no_padding_range

def display_instances(image, boxes, masks, class_ids, class_names,rois,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True, show_class=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    rois: Full image ROI patch(x1,y1,x2,y2)
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox, show_class: To show masks, classes, and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    print("{} facet detected!".format(N))

    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):    
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id] if show_class else ""
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
        
    
    ori_masked_image = masked_image.astype(np.uint8)
    #ax.imshow(ori_masked_image)
    
    no_pad_mask_image, no_padding_range = remove_padding(ori_masked_image)
    resized_mask_image = resize_back_roi_size(no_pad_mask_image,rois)
    
    #cv2.imwrite("test.png",resized_mask_image)
    return resized_mask_image, no_padding_range
    # if auto_show:
    #     plt.show()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def remove_padding(image):
    '''
    Given an image, remove zero padding and return the cropped roi range
    '''
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0, _ = coords.min(axis=0)
    x1, y1, _ = coords.max(axis=0) + 1   # slices are exclusive at the top

    #y: in image, it is x axis
    #x: in image, it is y axis
    no_padding_range = [y0,x0,y1,x1]

    # Get the contents of the bounding box.
    return image[x0:x1, y0:y1], no_padding_range

def remove_padding_know_range(image,no_padding_range):
    '''
    Given a range roi, remove zero padding part
    '''
    return image[no_padding_range[1]:no_padding_range[3],no_padding_range[0]:no_padding_range[2]]

def resize_back_roi_size(image,rois):
    '''
    Resize back to roi image size
    '''
    width = rois[2]-rois[0] #x2-x1
    height = rois[3]-rois[1] #y2-y1
    return cv2.resize(image, (width,height), interpolation=cv2.INTER_NEAREST)

def resize_back_full_size(image,rois,width=1280,height=720):
    '''
    Resize mask to full image size (default 1280*720)
    '''
    img = np.zeros((height,width),dtype = np.uint8)
    img[rois[1]:rois[3],rois[0]:rois[2]] = image
    return img

def calculate_mask(model,graph,image,rois):
    '''
    inputs: 
    model: sd mask rcnn model
    graph: model graph (provide keras to calculate)
    image: filled hole 8-bit depth cropped image (specified with rois)
    rois:  Full image ROI patch(x1,y1,x2,y2)

    returns:
    det_masks: mask on 512*512 scale (values=0 or 1)
    det_mask_specific: predict mask on CNN (28*28) (values=0 or 1)
    resized_mask_image: max facet detected image (ROI Size)
    '''

    print('MAKING SD MASK RCNN PREDICTIONS')
    
    class_names = ['bg','fg']

    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    print(image.shape)

    # Resize image for net
    image,_,_,_,_=resize_image(image,max_dim=512)

    #cv2.imshow("Test",image)
    #cv2.waitKey(0)

    # Get predictions of mask head
    #model.keras_model._make_predict_function()
    with graph.as_default():
        mrcnn = model.run_graph([image], [
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    
    
    print("{} detections: {}".format(
        det_count, np.array(class_names)[det_class_ids]))
    

    # Masks
    det_boxes = m_utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                                for i, c in enumerate(det_class_ids)])
    det_masks = np.array([m_utils.unmold_mask(m, det_boxes[i], image.shape)
                        for i, m in enumerate(det_mask_specific)])

    if det_count != 0:
        log("det_mask_specific", det_mask_specific)
        log("det_masks", det_masks)
    else:
        print("No facet detected!!")
        return

    resized_mask_image,no_padding_range = detect_show_instance(image,model,graph,class_names,rois)
    
    return det_masks,det_mask_specific,resized_mask_image,no_padding_range

def proj_back_to_original_image(org_img,img,rois):
    '''
    Resize back to original image size

    inputs:
    org_img: original depth 8-bit image (1280*720)
    img: ROI depth 8-bit image with segment result
    rois: crop ROI ((x1,y1): left top point (x2,y2):right down point)

    return:
    org_img: depth 8-bit image with segmentation result (1280*720)
    '''
    #specify crop range
    x1 = rois[0]
    x2 = rois[2]
    y1 = rois[1]
    y2 = rois[3]
    
    #get directory file
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)
    org_img[y1:y2,x1:x2] = img
    return org_img

def get_roi_obj(det_masks,rois,no_padding_range,ori_rois):
    '''
    Apply rois to obj mask image.
    inputs:
    det_masks: mask images (512*512)
    rois: crop ROI ((x1,y1): left top point (x2,y2):right down point)
    no_padding_range: (x1,y1,x2,y2): image that cotains useful information boundary
    ori_rois: original cropped image

    return:
    img_list: a list contains the masks in the given rois (1280*720)
    '''
    img_list = []
    for i in range (det_masks.shape[0]):
        # Must transpose from (n, h, w) to (h, w, n) and convert from 3D to 2D
        mask = np.transpose((det_masks[i:i+1]*255).astype(np.uint8), (1, 2, 0)).reshape(512,-1)
        mask = remove_padding_know_range(mask,no_padding_range) #512*512 remove padding
        mask = resize_back_roi_size(mask,ori_rois) #resize back to ori_rois size
        mask = resize_back_full_size(mask,ori_rois) #resize back to original image size
        
        #calculate moments and center
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        #if the mask is within rois range
        if rois[0] < cX < rois[2] and rois[1] < cY < rois[3]:
            img_list.append(mask)
    
    return img_list

def get_top_obj(raw_depth_img,roi_masks):
    '''
    Choose top object
    inputs:
    raw_depth_img: raw 16-bit depth image
    roi_masks: a list contains all masks in the roi

    return:
    roi_masks[n]: the toppest mask image in roi_masks
    '''  
    avg_depth = []
    
    for mask in roi_masks:
        depth_roi = np.where(mask==255,raw_depth_img,0)
        #calculate average depth value of each mask
        avg_depth.append(depth_roi[np.nonzero(depth_roi)].mean())
    
    #n is min avg_depth index
    n = np.argmin(avg_depth)
    print("Avg depth of each detected object in ROI: ",avg_depth)
    
    return roi_masks[n]

def gen_obj_mask(img):
    '''
    img.shape[0]: height
    img.shape[1]: width

    input: an image marked with biggest area object 
    output: generate target facet mask (1280*720 binary image)
    '''
    mask = np.zeros((img.shape[0],img.shape[1]))
    
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if not(img[i,j,0]==img[i,j,1]==img[i,j,2]):
                mask[i,j] = 255
    return mask

def map_to_raw_depth(depth_16bit_image,mask):
    '''
    Only preserve the depth value where mask is white
    '''
    return np.where(mask==255,depth_16bit_image,0)

def map_to_cam_coordinate(img,int_mat):
    '''
    input:
    img: 16-bit target facet depth image
    int_mat: camera intrinsic parameter

    output:
    target_obj_3D: target facet on camera 3D coordinate
    '''
    
    target_obj_3D = []
    
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if img[i,j] != 0:
                sacle_factor = img[i,j]
                X = sacle_factor * np.array([i,j,1]).T #img coordinate
                C = np.dot(np.linalg.inv(int_mat),X) #camera coordinate
                #C[2] = img[i,j] #get depth value
                target_obj_3D.append(C)
                
    target_obj_3D = np.array(target_obj_3D)

    return target_obj_3D

def PCA_on_cam_coor(cam_coor_pts):
    '''
    input: 3D camera coordinate points
    output: pca result on 3D camera coordinate points
    '''
    pca = PCA(n_components=3)
    pca.fit(cam_coor_pts) #do PCA computation
    
    '''
    print("PCA compoents:")
    print(pca.components_) #"components" to define the direction of the vector
    print()
    print("PCA variance:")
    print(pca.explained_variance_) #"explained variance" to define the squared-length of the vector
    print()
    print("PCA mean:")
    print(pca.mean_) #center points
    '''
    
    # plot data
    '''
    ax = plt.subplot(projection='3d')
    ax.scatter(cam_coor_pts[:,0], cam_coor_pts[:,1], cam_coor_pts[:,2], c = 'c',alpha = 0.05)
    ax.set_zlabel('Z (mm)')  
    ax.set_ylabel('Y (mm)')
    ax.set_xlabel('X (mm)')
    '''

    '''
    #scale to visualize normal vector
    scale = 3    
    
    i = 0
    for length, vector in zip(pca.explained_variance_, pca.components_):
        c = ['r','g','b']
        v = vector * 3 * np.sqrt(length)

        
        #plot arrow (first three parameter: start point 
        #            last three parameters: end point)
        X = pca.mean_[0]
        Y = pca.mean_[1]
        Z = pca.mean_[2]
        U = v[0]
        V = v[1]
        W = v[2]
        
        ax.quiver(X, Y, Z, U, V, W*scale,color=c[i]) #scale for visualization
        
        i = i+1
    
    ax.plot([pca.mean_[0]], [pca.mean_[1]], [pca.mean_[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)   
    plt.draw()
    '''
    
    return pca

def PCA_proj_to_image_coor(pca_on_cam_coor,int_mat):
    '''
    input:
    pca_on_cam_coor: pca on camera coordinate
    int_mat: camera intrinstic parameters

    output:
    center point (start point),principle axis vector(end point),secondary axis vector(end point),normal vector(end point)
    '''

    #scale to visualize normal vector
    #scale_v1 = 1
    #index to check current loop is principle axis, secondary axis or normal vector
    i = 0
    #arrow color
    #c = ['r','g','b']
    
    '''
    path = glob.glob(config['color_img']['path']+"/*.png")
    img = plt.imread(path[0])
    fig, ax = plt.subplots()
    ax.imshow(img)
    '''

    #output 2D vector
    output_2D_vec = []

    #transform to image coordinate
    I = np.dot(int_mat,pca_on_cam_coor.mean_.T)
    I[0] = round(I[0]/I[2])
    I[1] = round(I[1]/I[2])
    center_pt_on_img = [I[1],I[0]]
    output_2D_vec.append(center_pt_on_img)
    #ax.plot(center_pt_on_img[0],center_pt_on_img[1],'co')
    
    for length, vector in zip(pca_on_cam_coor.explained_variance_, pca_on_cam_coor.components_):
        v = vector * 3 * np.sqrt(length)
        
        #record v1 length
        '''
        if i==1:
            scale_v1 = length
        '''
        
        #normal vector (use unit vector to keep all normal length the same)
        if i==2:
            #v *= (scale_v1/length) * (1/2)
            v = ((v - pca_on_cam_coor.mean_)/np.hypot(v,pca_on_cam_coor.mean_)) * 50
        
        #transform to image coordinate
        J = np.dot(int_mat,pca_on_cam_coor.mean_+v)
        J[0] = round(J[0]/J[2])
        J[1] = round(J[1]/J[2])
        arrow_pt_on_img = [J[1],J[0]]
        output_2D_vec.append(arrow_pt_on_img)
        #draw_vector(center_pt_on_img, arrow_pt_on_img, ax=ax, color = c[i])
        
        i = i + 1

    return output_2D_vec


def draw_vector(v0, v1, ax=None,color='k'):
    '''
    Draw 2D vector on image
    v0: start point
    v1: end point
    '''
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0,color = color)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
