# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from model.fast_rcnn.config import cfg
from model.utils.blob import im_list_to_blob
from model.utils.timer import Timer
import numpy as np
import cv2

def _vis_proposals(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    class_name = 'obj'
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []

    assert len(cfg.TEST.SCALES_BASE) == 1
    im_scale = cfg.TRAIN.SCALES_BASE[0]

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info

def im_proposals(net, im):
    """Generate RPN proposals on a single image."""
    blobs = {}
    blobs['data'], blobs['im_info'] = _get_image_blob(im)
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(
            data=blobs['data'].astype(np.float32, copy=False),
            im_info=blobs['im_info'].astype(np.float32, copy=False))

    scale = blobs['im_info'][0, 2]
    boxes = blobs_out['rois'][:, 1:].copy() / scale
    scores = blobs_out['scores'].copy()
    return boxes, scores

def imdb_proposals(net, imdb):
    """Generate RPN proposals on all images in an imdb."""

    _t = Timer()
    imdb_boxes = [[] for _ in range(imdb.num_images)]
    for i in range(imdb.num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t.tic()
        imdb_boxes[i], scores = im_proposals(net, im)
        _t.toc()
        print ('im_proposals: {:d}/{:d} {:.3f}s' \
              .format(i + 1, imdb.num_images, _t.average_time))
        if 0:
            dets = np.hstack((imdb_boxes[i], scores))
            # from IPython import embed; embed()
            _vis_proposals(im, dets[:3, :], thresh=0.9)
            plt.show()

    return imdb_boxes

def imdb_proposals_det(net, imdb):
    """Generate RPN proposals on all images in an imdb."""

    _t = Timer()
    imdb_boxes = [[] for _ in range(imdb.num_images)]
    for i in range(imdb.num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t.tic()
        boxes, scores = im_proposals(net, im)
        _t.toc()
        print ('im_proposals: {:d}/{:d} {:.3f}s' \
              .format(i + 1, imdb.num_images, _t.average_time))
        dets = np.hstack((boxes, scores))
        imdb_boxes[i] = dets

        if 0:            
            # from IPython import embed; embed()
            _vis_proposals(im, dets[:3, :], thresh=0.9)
            plt.show()

    return imdb_boxes
