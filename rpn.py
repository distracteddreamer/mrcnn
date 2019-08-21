import tensorflow as tf

from boxes import boxes_to_deltas, deltas_to_boxes
from boxlist import apply_to_batch, BoxList, nms_select, topk_select
from collections import namedtuple
from losses import bbox_loss, class_loss
from roi_net import pyramid_roi_net
from targets import generate_rpn_targets

#TODO
#[] backbone
#[] figure out how to get the pyramid feature maps
#[] can write a note for each component
#[] don't think reduce mean is right - update this properly
#[] track shape
#[] anchor generation


def build_rpn(pyramids, anchors, gt_boxes, config):
    target_boxes, target_labels = generate_rpn_targets(anchors, gt_boxes, config) #(B, A, 4), (B, A)
    deltas, scores = pyramid_roi_net(pyramids, rpn_head, num_anchors=config.rpn.num_anchors) #(B, A, 4), (B, A)
    boxes = deltas_to_boxes(deltas, anchors)
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = boxlist.map(select_rpn_outputs, config)

    samples = boxlist.map(sample_proposals, config.rpn.num_samples)
    samples_clf = samples.masked_select(tf.not_equal(samples.target_labels, 0))
    samples_reg = samples.masked_select(tf.equal(samples.target_labels, 1))

    clf_loss = class_loss(samples_clf.target_labels, samples_clf.scores)
    reg_loss = bbox_loss(boxes_to_deltas(samples_reg.target_boxes, samples_reg.anchors), 
                      boxes_to_deltas(samples_reg.boxes, samples_reg.anchors))
    
    return boxlist, clf_loss, reg_loss

def rpn_head(inputs, num_anchors, filters=256, kernel_size=3):
    # (B, H, W, 256)
    inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding="same")
    # (B, H, W, 4*A)
    deltas = tf.layers.dense(inputs, units=4 * num_anchors)
    # (B, H, W, A)
    scores = tf.layers.dense(inputs, units=num_anchors)
    # (B, H*W*A, 4)
    deltas = tf.reshape(deltas, [tf.shape(deltas)[0], -1, 4])
    # (B, H*W*A)
    scores = tf.reshape(deltas, [tf.shape(deltas)[0], -1])
    return deltas, scores

def select_rpn_outputs(boxlist, config):
    boxlist = topk_select(boxlist)
    boxlist = nms_select(boxlist, config.rpn.max_output_size, config.rpn.iou_threshold)
    return boxlist

def sample_proposals(boxlist, num_samples):
    samples = []
    inds = [tf.where(tf.equal(boxlist.target_labels, label)) for label in [1, -1]]
    nums = [num_samples//2, num_samples - tf.shape(inds[0])[0]]
    for ind, num in zip(inds, nums):
        sample_inds = tf.random.shuffle(inds)[:num]
        samples.append(boxlist.select(sample_inds))
    samples = samples[0].concat(samples[1])
    return samples.pad(num_samples)

