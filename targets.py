import tensorflow as tf
from boxes import get_pad_mask, box_iou

def generate_rpn_targets(anchors, target_boxes, config):
    has_target_boxes = tf.reduce_any(get_pad_mask(target_boxes))
    return tf.cond(has_target_boxes,
                  lambda: generate_rpn_positive_targets(anchors, target_boxes, config),
                  lambda: generate_rpn_negative_targets(anchors))

def generate_rpn_positive_targets(anchors, target_boxes, config):
    overlaps = box_iou(anchors, target_boxes) # (n_anchors, n_boxes)
    
    pos_ids_max = tf.argmax(overlaps, axis=0) # (n_boxes,) - values in range [0, n_anchors)
    pos_cond_iou = tf.reduce_any(tf.greater(overlaps, config.rpn.pos_iou_th), axis=-1) # (n_anchors, )
    pos_ids_iou = tf.where(pos_cond_iou)
    pos_ids = tf.sets.set_union(pos_ids_iou, pos_ids_max)  # (n_pos,)

    neg_cond_iou = tf.reduce_all(tf.less(overlaps, config.rpn.neg_iou_th), axis=-1) # (n_anchors, )
    neg_ids_iou = tf.where(neg_cond_iou) # (n_anch)
    neg_ids = tf.sets.set_difference(neg_ids_iou, pos_ids_max) # (n_neg, )

    labels = tf.scatter_nd([tf.shape(anchors)[0]], 
                        tf.concat([pos_ids, neg_ids], 0)[None],
                        tf.concat([tf.ones_like(pos_ids), tf.negative(tf.ones_like(neg_ids))], axis=0))
    
    pos_overlaps = tf.gather(overlaps, pos_ids) # (n_pos, n_boxes)
    pos_target_ids = tf.argmax(pos_overlaps, axis=-1) # (n_pos) - values in the range [0, n_boxes)

    boxes = tf.scatter_nd(tf.shape(anchors),
                        pos_ids,
                        tf.gather(target_boxes, pos_target_ids))

    return boxes, labels

def generate_rpn_negative_targets(anchors):
    labels = tf.negative(tf.ones_like(anchors))
    boxes = tf.zeros(anchors)
    return boxes, labels