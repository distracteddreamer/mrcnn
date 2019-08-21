from collections import namedtuple
import tensorflow as tf
from boxlist import BoxList

#TODO
#[/] coords/deltas transforms
#[] get_pyramid_levels
#[] figure out how to get the pyramid feature maps
#[/] overlaps
#[] put together rpn 
#[] can write a note for each component
#[] don't think reduce mean is right - update this properly
#[] determine the padding flow and find a way to deal with this
#[/] pos/neg image - no positive anchors in such a case
#[/] add pad mask to BoxList
#[] track shape
#[] anchor generation
#[]


# TODO: inputs should be a named tuple with an inputs and padding component 


def apply_to_batch(fn):
    # fn should return a boxlist
    def _fn(boxlist, *args, **kwargs):
        result = tf.map_fn(lambda x: fn(x, *args, **kwargs), boxlist)
        return result
    return _fn
    
def get_pad_mask(boxes):
    return tf.reduce_any(tf.not_equal(boxes, 0), axis=-1)
             
def set_difference(x, y):
    is_any_equal = tf.reduce_any(tf.equal(x[:, None], y[None]), axis=-1)
    return tf.where(tf.logical_not(is_any_equal))

def set_union(x, y):
    return tf.concat([x, set_difference(x, y)], axis=0)

def rpn_head(inputs, num_anchors):
    inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, padding="same")
    inputs = tf.layers.flatten(inputs)
    deltas = tf.layers.dense(inputs, units=4 * num_anchors)
    scores = tf.layers.dense(inputs, units=num_anchors)
    deltas = tf.reshape(deltas, [tf.shape(deltas)[0], -1, 4])
    scores = tf.reshape(deltas, [tf.shape(deltas)[0], -1])
    return deltas, scores

def retina_net_head(inputs, num_anchors, num_classes, num_conv):
    def _head(inputs, num_units):
        for i in range(num_conv):
            inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, padding="same")
        inputs = tf.layers.flatten(inputs)
        out = tf.layers.dense(inputs, units=num_units * num_anchors)
        return tf.reshape(out, [tf.shape(out)[0], -1, num_units])
    deltas = _head(inputs, 4)
    scores = _head(inputs, num_classes)
    return deltas, scores

def pyramid_rpn(inputs, head=rpn_head):
    outputs = []
    for fmaps in inputs:
        outputs.append(head(fmaps))
    deltas, scores = zip(*outputs)
    return tf.concat(deltas, axis=0), tf.concat(scores, axis=0)
    
def topk_select(boxlist):
    # Defined for single image
    keep = tf.nn.top_k(boxlist.scores).indices
    boxlist = boxlist.select(keep)
    return boxlist

def nms_select(boxlist):
    keep = tf.image.non_max_suppression(boxlist.boxes, boxlist.scores,
            max_output_size=config.max_output_size, 
            iou_threshold=config.iou_threshold)
    boxlist = boxlist.select(keep, config.max_output_size)
    return boxlist

@apply_to_batch
def select_rpn_outputs(boxlist):
    boxlist = topk_select(boxlist)
    boxlist = nms_select(boxlist)
    return boxlist

@apply_to_batch
def sample_rpn_outputs(inputs):
    boxes, scores, target_boxes, target_labels, anchors = inputs
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = topk_select(boxlist) #(N,) rois
    boxlist = nms_select(boxlist)
    samples = sample_proposals(boxlist, config.rpn.num_samples)
    return list(samples)


def build_retina_net(pyramids, anchors, gt_boxes):
    pass    


def build_rpn(pyramids, anchors, gt_boxes):
    target_boxes, target_labels = generate_targets(anchors, gt_boxes) #(B, A, 4), (B, A)
    deltas, scores = pyramid_rpn(pyramids) #(B, A, 4), (B, A)
    boxes = deltas_to_boxes(deltas, anchors)
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = select_rpn_outputs(boxlist)

    samples = sample_proposals(boxlist, config.rpn.num_samples)
    samples_clf = samples.trim(tf.not_equal(samples.target_labels, 0))
    samples_reg = samples.trim(tf.equal(samples.target_labels, 1))

    clf_loss = class_loss(samples_clf.target_labels, samples_clf.scores)
    reg_loss = bbox_loss(boxes_to_deltas(samples_reg.target_boxes, samples_reg.anchors), 
                      boxes_to_deltas(samples_reg.boxes, samples_reg.anchors))
    
    return boxlist, clf_loss, reg_loss

def build_roi_pool(feature_maps, rois):
    roi_levels = get_pyramid_levels(rois)
    pooled_features = []
    inds = []

    for level, feature_map in enumerate(feature_maps, 2):
        where = tf.split(tf.where(tf.equal(roi_levels, level)), num_or_size_splits=2, axis=-1)
        pooled = tf.image.crop_and_resize(feature_maps, tf.gather(rois, where), where[:, 0])
        pooled_features.append(pooled)
        inds.append(where[:, 0] * tf.shape(rois)[1] + where[:, 1])
    
    inds = tf.concat(inds, axis=0)
    inds = tf.nn.top_k(inds, tf.shape(inds)).indices

    pooled_features = tf.concat(pooled_features, axis=0)
    pooled_features = tf.gather(pooled_features, inds)
    pooled_features = tf.reshape(pooled_features, (tf.shape(rois)[0], tf.shape(rois)[1],
        config.detector.pool_height, config.detector.pool_width))

    return pooled_features




#########################################################
#TODO: all these should become part of BoxList
# def trim_arr(arr):
#     trim = tf.reduce_any(tf.not_equal(boxes, 0), axis=-1)
#     return tf.boolean_mask(boxes, trim), trim

# def trim_boxlist(boxlist, trim=None):
#     boxes, *arrs = boxlist
#     if trim is not None:
#         boxes = tf.boolean_mask(boxes, trim)
#     else:
#         boxes, trim = trim_arr(boxes)
#     arrs = [tf.boolean_mask(arr, trim) for arr in arrs]
#     return [boxes] + arrs

# def pad_arr(arr, pad_size):
#     padding = tf.concat([
#         [[0, pad_size]],
#         tf.zeros([tf.rank(arr)-1, 2])], axis=0)
#     return tf.pad(arr, padding)

# def pad_boxlist(boxlist, max_size):
#     pad_size = max_size - tf.shape(boxlist[0])[0]
#     return [pad_arr(arr, pad_size) for arr in boxlist]

# def select_boxlist(boxlist, inds):
#     return [tf.gather(arr, inds) for arr in boxlist]
#########################################################

def generate_positive_targets(anchors, target_boxes):
    overlaps = box_iou(anchors, target_boxes) # (n_anchors, n_boxes)
    
    pos_ids_max = tf.argmax(overlaps, axis=0) # (n_boxes,) - values in range [0, n_anchors)
    pos_cond_iou = tf.reduce_any(tf.greater(overlaps, config.rpn.pos_iou_th), axis=-1) # (n_anchors, )
    pos_ids_iou = tf.where(pos_cond_iou)
    pos_ids = set_union(pos_ids_iou, pos_ids_max)  # (n_pos,)

    neg_cond_iou = tf.reduce_all(tf.less(overlaps, config.rpn.neg_iou_th), axis=-1) # (n_anchors, )
    neg_ids_iou = tf.where(neg_cond_iou) # (n_anch)
    neg_ids = set_difference(neg_ids_iou, pos_ids_max) # (n_neg, )

    labels = tf.scatter_nd([tf.shape(anchors)[0]], 
                        tf.concat([pos_ids, neg_ids], 0)[None],
                        tf.concat([tf.ones_like(pos_ids), tf.negative(tf.ones_like(neg_ids))], axis=0))
    
    pos_overlaps = tf.gather(overlaps, pos_ids) # (n_pos, n_boxes)
    pos_target_ids = tf.argmax(pos_overlaps, axis=-1) # (n_pos) - values in the range [0, n_boxes)

    boxes = tf.scatter_nd(tf.shape(anchors),
                        pos_ids,
                        tf.gather(target_boxes, pos_target_ids))

    return boxes, labels


def generate_negative_targets(anchors):
    labels = tf.negative(tf.ones_like(anchors))
    boxes = tf.zeros(anchors)
    return boxes, labels

def generate_targets(anchors, target_boxes):
    has_target_boxes = tf.reduce_any(get_pad_mask(target_boxes))
    return tf.cond(has_target_boxes,
                  lambda: generate_positive_targets(anchors, target_boxes),
                  lambda: generate_negative_targets(anchors))


def sample_proposals(boxlist, num_samples):
    samples = []
    inds = [tf.where(tf.equal(boxlist.target_labels, label)) for label in [1, -1]]
    nums = [num_samples//2, num_samples - tf.shape(inds[0])[0]]
    for ind, num in zip(inds, nums):
        sample_inds = tf.random.shuffle(inds)[:num]
        samples.append(boxlist.select(sample_inds))
    samples = samples[0].concat(samples[1])
    return samples.pad(num_samples)

def class_loss(gt_labels, pred_scores):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=pred_scores)
    return tf.cond(tf.greater(tf.shape(gt_labels)[0], 0),
                            lambda: tf.reduce_mean(losses),
                            lambda: 0.0)

def bbox_loss(pos_gt_deltas, pos_pred_deltas):
    losses = tf.losses.huber_loss(pos_gt_deltas, pos_pred_deltas)
    return tf.cond(tf.greater(tf.shape(pos_gt_deltas)[0], 0),
                            lambda: tf.reduce_mean(losses),
                            lambda: 0.0)


def generate_anchors(*args, **kwargs):
    pass

