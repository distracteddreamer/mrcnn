from collections import namedtuple
import tensorflow as tf

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


# TODO: inputs should be a named tuple with an inputs and padding component 

BoxListTuple = namedtuple('BoxList', ['boxes', 'scores', 'target_boxes', 'target_labels', 'anchors', 'pad_mask'])

# def apply_to_list(fn, arrs, *args, **kwargs):
#     return BoxList([fn(arr, *args, **kwargs) for arr in arrs])


# def select(self, boxlist, inds):
#     def _select(arr):
#         return tf.gather(arr, inds)
#     return apply_to_list(_trim, boxlist)

# def concat(boxlist1, boxlist2):
#     def _concat(self, arr):
#         arr1, arr2, elem1, elem2 = arr
#         assert elem1 == elem2
#         return tf.concat([arr1, arr2])
#     inputs = zip(boxlist1, boxlist2, boxlist1.keys(), boxlist2.keys())
#     return apply_to_list(_concat, arrs=inputs)

# def trim(boxlist):
#     def _trim(arr): 
#         return tf.boolean_mask(arr, boxlist.padmask)
#     return apply_to_list(_trim, boxlist)

# def pad(boxlist, max_size):
#     pad_size = max_size - tf.shape(boxlist.boxes)[0]
#     def _pad(arr):
#         padding = tf.concat([
#             [[0, pad_size]],
#             tf.zeros([tf.rank(arr)-1, 2])], axis=0)
#         return tf.pad(arr, padding)
#     return apply_to_list(_pad, boxlist)

class BoxList(BoxListTuple):
    # keys = ['boxes', 'scores', 'target_boxes', 'target_labels', 'anchors', 'pad_mask']
    # def __init__(self, boxes, scores, target_boxes, target_labels, anchors):
    #     self.boxes = boxes
    #     self.scores = scores
    #     self.target_boxes = target_boxes
    #     self.target_labels = target_labels
    #     self.anchors = anchors
    #     self.pad_mask = get_pad_mask(target_boxes)
    #     #TODO: [/] elements should be only those which are not None
    #     self.elements = [key for key in self.keys if getattr(self, key) is not None]

    def apply_to_boxlist(self, fn, *args, arrs=None, **kwargs):
        if arrs is None:
            arrs = self
        result = [fn(arr, *args, **kwargs) for arr in arrs]
        return BoxList(*result)

    def _select(self, arr, inds):
        return tf.gather(arr, inds)

    def _pad(self, arr, pad_size):
        padding = tf.concat([
            [[0, pad_size]],
            tf.zeros([tf.rank(arr)-1, 2])], axis=0)
        return tf.pad(arr, padding)

    def _trim(self, arr):
        return tf.boolean_mask(arr, self.pad_mask)

    def _concat(self, arr):
        self_elem, other_elem, elem1, elem2 = arr
        assert elem1 == elem2
        return tf.concat([self_elem, other_elem])

    def trim(self):
        return self.apply_to_boxlist(self._trim)

    def pad(self, max_size):
        pad_size = max_size - tf.shape(self.boxes)[0]
        return self.apply_to_boxlist(self._pad, pad_size)

    def select(self, inds):
        return self.apply_to_boxlist(self._select, inds)

    def concat(self, other):
        inputs = zip(list(self), list(other), self._fields, other._fields)
        return self.apply_to_boxlist(self._concat, arr=inputs)

    # def concat(self, other):
    #     elems = {}
    #     for elem1, elem2 in zip(self.elements, other.elements):
    #         assert elem1 == elem2
    #         elems[elem1] = tf.concat(list(map(getattr, [self, other], [elem1, elem2])), axis=0)
    #     return BoxList(**elems)

    def to_coords(self):
        pass
        # return new box list going from deltas to coords

    def to_deltas(self):
        pass
        # return new box list going from coords to deltas

def apply_to_batch(fn):
    def _fn(inputs, *args, **kwargs):
        result = zip(*tf.map_fn(lambda x: fn(x, *args, **kwargs), inputs))
        return [tf.concat(res, axis=0) for res in result]
    return _fn
    
def get_pad_mask(boxes):
    return tf.reduce_any(tf.not_equal(boxes, 0), axis=-1)
             
def set_difference(x, y):
    is_any_equal = tf.reduce_any(tf.equal(x[:, None], y[None]), axis=-1)
    return tf.where(tf.logical_not(is_any_equal))

def set_union(x, y):
    return tf.concat([x, set_difference(x, y)], axis=0)

def boxes_to_deltas(boxes, ref):
    dims_ref = ref[..., 2:]
    coords_diff = (boxes[..., :2] - ref[..., :2]) / dims_ref
    dims_diff = boxes[..., 2:] / dims_ref
    deltas = tf.log(tf.concat([coords_diff, dims_diff], axis=-1))
    return deltas

def deltas_to_boxes(deltas, ref):
    dims_ref = ref[..., 2:]
    coords_deltas = (deltas[..., :2] * dims_ref) + ref[..., :2]
    dims_deltas = (deltas[..., 2:] * dims_ref) 
    deltas = tf.exp(tf.concat([coords_deltas, dims_deltas], axis=-1))
    return deltas

def box_area(boxes=None, coords=None):
    assert (coords is not None) | (boxes is not None) 
    if coords is None:
        coords_left = boxes[..., ::2] # (..., 2)
        coords_right = boxes[..., 1::2] # (..., 2)
    else:
        coords_left, coords_right = coords
    side_lengths = coords_right - coords_left # (..., 2)
    return tf.reduce_prod(side_lengths, axis=-1) # (...)

def box_intersection(boxes1, boxes2):
    boxes1 = boxes1[..., None, :]  # (...,N1, 1, 2)
    boxes2 = boxes2[..., None, :, :] # (...,1, N2, 2)
    coords_left = tf.maximum(boxes1[...,::2], boxes2[...,::2]) # (..., N1, N2, 2)
    coords_right = tf.maximum(boxes1[...,1::2], boxes2[...,1::2])# (..., N1, N2, 2)
    return box_area([coords_left, coords_right]) # (..., N1, N2)

def box_union(boxes1, boxes2, intr=None):
    if intr is None:
        intr = box_intersection(boxes1, boxes2) # (..., N1, N2)
    area1 = box_area(boxes1)[..., None] # (...,N1, 1)
    area2 = box_area(boxes1)[..., None, :] # (...,1, N2)
    return area1 + area2 - intr  # (..., N1, N2)

def box_iou(boxes1, boxes2, tol=1e-10):
    intr = box_intersection(boxes1, boxes2) # (..., N1, N2)
    uni = box_union(boxes1, boxes2, intr) # (..., N1, N2)
    return (intr + tol)/(uni + tol) # (..., N1, N2)

def generate_proposals(inputs):
    inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3)
    inputs = tf.layers.flatten(inputs)
    deltas = tf.layers.dense(inputs, units=4)
    scores = tf.layers.dense(inputs, units=1)
    return deltas, scores

def pyramid_rpn(inputs):
    outputs = []
    for fmaps in inputs:
        outputs.append(generate_proposals(fmaps))
    deltas, scores = zip(*outputs)
    return tf.concat(deltas, axis=0), tf.concat(scores, axis=0)
    
def topk_select(boxlist):
    # Defined for single image
    _, keep = tf.nn.top_k(boxlist.scores)
    boxlist = boxlist.select(keep)
    return boxlist

def nms_select(boxlist):
    keep = tf.image.non_max_suppression(boxlist.boxes, boxlist.scores,
            max_output_size=config.max_output_size, 
            iou_threshold=config.iou_threshold)
    boxlist = boxlist.select(keep, config.max_output_size)
    return boxlist


@apply_to_batch
def process_rpn_outputs(inputs):
    boxes, scores, target_boxes, target_labels, anchors = inputs
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = topk_select(boxlist) #(N,) rois
    boxlist = nms_select(boxlist)
    return list(boxlist)

def sample_rpn_outputs(inputs):
    boxes, scores, target_boxes, target_labels, anchors = inputs
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = topk_select(boxlist) #(N,) rois
    boxlist = nms_select(boxlist)
    samples = sample_proposals(boxlist, config.rpn.num_samples)
    return list(samples)


def build_rpn(pyramids, anchors, gt_boxes):
    target_boxes, target_labels = generate_targets(anchors, gt_boxes) #(B, A, 4), (B, A)
    deltas, scores = pyramid_rpn(pyramids) #(B, A, 4), (B, A)
    boxes = deltas_to_boxes(deltas, anchors)
    boxlist = BoxList(boxes=boxes, scores=scores, target_boxes=target_boxes, target_labels=target_labels, anchors=anchors)
    boxlist = topk_select(boxlist) #(N,) rois
    boxlist = nms_select(boxlist)
    samples = sample_proposals(boxlist, config.rpn.num_samples)
    clf_loss = class_loss(sample_target_labels, sample_scores)
    reg_loss = bbox_loss(boxes_to_deltas(sample_target_boxes, sample_anchors), 
                      boxes_to_deltas(samples_boxes, sample_anchors), 
                      samples_target_labels)
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
    for label in [-1, 1]:
        inds = tf.where(tf.equal(boxlist.target_labels, label))
        sample_inds = tf.random.shuffle(inds)[:num_samples]
        samples.append(boxlist.select(sample_inds))
    return samples[0].concat(samples[1])

def class_loss(gt_labels, pred_scores):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=pred_scores)
    return tf.reduce_mean(losses)

def bbox_loss(gt_deltas, pred_deltas, gt_labels):
    pos_ids = tf.equal(gt_labels, 1)
    pos_pred_deltas = tf.boolean_mask(pred_deltas, pos_ids)
    pos_gt_deltas = tf.boolean_mask(gt_deltas, pos_ids)
    losses = tf.losses.huber_loss(pos_gt_deltas, pos_pred_deltas)
    return tf.reduce_mean(losses)

