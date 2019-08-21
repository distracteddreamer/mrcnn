import tensorflow as tf

def get_pad_mask(boxes):
    return tf.reduce_any(tf.not_equal(boxes, 0), axis=-1)

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