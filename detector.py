import tensorflow as tf

def build_detector(pyramids, rois, config):
    pass

def build_roi_pool(pyramids, rois, pool_size):
    roi_levels = get_pyramid_levels(rois)
    pooled_features = []
    inds = []
    for level, pyramid in enumerate(pyramids, 2):
        where = tf.split(tf.where(tf.equal(roi_levels, level)), num_or_size_splits=2, axis=-1)
        # No batch dimension after gather
        # (num_level_rois, 4)
        level_rois = tf.gather(rois, where)
        # (num_level_rois, pool_height, pool_width)
        pooled = tf.image.crop_and_resize(
            images=pyramid,               
            boxes=level_rois, 
            box_ind=where[:, 0],    
            crop_size=pool_size)
        pooled_features.append(pooled)
        inds.append(where)
    
    # (batch_size * num_rois, 2)
    inds = tf.concat(inds, axis=0)
 
    # (batch_size * num_rois, pool_height, pool_width)
    pooled_features = tf.concat(pooled_features, axis=0)
    pooled_shape = (tf.shape(rois)[0], tf.shape(rois)[1]) +  pool_size
    # Arranges the pooled output to match the order of the input rois whilst restoring the batch dimension
    # (batch_size, num_rois, pool_height, pool_width)
    pooled = tf.scatter_nd(inds, pooled_features, pooled_shape)

    return pooled #_features
