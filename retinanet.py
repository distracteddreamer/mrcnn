def retina_net_head(inputs, num_anchors, num_classes, num_conv):
    def _head(inputs, num_units):
        for i in range(num_conv):
            inputs = tf.layers.conv2d(inputs, filters=256, kernel_size=3, padding="same")
        out = tf.layers.dense(inputs, units=num_units * num_anchors)
        return tf.reshape(out, [tf.shape(out)[0], -1, num_units])
    deltas = _head(inputs, 4)
    scores = _head(inputs, num_classes)
    return deltas, scores

def build_retina_net(pyramids, anchors, gt_boxes):
    pass    