import tensorflow as tf


def pyramid_roi_net(inputs, head, **head_kwargs):
    outputs = []
    for fmaps in inputs:
        outputs.append(head(fmaps, **head_kwargs))
    deltas, scores = zip(*outputs)
    return tf.concat(deltas, axis=0), tf.concat(scores, axis=0)