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
