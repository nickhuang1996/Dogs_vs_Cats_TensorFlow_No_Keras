import tensorflow as tf


def get_loss(logits, labels):
    tf.losses.sparse_softmax_cross_entropy(labels, logits)
    total_loss = tf.losses.get_total_loss()
    return total_loss
