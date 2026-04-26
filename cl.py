# Contrastive Loss Function for Cross-Domain Aspect-Based Sentiment Analysis.
#
# HCL extension by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026):
# "Hierarchical Contrastive Learning in Cross-Domain Aspect-Based Sentiment Classification"
# https://github.com/tekyawng/HCL-DAT-LCR-Rot-hop-plusplus/
#
# Originally developed by Johan Verschoor (2025) for the thesis:
# "Enhancing Cross-Domain Aspect-Based Sentiment Analysis with Contrastive Learning."
#
# Numerical stability improvements by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026)
# for the HCL extension. Key change: log-sum-exp trick applied to prevent exp() overflow
# when temperature tau is small (e.g. 0.05), which caused NaN during HCL training.

import tensorflow as tf
import numpy as np


def cosine_similarity(x, y):
    """
    Compute cosine similarity between two tensors.
    Uses epsilon in l2_normalize to prevent division by zero when
    representations collapse to zero vectors (can happen with aggressive
    learning rates or after tanh FFN squashing).

    :param x: Tensor of shape [1, hidden_dim] or [batch_size, hidden_dim]
    :param y: Tensor of shape [batch_size, hidden_dim]
    :return: Cosine similarity, shape [batch_size]
    """
    # Add epsilon to norm before dividing to prevent 0/0 = NaN
    x_norm = tf.nn.l2_normalize(x, axis=-1, epsilon=1e-12)
    y_norm = tf.nn.l2_normalize(y, axis=-1, epsilon=1e-12)
    return tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=-1)


def contrastive_loss(outputs, y_src, tau):
    """
    Numerically stable supervised contrastive loss using the log-sum-exp trick.

    For each anchor i, the loss is:
        L_i = - (1/|P_i|) * sum_{l in P_i} log [
                exp(sim(i,l)/tau) / sum_{j != i} exp(sim(i,j)/tau)
              ]

    Numerical stability: instead of computing exp(sim/tau) directly (which
    overflows for small tau), we subtract the max similarity before exponentiating:
        log-sum-exp: log( sum exp(x_j) ) = max(x) + log( sum exp(x_j - max(x)) )
    This keeps all exp() arguments <= 0, preventing overflow entirely.

    :param outputs: Feature representations, shape [batch_size, hidden_dim]
    :param y_src:   One-hot labels, shape [batch_size, n_class]
    :param tau:     Temperature parameter (must be > 0)
    :return: Scalar mean contrastive loss over the batch
    """
    batch_size = tf.shape(outputs)[0]

    def compute_loss_for_instance(i):
        vi = outputs[i]

        # Scaled cosine similarities between anchor i and all instances [batch_size]
        sim_all = cosine_similarity(tf.expand_dims(vi, axis=0), outputs) / tau

        # Mask: same class as anchor i, excluding self
        yi = y_src[i]
        mask_same_class = tf.cast(
            tf.equal(tf.argmax(y_src, 1), tf.argmax(yi)), tf.float32)
        mask_not_self = 1.0 - tf.one_hot(i, depth=batch_size)
        mask_pos = mask_same_class * mask_not_self   # positive pairs

        # --- Log-sum-exp trick for the denominator ---
        # Only sum over j != i (exclude self from denominator)
        # Set self-similarity to -inf so exp(-inf) = 0 exactly
        neg_inf_self = tf.one_hot(i, depth=batch_size) * (-1e9)
        sim_no_self = sim_all + neg_inf_self         # self position -> -inf

        # Stable log-sum-exp: subtract max before exp
        # epsilon=1e-6 (stronger than 1e-9) prevents log(0) when all
        # exp values underflow to zero in extreme cases
        sim_max = tf.stop_gradient(tf.reduce_max(sim_no_self))
        log_sum_exp_denom = sim_max + tf.math.log(
            tf.reduce_sum(tf.exp(sim_no_self - sim_max)) + 1e-6)

        # Log probability for each positive pair l:
        #   log p(l|i) = sim(i,l)/tau - log_sum_exp_denom
        # Clip to prevent -inf values (from masked positions) propagating
        # as NaN when multiplied by mask_pos
        log_prob = sim_all - log_sum_exp_denom       # shape [batch_size]
        log_prob = tf.clip_by_value(log_prob, -100.0, 0.0)

        # Sum log probs over positive pairs, average by |P_i|
        sum_mask_pos = tf.reduce_sum(mask_pos)
        loss_i = tf.cond(
            sum_mask_pos > 0,
            lambda: -tf.reduce_sum(mask_pos * log_prob) / sum_mask_pos,
            lambda: tf.constant(0.0)
        )
        return loss_i

    loss_per_instance = tf.map_fn(
        compute_loss_for_instance,
        tf.range(batch_size),
        dtype=tf.float32
    )
    return tf.reduce_mean(loss_per_instance)


# Testing
def test_contrastive_loss():
    batch_size = 20
    hidden_dim = 768
    num_classes = 3
    tau = 0.07

    outputs = tf.random.normal([batch_size, hidden_dim], mean=0, stddev=1)
    y_src = tf.one_hot(np.random.randint(0, num_classes, size=batch_size),
                       depth=num_classes)
    loss_value = contrastive_loss(outputs, y_src, tau)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_np = sess.run(loss_value)
        print("Contrastive Loss: {}".format(loss_np))
        assert not np.isnan(loss_np), "Loss is NaN!"
        assert loss_np >= 0, "Loss is negative!"
        print("Test passed.")


if __name__ == "__main__":
    test_contrastive_loss()