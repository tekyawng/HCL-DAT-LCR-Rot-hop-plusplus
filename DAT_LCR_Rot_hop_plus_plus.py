# Method for obtaining cross-domain feature representations with Hierarchical Contrastive Learning.
#
# Extends CL-XD-ABSA by Johan Verschoor (2025):
# "Enhancing Cross-Domain Aspect-Based Sentiment Analysis with Contrastive Learning"
#
# HCL extension by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026):
# "Hierarchical Contrastive Learning in Cross-Domain Aspect-Based Sentiment Classification"
# https://github.com/tekyawng/HCL-DAT-LCR-Rot-hop-plusplus/
#
# Erasmus University Rotterdam
#
# Key change: after computing r(0) from LCR-Rot-hop++, apply m hierarchical CL layers.
# At each layer k, compute CL losses on r(k-1), then apply an FFN to get r(k).
# The final representation r(m-1) is fed into the domain and class discriminators.
# When m=1 and beta=[1,0,0], this reduces exactly to Johan's CL-XD-ABSA model.
#
# Adapted from Johan Verschoor (2025):
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Adapted from Knoester, Frasincar, and Trusca (2022)
# https://doi.org/10.1007/978-3-031-20891-1_3

import os
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from nn_layer import class_discriminator, domain_discriminator, bi_dynamic_rnn, reduce_mean_with_len
from utils import load_w2v, batch_index, load_inputs_twitter
from cl import cosine_similarity, contrastive_loss

from tsne_plot import plot_tsne

sys.path.append(os.getcwd())
tf.set_random_seed(1)


def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob, l2, _id='all'):
    """
    Structure of LCR-Rot-hop++ neural network. Method adapted from Trusca et al. (2020).

    :param input_fw: forward input embeddings
    :param input_bw: backward input embeddings
    :param sen_len_fw: forward sentence lengths
    :param sen_len_bw: backward sentence lengths
    :param target: target phrase embeddings
    :param sen_len_tr: target phrase lengths
    :param keep_prob: dropout keep probability
    :param l2: l2 regularisation term
    :param _id: scope identifier string
    :return: concatenated representation r (shape: [batch, 8*n_hidden]) and attention weights
    """
    print('I am lcr_rot_hop_plusplus.')
    cell = tf.contrib.rnn.LSTMCell

    # Left Bi-LSTM.
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')

    # Right Bi-LSTM.
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')

    # Target Bi-LSTM.
    target = tf.nn.dropout(target, keep_prob=keep_prob)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # Left context attention layer.
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'l')
    outputs_l_init = tf.matmul(att_l, hiddens_l)
    outputs_l = tf.squeeze(outputs_l_init)

    # Right context attention layer.
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_r, hiddens_r)
    outputs_r = tf.squeeze(outputs_r_init)

    # Left-aware target attention layer.
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tl')
    outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

    # Right-aware target attention layer.
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tr')
    outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

    # Context and target hierarchical attention layers.
    outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
    outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    # Add two more hops.
    for i in range(2):
        # Left context attention layer.
        att_l = bilinear_attention_layer(hiddens_l, outputs_t_l, sen_len_fw, 2 * FLAGS.n_hidden, l2,
                                         FLAGS.random_base, 'l' + str(i))
        outputs_l_init = tf.matmul(att_l, hiddens_l)
        outputs_l = tf.squeeze(outputs_l_init)

        # Right context attention layer.
        att_r = bilinear_attention_layer(hiddens_r, outputs_t_r, sen_len_bw, 2 * FLAGS.n_hidden, l2,
                                         FLAGS.random_base, 'r' + str(i))
        outputs_r_init = tf.matmul(att_r, hiddens_r)
        outputs_r = tf.squeeze(outputs_r_init)

        # Left-aware target attention layer.
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

        # Right-aware target attention layer.
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

        # Context and target hierarchical attention layers.
        outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
        outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    return outputs_fin, att_l, att_r, att_t_l, att_t_r


def hcl_ffn(inputs, layer_idx, l2, dim=2400, reuse=False):
    """
    Single fully-connected FFN layer for HCL representation refinement.
    Implements eq. (21) from the HCL proposal:
        r(k+1) = tanh(W_FFN(k) * r(k) + b_FFN(k))

    Weight matrix W is initialized from N(0, 0.01) and bias b is initialized to zero,
    following the proposal. Both are trainable and updated during backpropagation.

    Compatible with TensorFlow 1.15 / Python 3.7.
    Use reuse=False on first call (training graph), reuse=True on subsequent calls
    (e.g. test graph) to share the same trained weights.

    :param inputs:    input representation r(k), shape [batch_size, dim]
    :param layer_idx: integer index k, creates unique scope 'hcl_ffn_layer_k'
    :param l2:        l2 regularisation coefficient
    :param dim:       input and output dimension (default 2400 = 4 * 2 * 300)
    :param reuse:     False to create variables, True to reuse existing variables
    :return: refined representation r(k+1), shape [batch_size, dim]
    """
    with tf.variable_scope("hcl_ffn_layer_{}".format(layer_idx), reuse=reuse):
        w = tf.get_variable(
            name='w',
            shape=[dim, dim],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            regularizer=tf.contrib.layers.l2_regularizer(l2)
        )
        b = tf.get_variable(
            name='b',
            shape=[dim],
            initializer=tf.zeros_initializer()
        )
    return tf.nn.tanh(tf.matmul(inputs, w) + b)


def main(train_path_source, train_path_target, test_path,
         learning_rate_dis=0.03, learning_rate_f=0.03, keep_prob=0.3,
         momentum_dis=0.80, momentum_f=0.85, l2_dis=0.001, l2_f=0.001,
         balance_lambda=0.80,
         tau_d=0.1, tau_c=0.1, lambda_dcl=0.2, lambda_ccl=0.2,
         hcl_m=1, hcl_betas=None):
    """
    Runs the HCL-DAT-LCR-Rot-hop++ neural network.

    Hierarchical Contrastive Learning (HCL) extension:
    - hcl_m: number of CL layers (1, 2, or 3). When hcl_m=1 and hcl_betas=[1.0],
      the model reduces exactly to Johan's CL-XD-ABSA with both CCL and DCL active.
    - hcl_betas: list of m weights [beta_1, ..., beta_m] summing to 1.0, controlling
      the contribution of each HCL layer to the total contrastive loss.

    At each layer k in {1, ..., m}:
      1. Compute L(k)_CL,d and L(k)_CL,c on r(k-1)  [eqs. 18-19]
      2. If k < m: apply FFN to get r(k)              [eq. 21]
    The final representation r(m-1) is passed to the domain and class discriminators.
    Total loss follows eq. (24) of the proposal.

    :param train_path_source: training path for source domain
    :param train_path_target: training path for target domain
    :param test_path: test set path
    :param learning_rate_dis: learning rate for domain discriminator
    :param learning_rate_f: learning rate for feature extractor and class discriminator
    :param keep_prob: dropout keep probability
    :param momentum_dis: momentum for domain discriminator
    :param momentum_f: momentum for feature extractor and class discriminator
    :param l2_dis: l2 regularisation for domain discriminator
    :param l2_f: l2 regularisation for feature extractor and class discriminator
    :param balance_lambda: DANN balance parameter (lambda_d)
    :param tau_d: temperature for domain CL
    :param tau_c: temperature for sentiment CL
    :param lambda_dcl: weight of domain CL loss vs domain classification loss
    :param lambda_ccl: weight of sentiment CL loss vs sentiment classification loss
    :param hcl_m: number of HCL layers (1, 2, or 3)
    :param hcl_betas: list of beta weights for each HCL layer, length = hcl_m, must sum to 1
    :return: (acc, pred_errors, fw, bw, tl, tr)
    """
    # Default betas: single layer with full weight (equivalent to Johan's model)
    if hcl_betas is None:
        hcl_betas = [1.0]

    assert len(hcl_betas) == hcl_m, "hcl_betas must have exactly hcl_m entries"
    assert abs(sum(hcl_betas) - 1.0) < 1e-5, "hcl_betas must sum to 1.0"
    assert 1 <= hcl_m <= 3, "hcl_m must be 1, 2, or 3"

    print_config()
    tf.reset_default_graph()
    with tf.device('/gpu:1'):
        # ------------------------------------------------------------------ #
        # Load embeddings
        # ------------------------------------------------------------------ #
        train_word_id_mapping_source, train_w2v_source = load_w2v(FLAGS.train_embedding_source, FLAGS.embedding_dim)
        train_word_embedding_source = tf.constant(train_w2v_source, dtype=np.float32,
                                                  name='train_word_embedding_source')
        train_word_id_mapping_target, train_w2v_target = load_w2v(FLAGS.train_embedding_target, FLAGS.embedding_dim)
        train_word_embedding_target = tf.constant(train_w2v_target, dtype=np.float32,
                                                  name='train_word_embedding_target')
        test_word_id_mapping, test_w2v = load_w2v(FLAGS.test_embedding, FLAGS.embedding_dim)
        test_word_embedding = tf.constant(test_w2v, dtype=np.float32, name='test_word_embedding')

        keep_prob_all = tf.placeholder(tf.float32)

        # ------------------------------------------------------------------ #
        # Input placeholders
        # ------------------------------------------------------------------ #
        with tf.name_scope('inputs'):
            x_src = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y_src = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            d_src = tf.placeholder(tf.float32, [None, FLAGS.n_domain])
            sen_len_src = tf.placeholder(tf.int32, None)
            x_bw_src = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_src = tf.placeholder(tf.int32, [None])
            target_words_src = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_src = tf.placeholder(tf.int32, [None])

            x_tar = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y_tar = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            d_tar = tf.placeholder(tf.float32, [None, FLAGS.n_domain])
            sen_len_tar = tf.placeholder(tf.int32, None)
            x_bw_tar = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_tar = tf.placeholder(tf.int32, [None])
            target_words_tar = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_tar = tf.placeholder(tf.int32, [None])

            x_te = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y_te = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len_te = tf.placeholder(tf.int32, None)
            x_bw_te = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_te = tf.placeholder(tf.int32, [None])
            target_words_te = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_te = tf.placeholder(tf.int32, [None])

        # ------------------------------------------------------------------ #
        # Embedding lookups
        # ------------------------------------------------------------------ #
        inputs_fw_source = tf.nn.embedding_lookup(train_word_embedding_source, x_src)
        inputs_bw_source = tf.nn.embedding_lookup(train_word_embedding_source, x_bw_src)
        target_source = tf.nn.embedding_lookup(train_word_embedding_source, target_words_src)
        inputs_fw_target = tf.nn.embedding_lookup(train_word_embedding_target, x_tar)
        inputs_bw_target = tf.nn.embedding_lookup(train_word_embedding_target, x_bw_tar)
        target_target = tf.nn.embedding_lookup(train_word_embedding_target, target_words_tar)
        inputs_fw_test = tf.nn.embedding_lookup(test_word_embedding, x_te)
        inputs_bw_test = tf.nn.embedding_lookup(test_word_embedding, x_bw_te)
        target_test = tf.nn.embedding_lookup(test_word_embedding, target_words_te)

        # ------------------------------------------------------------------ #
        # LCR-Rot-hop++ — produces r(0) for source, target, and (later) test
        # ------------------------------------------------------------------ #
        with tf.variable_scope("lcr_rot", reuse=tf.AUTO_REUSE):
            outputs_fin_source, alpha_fw_source, alpha_bw_source, alpha_t_l_source, alpha_t_r_source = lcr_rot(
                inputs_fw_source, inputs_bw_source, sen_len_src, sen_len_bw_src,
                target_source, tar_len_src, keep_prob_all, l2_f, 'all')
        with tf.variable_scope("lcr_rot", reuse=tf.AUTO_REUSE):
            outputs_fin_target, alpha_fw_target, alpha_bw_target, alpha_t_l_target, alpha_t_r_target = lcr_rot(
                inputs_fw_target, inputs_bw_target, sen_len_tar, sen_len_bw_tar,
                target_target, tar_len_tar, keep_prob_all, l2_f, 'all')

        # ------------------------------------------------------------------ #
        # Hierarchical Contrastive Learning
        #
        # r_src^(0) = outputs_fin_source   (shape: [batch_src, 2400])
        # r_tar^(0) = outputs_fin_target   (shape: [batch_tar, 2400])
        #
        # For k = 1 ... m:
        #   combined = concat(r_src^(k-1), r_tar^(k-1))
        #   L^(k)_CL,d = contrastive_loss(combined, domain_labels, tau_d)
        #   L^(k)_CL,c = contrastive_loss(r_src^(k-1), sentiment_labels, tau_c)
        #   L^(k)_CL,total = -lambda_d * lambda_dcl * L^(k)_CL,d
        #                    + lambda_ccl * L^(k)_CL,c           [eq. 20]
        #   if k < m:
        #       r_src^(k) = FFN_k(r_src^(k-1))
        #       r_tar^(k) = FFN_k(r_tar^(k-1))   [eq. 21, shared weights]
        #
        # L_HCL = sum_k beta_k * L^(k)_CL,total                 [eq. 23]
        #
        # Final representations passed to discriminators: r_src^(m-1), r_tar^(m-1)
        # ------------------------------------------------------------------ #

        # Concatenated domain labels (used for domain CL at each layer)
        d_combined_all = tf.concat([d_src, d_tar], axis=0)

        # Accumulate domain and class HCL losses separately so the adversarial
        # sign can be applied correctly to the domain part in loss_f.
        hcl_loss_domain = tf.zeros([], dtype=tf.float32)
        hcl_loss_class  = tf.zeros([], dtype=tf.float32)

        # Current representations start at r(0) = LCR-Rot-hop++ output
        r_src_current = outputs_fin_source
        r_tar_current = outputs_fin_target

        for k in range(hcl_m):
            beta_k = float(hcl_betas[k])

            # Combined source+target representation for domain CL at this layer
            r_combined_k = tf.concat([r_src_current, r_tar_current], axis=0)

            # L^(k)_CL,d — domain contrastive loss on combined batch   [eq. 18]
            cl_domain_k = contrastive_loss(r_combined_k, d_combined_all, tau_d)

            # L^(k)_CL,c — sentiment contrastive loss on source batch   [eq. 19]
            cl_class_k = contrastive_loss(r_src_current, y_src, tau_c)

            # Accumulate weighted contributions separately   [eq. 23]
            hcl_loss_domain = hcl_loss_domain + beta_k * lambda_dcl * cl_domain_k
            hcl_loss_class  = hcl_loss_class  + beta_k * lambda_ccl * cl_class_k

            # Apply FFN to refine representation for next layer (only if not last layer)
            if k < hcl_m - 1:
                r_src_current = hcl_ffn(r_src_current, k, l2_f, reuse=False)
                r_tar_current = hcl_ffn(r_tar_current, k, l2_f, reuse=True)

        # Combined for logging only
        hcl_loss_total = hcl_loss_domain + hcl_loss_class

        # After the loop, r_src_current = r^(m-1), r_tar_current = r^(m-1)
        # These are the representations fed to the discriminators.
        outputs_fin_source_final = r_src_current
        outputs_fin_target_final = r_tar_current

        # ------------------------------------------------------------------ #
        # Domain discriminator — uses final HCL representation r^(m-1)
        # ------------------------------------------------------------------ #
        with tf.variable_scope("dis", reuse=tf.AUTO_REUSE):
            prob_domain_source, weights_dis = domain_discriminator(
                outputs_fin_source_final, keep_prob_all, l2_dis, '1', False)
            prob_domain_target, weights_dis = domain_discriminator(
                outputs_fin_target_final, keep_prob_all, l2_dis, '1', False)

        loss_domain_source = loss_func_domain_discr(d_src, prob_domain_source, weights_dis, True)
        loss_domain_target = loss_func_domain_discr(d_tar, prob_domain_target, weights_dis, False)
        acc_num_domain_source, acc_prob_domain_source = acc_func(d_src, prob_domain_source)
        acc_num_domain_target, acc_prob_domain_target = acc_func(d_tar, prob_domain_target)

        loss_domain_target_source = loss_domain_target + loss_domain_source

        # Domain loss: standard classification loss only (CL is handled via hcl_loss_total)
        # This mirrors Johan's (1 - lambda_dcl) * L_d term from eq. (24)
        loss_domain = (1 - lambda_dcl) * loss_domain_target_source
        acc_num_domain = acc_num_domain_source + acc_num_domain_target

        # ------------------------------------------------------------------ #
        # Class discriminator — uses final HCL representation r^(m-1) of source
        # ------------------------------------------------------------------ #
        with tf.variable_scope("class", reuse=tf.AUTO_REUSE):
            prob_class, weights_cls = class_discriminator(
                outputs_fin_source_final, keep_prob_all, l2_f, '1', False)

        loss_class_basic = loss_func_class_discr(y_src, prob_class, weights_cls)

        # Sentiment classification loss: standard part only (CL handled in hcl_loss_total)
        # This mirrors Johan's (1 - lambda_ccl) * L_c term from eq. (24)
        loss_class = (1 - lambda_ccl) * loss_class_basic
        acc_num_class, acc_prob_class = acc_func(y_src, prob_class)

        # ------------------------------------------------------------------ #
        # Total loss — eq. (24) of the HCL proposal exactly:
        #   L_total = -lambda_d*(1-lambda_dcl)*L_d     [domain CE, adversarial]
        #             + (1-lambda_ccl)*L_c              [class CE]
        #             - lambda_d*lambda_dcl*L_HCL_d    [domain CL, adversarial]
        #             + lambda_ccl*L_HCL_c             [class CL]
        #
        # hcl_loss_domain = sum_k beta_k * lambda_dcl * cl_domain_k
        # hcl_loss_class  = sum_k beta_k * lambda_ccl * cl_class_k
        # Both are positive quantities — the adversarial sign is applied here.
        # ------------------------------------------------------------------ #
        loss_f = (loss_class                           # (1-lambda_ccl)*L_c
                  - balance_lambda * loss_domain       # -lambda_d*(1-lambda_dcl)*L_d
                  - balance_lambda * hcl_loss_domain   # -lambda_d*lambda_dcl*L_HCL_d
                  + hcl_loss_class                     # lambda_ccl*L_HCL_c
                  + 0.0001)
        loss_f2 = loss_class - balance_lambda * loss_domain_target_source + 0.0001

        # Debug print — mirrors Johan's print_ops
        print_ops = tf.print(
            "Domain CE Loss:", loss_domain_target_source,
            "Class CE Loss:", loss_class_basic,
            "HCL Loss:", hcl_loss_total,
        )

        # ------------------------------------------------------------------ #
        # Optimisers
        #
        # var_list_f must include:
        #   - LCR-Rot-hop++ variables (first 30 trainable vars, same as Johan)
        #   - Class discriminator variables
        #   - HCL FFN variables (new — needed so FFN weights are trained)
        # ------------------------------------------------------------------ #
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        var_list_D       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        var_list_C       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='class')
        var_list_HCL_FFN = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hcl_ffn')
        var_list_LCR     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lcr_rot')

        # For m=1 (no FFN vars): use trainable[:30] to exactly match Johan's
        # original var_list_f = trainable[:30] + var_list_C.
        # Johan's LCR-Rot-hop++ has 24 vars, so [:30] = 24 LCR + 6 class disc vars.
        # Duplicating those 6 class vars is harmless in TF1 and ensures identical
        # gradient updates, reproducing Johan's results for all 6 domain pairs.
        #
        # For m>1 (FFN vars exist between LCR and disc in creation order):
        # trainable[:30] would accidentally include FFN vars, so we collect
        # explicitly by scope instead.
        if hcl_m == 1:
            trainable = tf.trainable_variables()
            var_list_f = trainable[:30] + var_list_C
        else:
            var_list_f = var_list_LCR + var_list_C + var_list_HCL_FFN

        # Gradient clipping — prevents exploding gradients for aggressive
        # hyperparameter settings (e.g. laptop-restaurant with lr=0.03,
        # momentum=0.90, balance_lambda=1.1, lambda_ccl=0.7, tau_c=0.05).
        # Clip norm of 5.0 is a standard conservative value that does not
        # interfere with normal training but catches runaway gradients.
        opt_domain  = tf.train.MomentumOptimizer(
            learning_rate=learning_rate_dis, momentum=momentum_dis)
        opt_feature = tf.train.MomentumOptimizer(
            learning_rate=learning_rate_f, momentum=momentum_f)

        # Domain discriminator optimiser (maximises loss via -loss_f)
        grads_D, vars_D = zip(*opt_domain.compute_gradients(-loss_f, var_list=var_list_D))
        grads_D_clipped, _ = tf.clip_by_global_norm(grads_D, clip_norm=5.0)
        opti_min_domain = opt_domain.apply_gradients(
            zip(grads_D_clipped, vars_D), global_step=global_step)

        # Feature extractor + class discriminator + HCL FFN optimiser
        grads_f, vars_f = zip(*opt_feature.compute_gradients(loss_f, var_list=var_list_f))
        grads_f_clipped, _ = tf.clip_by_global_norm(grads_f, clip_norm=5.0)
        opti_feature = opt_feature.apply_gradients(
            zip(grads_f_clipped, vars_f), global_step=global_step)

        # ------------------------------------------------------------------ #
        # Test graph — test set passes through LCR-Rot-hop++ and the same
        # HCL FFN layers (reusing trained weights), then through class head.
        # ------------------------------------------------------------------ #
        with tf.variable_scope("lcr_rot", reuse=True):
            outputs_fin_test, alpha_fw_test, alpha_bw_test, alpha_t_l_test, alpha_t_r_test = lcr_rot(
                inputs_fw_test, inputs_bw_test, sen_len_te, sen_len_bw_te,
                target_test, tar_len_te, keep_prob_all, l2_f, 'all')

        # Apply the same HCL FFN layers to test representations (reuse=True shares trained weights)
        r_test_current = outputs_fin_test
        for k in range(hcl_m - 1):
            r_test_current = hcl_ffn(r_test_current, k, l2_f, reuse=True)

        with tf.variable_scope("class", reuse=True):
            prob_class_test, weights_cls_test = class_discriminator(
                r_test_current, keep_prob_all, l2_f, '1', True)

        loss_class_test = loss_func_class_discr(y_te, prob_class_test, weights_cls_test)
        acc_num_class_test, acc_prob_class_test = acc_func(y_te, prob_class_test)
        pred_y = tf.argmax(prob_class_test, 1)
        true_y = tf.argmax(y_te, 1)

        # Combined representations for t-SNE (using final HCL representation)
        outputs_fin_combined = tf.concat([outputs_fin_source_final, outputs_fin_target_final], axis=0)
        d_combined = tf.concat([d_src, d_tar], axis=0)
        y_combined = tf.concat([y_src, y_tar], axis=0)

    # ------------------------------------------------------------------ #
    # Session: load data and train
    # ------------------------------------------------------------------ #
    config_tf = tf.ConfigProto(allow_soft_placement=True)
    config_tf.gpu_options.allow_growth = True
    with tf.Session(config=config_tf) as sess:

        sess.run(tf.global_variables_initializer())

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Train data source. ")
        tr_x_src, tr_sen_len_src, tr_x_bw_src, tr_sen_len_bw_src, tr_y_src, tr_target_word_src, tr_tar_len_src, \
            _, _, _, y_onehot_mapping_src = load_inputs_twitter(
                train_path_source,
                train_word_id_mapping_source,
                FLAGS.max_sentence_len, 'TC', is_r, FLAGS.max_target_len)

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Train data target. ")
        tr_x_tar, tr_sen_len_tar, tr_x_bw_tar, tr_sen_len_bw_tar, tr_y_tar, tr_target_word_tar, tr_tar_len_tar, \
            _, _, _, y_onehot_mapping_tar = load_inputs_twitter(
                train_path_target,
                train_word_id_mapping_target,
                FLAGS.max_sentence_len, 'TC', is_r, FLAGS.max_target_len)

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test data. ")
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, \
            y_onehot_mapping_te = load_inputs_twitter(
                test_path,
                test_word_id_mapping,
                FLAGS.max_sentence_len, 'TC', is_r, FLAGS.max_target_len)

        def get_batch_data(x_f_src, x_f_tar, x_f_te, sen_len_f_src, sen_len_f_tar, sen_len_f_te,
                           x_b_src, x_b_tar, x_b_te, sen_len_b_src, sen_len_b_tar, sen_len_b_te,
                           yi_src, yi_tar, yi_te,
                           batch_target_src, batch_target_tar, batch_target_te,
                           batch_tl_src, batch_tl_tar, batch_tl_te,
                           batch_size_src, batch_size_tar, batch_size_te,
                           keep_pr, domain_src, domain_tar, run_test, is_shuffle=True):
            """Yields feed_dict batches for training and testing."""
            for index_src, index_tar, index_te in batch_index(
                    len(yi_src), len(yi_tar), len(yi_te),
                    batch_size_src, batch_size_tar, batch_size_te,
                    is_shuffle, run_test):
                feed_dict = {
                    x_src: x_f_src[index_src],
                    x_bw_src: x_b_src[index_src],
                    y_src: yi_src[index_src],
                    sen_len_src: sen_len_f_src[index_src],
                    sen_len_bw_src: sen_len_b_src[index_src],
                    target_words_src: batch_target_src[index_src],
                    tar_len_src: batch_tl_src[index_src],
                    d_src: domain_src[index_src],
                    x_tar: x_f_tar[index_tar],
                    y_tar: yi_tar[index_tar],
                    x_bw_tar: x_b_tar[index_tar],
                    sen_len_tar: sen_len_f_tar[index_tar],
                    sen_len_bw_tar: sen_len_b_tar[index_tar],
                    target_words_tar: batch_target_tar[index_tar],
                    tar_len_tar: batch_tl_tar[index_tar],
                    d_tar: domain_tar[index_tar],
                    x_te: x_f_te[index_te],
                    x_bw_te: x_b_te[index_te],
                    sen_len_te: sen_len_f_te[index_te],
                    sen_len_bw_te: sen_len_b_te[index_te],
                    target_words_te: batch_target_te[index_te],
                    tar_len_te: batch_tl_te[index_te],
                    y_te: yi_te[index_te],
                    keep_prob_all: keep_pr,
                }
                if run_test:
                    yield feed_dict, len(index_te)
                else:
                    yield feed_dict, len(index_src), len(index_tar)

        max_acc = 0
        for i in range(FLAGS.n_iter):
            train_count = 0
            train_count_tar = 0
            domain_trainacc = 0
            class_trainacc = 0

            src_domain = np.zeros((len(tr_y_src), 2))
            src_domain[:, 0] = 1
            tar_domain = np.zeros((len(tr_y_tar), 2))
            tar_domain[:, 1] = 1

            # Training loop
            for train, numtrain, train_count_t in get_batch_data(
                    tr_x_src, tr_x_tar, te_x,
                    tr_sen_len_src, tr_sen_len_tar, te_sen_len,
                    tr_x_bw_src, tr_x_bw_tar, te_x_bw,
                    tr_sen_len_bw_src, tr_sen_len_bw_tar, te_sen_len_bw,
                    tr_y_src, tr_y_tar, te_y,
                    tr_target_word_src, tr_target_word_tar, te_target_word,
                    tr_tar_len_src, tr_tar_len_tar, te_tar_len,
                    FLAGS.batch_size_src, FLAGS.batch_size_tar, FLAGS.batch_size_te,
                    keep_prob, src_domain, tar_domain, False):

                train_count += numtrain
                train_count_tar += train_count_t
                _, _, step, _domain_trainacc, _class_trainacc, _ = sess.run(
                    [opti_min_domain, opti_feature, global_step,
                     acc_num_domain, acc_num_class, print_ops],
                    feed_dict=train)
                domain_trainacc += _domain_trainacc
                class_trainacc += _class_trainacc

                # Occasional t-SNE snapshot (~1% of batches)
                # Occasional t-SNE snapshot (~1% of batches)
                o = random.random()
                if o < 0.01:
                    try:
                        out_combined_val, d_combined_val, y_combined_val = sess.run(
                            [outputs_fin_combined, d_combined, y_combined], feed_dict=train)
                        if not np.any(np.isnan(out_combined_val)) and \
                           not np.any(np.isinf(out_combined_val)):
                            # DOMAIN PLOT — comment out when generating sentiment plots
                            # plot_tsne(
                            #     features=out_combined_val,
                            #     labels=d_combined_val,
                            #     plot_title="hcl_m{}_iter{}_{}".format(hcl_m, i, round(o, 4)),
                            #     save_dir="tsne_plots",
                            # )
                            # SENTIMENT PLOT — comment out when generating domain plots
                            plot_tsne(
                                features=out_combined_val,
                                labels=y_combined_val,
                                plot_title="hcl_m{}_iter{}_{}".format(hcl_m, i, round(o, 4)),
                                save_dir="tsne_plots",
                            )
                    except Exception as e:
                        print("[t-SNE] Skipped due to error: {}".format(e))

            # Evaluation on test set
            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []

            for test, num in get_batch_data(
                    tr_x_src, tr_x_tar, te_x,
                    tr_sen_len_src, tr_sen_len_tar, te_sen_len,
                    tr_x_bw_src, tr_x_bw_tar, te_x_bw,
                    tr_sen_len_bw_src, tr_sen_len_bw_tar, te_sen_len_bw,
                    tr_y_src, tr_y_tar, te_y,
                    tr_target_word_src, tr_target_word_tar, te_target_word,
                    tr_tar_len_src, tr_tar_len_tar, te_tar_len,
                    FLAGS.batch_size_te, FLAGS.batch_size_te, FLAGS.batch_size_te,
                    1.0, src_domain, tar_domain, True):

                _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                    [loss_class_test, acc_num_class_test, true_y, pred_y, prob_class_test,
                     alpha_fw_test, alpha_bw_test, alpha_t_l_test, alpha_t_r_test],
                    feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                cost += _loss * num
                cnt += num

            class_trainacc = class_trainacc / train_count
            domain_trainacc = domain_trainacc / (train_count + train_count_tar)
            acc = acc / cnt
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train domain acc={:.6f}, '
                  'class acc={:.6f}, test acc={:.6f}'.format(
                      i, cost, domain_trainacc, class_trainacc, acc))

            if acc > max_acc:
                max_acc = acc
            if np.isnan(cost):
                break

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    "---\nHCL-DAT-LCR-Rot-hop++ (m={}). "
                    "Train accuracy: {:.6f}, Test accuracy: {:.6f}\n".format(
                        hcl_m, class_trainacc, acc))
                results.write("Maximum. Test accuracy: {:.6f}\n---\n".format(max_acc))

        precision = precision_score(ty, py, average=None)
        recall = recall_score(ty, py, average=None)
        f1 = f1_score(ty, py, average=None)
        print('\nP:', precision, 'avg=', sum(precision) / FLAGS.n_class)
        print('R:', recall, 'avg=', sum(recall) / FLAGS.n_class)
        print('F1:', f1, 'avg=', str(sum(f1) / FLAGS.n_class) + '\n')

        with open(FLAGS.prob_file + '.txt', 'w') as fp:
            for item in p:
                fp.write(' '.join([str(it) for it in item]) + '\n')
        with open(FLAGS.prob_file + '_fw.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, fw):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_bw.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, bw):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_tl.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, tl):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_tr.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, tr):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        # Per-sentiment accuracy breakdown
        FLAGS.pos = y_onehot_mapping_te['1']
        FLAGS.neg = y_onehot_mapping_te['-1']
        pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0
        if FLAGS.neutral_sentiment == 1:
            FLAGS.neu = y_onehot_mapping_te['0']
            neu_count, neu_correct = 0, 0

        for idx in range(len(ty)):
            if FLAGS.neutral_sentiment == 1:
                if ty[idx] == FLAGS.pos:
                    pos_count += 1
                    if py[idx] == FLAGS.pos:
                        pos_correct += 1
                elif ty[idx] == FLAGS.neu:
                    neu_count += 1
                    if py[idx] == FLAGS.neu:
                        neu_correct += 1
                else:
                    neg_count += 1
                    if py[idx] == FLAGS.neg:
                        neg_correct += 1
            else:
                if ty[idx] == FLAGS.pos:
                    pos_count += 1
                    if py[idx] == FLAGS.pos:
                        pos_correct += 1
                else:
                    neg_count += 1
                    if py[idx] == FLAGS.neg:
                        neg_correct += 1

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test results.\n")
                results.write("Positive. Correct: {}, Incorrect: {}, Total: {}\n".format(
                    pos_correct, pos_count - pos_correct, pos_count))
                if FLAGS.neutral_sentiment == 1:
                    results.write("Neutral. Correct: {}, Incorrect: {}, Total: {}\n".format(
                        neu_correct, neu_count - neu_correct, neu_count))
                results.write("Negative. Correct: {}, Incorrect: {}, Total: {}\n---\n".format(
                    neg_correct, neg_count - neg_correct, neg_count))

        print('Optimization Finished! Test accuracy={}\n'.format(acc) + ', max accuracy ' + str(max_acc))

        if FLAGS.savable == 1:
            save_dir = "model/" + FLAGS.source_domain + "_" + FLAGS.target_domain + "/"
            saver = saver_func(save_dir)
            saver.save(sess, save_dir)

        return acc, np.where(np.subtract(py, ty) == 0, 0, 1), \
               fw.tolist(), bw.tolist(), tl.tolist(), tr.tolist()


if __name__ == '__main__':
    tf.app.run()
