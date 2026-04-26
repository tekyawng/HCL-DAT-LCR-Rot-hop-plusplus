# Main method for running all tests with HCL-DAT-LCR-Rot-hop++.
#
# HCL extension by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026):
# "Hierarchical Contrastive Learning in Cross-Domain Aspect-Based Sentiment Classification"
# https://github.com/tekyawng/HCL-DAT-LCR-Rot-hop-plusplus/
#
# Erasmus University Rotterdam
#
# Based on main_test.py by Johan Verschoor (2025):
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Adapted from Knoester, Frasincar, and Trusca (2022)
# https://doi.org/10.1007/978-3-031-20891-1_3
#
# Key change: Johan's optimal base hyperparameters (Table 5 of his thesis) are FIXED.
# Only the HCL configuration (m and betas) is varied here.
# Setting hcl_m=1, hcl_betas=[1.0] reproduces Johan's CL-XD-ABSA result exactly.

import time
import nltk
from config import *
from load_data import *
import DAT_LCR_Rot_hop_plus_plus

nltk.download('punkt')

# ------------------------------------------------------------------ #
# Johan's optimal base hyperparameters from Table 5 of his thesis.
# These are FIXED and not tuned again — only HCL configs are tuned.
# Keys are "source-target" domain pair strings.
# ------------------------------------------------------------------ #
JOHAN_BASE_HYPERPARAMS = {
    "restaurant-laptop": dict(
        learning_rate_dis=0.005, learning_rate_f=0.005,
        momentum_dis=0.80,       momentum_f=0.90,
        l2_dis=0.001,            l2_f=0.0001,
        balance_lambda=0.6
    ),
    "restaurant-book": dict(
        learning_rate_dis=0.03,  learning_rate_f=0.01,
        momentum_dis=0.90,       momentum_f=0.80,
        l2_dis=0.0001,           l2_f=0.001,
        balance_lambda=1.1
    ),
    "laptop-restaurant": dict(
        learning_rate_dis=0.03,  learning_rate_f=0.01,
        momentum_dis=0.90,       momentum_f=0.80,
        l2_dis=0.001,            l2_f=0.01,
        balance_lambda=1.1
    ),
    "laptop-book": dict(
        learning_rate_dis=0.005, learning_rate_f=0.005,
        momentum_dis=0.85,       momentum_f=0.80,
        l2_dis=0.01,            l2_f=0.001,
        balance_lambda=0.6
    ),
    "book-restaurant": dict(
        learning_rate_dis=0.01,  learning_rate_f=0.01,
        momentum_dis=0.80,       momentum_f=0.85,
        l2_dis=0.0001,           l2_f=0.001,
        balance_lambda=0.6
    ),
    "book-laptop": dict(
        learning_rate_dis=0.01,  learning_rate_f=0.01,
        momentum_dis=0.80,       momentum_f=0.90,
        l2_dis=0.0001,             l2_f=0.01,
        balance_lambda=0.6
    ),
}

# ------------------------------------------------------------------ #
# Johan's optimal CL hyperparameters from Table 7 of his thesis
# (CL-XD-ABSA with both CCL and DCL — full model column).
# These are also fixed; only HCL-specific betas/m are tuned.
# ------------------------------------------------------------------ #
JOHAN_CL_HYPERPARAMS = {
    "restaurant-laptop": dict(tau_d=0.07, tau_c=0.05, lambda_dcl=0.05, lambda_ccl=0.7),
    "restaurant-book":   dict(tau_d=0.2,  tau_c=0.2,  lambda_dcl=0.05, lambda_ccl=0.2),
    "laptop-restaurant": dict(tau_d=0.07, tau_c=0.05, lambda_dcl=0.5,  lambda_ccl=0.7),
    "laptop-book":       dict(tau_d=0.2,  tau_c=0.1,  lambda_dcl=0.5,  lambda_ccl=0.5),
    "book-restaurant":   dict(tau_d=0.07, tau_c=0.1,  lambda_dcl=0.1,  lambda_ccl=0.7),
    "book-laptop":       dict(tau_d=0.1,  tau_c=0.05, lambda_dcl=0.2,  lambda_ccl=0.7),
}


def main(_):
    """
    Runs all specified domain pair tests for HCL-DAT-LCR-Rot-hop++.

    To reproduce Johan's result exactly, set:
        hcl_m=1, hcl_betas=[1.0]
    and ensure tau_d, tau_c, lambda_dcl, lambda_ccl match Table 7.

    To run the HCL extension, set hcl_m and hcl_betas to the values
    obtained from hyperparameter tuning in main_hyper.py.
    """
    # Toggle which domain pairs to run
    rest_lapt  = False
    rest_book  = False
    lapt_rest  = False
    lapt_book  = False
    book_rest  = True
    book_lapt  = False

    write_result = True
    n_iter = 25
    FLAGS.n_iter = n_iter

    # ------------------------------------------------------------------ #
    # HCL configuration — set these from your hyperparameter tuning results.
    #
    # hcl_m:     number of CL layers (1, 2, or 3)
    # hcl_betas: list of m beta weights summing to 1.0
    #
    # Example: Johan's single-layer CL (sanity check):
    #   hcl_m=1, hcl_betas=[1.0]
    #
    # Example: 2-layer HCL with weights 0.6 / 0.4:
    #   hcl_m=2, hcl_betas=[0.6, 0.4]
    # ------------------------------------------------------------------ #
    hcl_m = 3
    hcl_betas = [0.0, 0.0, 1.0]

    if rest_lapt:
        FLAGS.batch_size_src = 24
        FLAGS.batch_size_tar = 15
        FLAGS.batch_size_te  = 701
        run_HCL(source_domain="restaurant", target_domain="laptop",
                year_source=2014, year_target=2014,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    if rest_book:
        FLAGS.batch_size_src = 24
        FLAGS.batch_size_tar = 18
        FLAGS.batch_size_te  = 804
        run_HCL(source_domain="restaurant", target_domain="book",
                year_source=2014, year_target=2019,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    if lapt_rest:
        FLAGS.batch_size_src = 15
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te  = 1122
        run_HCL(source_domain="laptop", target_domain="restaurant",
                year_source=2014, year_target=2014,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    if lapt_book:
        FLAGS.batch_size_src = 20
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te  = 804
        run_HCL(source_domain="laptop", target_domain="book",
                year_source=2014, year_target=2019,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    if book_rest:
        FLAGS.batch_size_src = 18
        FLAGS.batch_size_tar = 24
        FLAGS.batch_size_te  = 1122
        run_HCL(source_domain="book", target_domain="restaurant",
                year_source=2019, year_target=2014,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    if book_lapt:
        FLAGS.batch_size_src = 24
        FLAGS.batch_size_tar = 20
        FLAGS.batch_size_te  = 701
        run_HCL(source_domain="book", target_domain="laptop",
                year_source=2019, year_target=2014,
                hcl_m=hcl_m, hcl_betas=hcl_betas,
                write_result=write_result)

    print('Finished program successfully.')


def run_HCL(source_domain, target_domain, year_source, year_target,
            hcl_m, hcl_betas, write_result):
    """
    Run HCL-DAT-LCR-Rot-hop++ for one domain pair.

    Base hyperparameters are taken from JOHAN_BASE_HYPERPARAMS (fixed).
    CL hyperparameters are taken from JOHAN_CL_HYPERPARAMS (fixed).
    Only hcl_m and hcl_betas are the HCL-specific configuration.

    :param source_domain: source domain name
    :param target_domain: target domain name
    :param year_source:   year of source domain dataset
    :param year_target:   year of target domain dataset
    :param hcl_m:         number of HCL layers (1-3)
    :param hcl_betas:     beta weights per layer, list of length hcl_m, sums to 1
    :param write_result:  write results to file if True
    """
    pair_key = "{}-{}".format(source_domain, target_domain)
    base_hp  = JOHAN_BASE_HYPERPARAMS[pair_key]
    cl_hp    = JOHAN_CL_HYPERPARAMS[pair_key]

    set_other_flags(source_domain=source_domain, source_year=year_source,
                    target_domain=target_domain, target_year=year_target)

    # Apply Johan's fixed base hyperparameters
    FLAGS.balance_lambda    = base_hp["balance_lambda"]
    FLAGS.learning_rate_dis = base_hp["learning_rate_dis"]
    FLAGS.learning_rate_f   = base_hp["learning_rate_f"]
    FLAGS.keep_prob         = base_hp.get("keep_prob", 0.3)
    FLAGS.momentum_dis      = base_hp["momentum_dis"]
    FLAGS.momentum_f        = base_hp["momentum_f"]
    FLAGS.l2_dis            = base_hp["l2_dis"]
    FLAGS.l2_f              = base_hp["l2_f"]

    if write_result:
        with open(FLAGS.results_file, "w") as results:
            results.write("{} to {} | HCL m={} betas={}\n---\n".format(
                source_domain, target_domain, hcl_m, hcl_betas))
        FLAGS.writable = 1

    start_time = time.time()
    _, _, _, _, _, _ = load_data_and_embeddings(FLAGS, False)

    print('Running HCL-DAT-LCR-Rot-hop++ | {} -> {} | m={} betas={}'.format(
        source_domain, target_domain, hcl_m, hcl_betas))

    _, pred2, fw2, bw2, tl2, tr2 = DAT_LCR_Rot_hop_plus_plus.main(
        FLAGS.train_path_source,
        FLAGS.train_path_target,
        FLAGS.test_path,
        learning_rate_dis = base_hp["learning_rate_dis"],
        learning_rate_f   = base_hp["learning_rate_f"],
        keep_prob         = base_hp.get("keep_prob", 0.3),
        momentum_dis      = base_hp["momentum_dis"],
        momentum_f        = base_hp["momentum_f"],
        l2_dis            = base_hp["l2_dis"],
        l2_f              = base_hp["l2_f"],
        balance_lambda    = base_hp["balance_lambda"],
        tau_d             = cl_hp["tau_d"],
        tau_c             = cl_hp["tau_c"],
        lambda_dcl        = cl_hp["lambda_dcl"],
        lambda_ccl        = cl_hp["lambda_ccl"],
        hcl_m             = hcl_m,
        hcl_betas         = hcl_betas,
    )

    end_time = time.time()
    if write_result:
        with open(FLAGS.results_file, "a") as results:
            results.write("Runtime: {:.2f} seconds.\n\n".format(end_time - start_time))


def set_other_flags(source_domain, source_year, target_domain, target_year):
    """
    Sets all path-related FLAGS for the given domain pair.
    Paths match Johan Verschoor's original set_other_flags exactly.
    """
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.source_year   = source_year
    FLAGS.target_year   = target_year

    FLAGS.train_data_source = "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + ".xml"
    FLAGS.train_data_target = "data/externalData/" + FLAGS.target_domain + "_train_" + str(FLAGS.target_year) + ".xml"
    FLAGS.test_data         = "data/externalData/" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + ".xml"

    FLAGS.train_path_source = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(source_year) + "_BERT.txt"
    FLAGS.train_path_target = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_train_" + str(target_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + "_BERT.txt"

    FLAGS.train_embedding_source = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.train_embedding_target = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.prob_file    = 'prob_' + str(FLAGS.source_domain) + "_" + str(FLAGS.target_domain)
    FLAGS.results_file = "Result_Files/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + ".txt"


if __name__ == '__main__':
    tf.app.run()
