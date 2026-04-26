# Hyperparameter tuning for HCL-DAT-LCR-Rot-hop++ using Tree Parzen Estimator (TPE).
#
# HCL extension by Tek Yaw Ng, Lotte van den Berg, Jason Tran (2026):
# "Hierarchical Contrastive Learning in Cross-Domain Aspect-Based Sentiment Classification"
# https://github.com/tekyawng/HCL-DAT-LCR-Rot-hop-plusplus/
#
# Erasmus University Rotterdam
#
# Based on main_hyper.py by Johan Verschoor (2025):
# https://github.com/Johan-Verschoor/CL-XD-ABSA/
#
# Adapted from Knoester, Frasincar, and Trusca (2022)
# https://doi.org/10.1007/978-3-031-20891-1_3
#
# Key change: Johan's base hyperparameters (Table 5) AND CL hyperparameters (Table 7)
# are FIXED per domain pair. Only the HCL configuration is tuned:
#   - hcl_m: number of CL layers (1, 2, or 3)
#   - hcl_betas: per-layer weights from a predefined discrete set (Appendix A, Table 5
#     of the proposal, restricted to m <= 3)
#
# The predefined HCL configurations are the same as in the proposal Appendix A,
# but filtered to m in {1, 2, 3} only (as per the revised experimental scope).

import json
import os
import pickle
from functools import partial

from bson import json_util
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

import DAT_LCR_Rot_hop_plus_plus
from config import *
from load_data import *

global eval_num, best_loss, best_hyperparams

# ------------------------------------------------------------------ #
# Johan's optimal base hyperparameters — FIXED, not tuned.
# Source: Table 5 of Johan Verschoor's thesis (2025).
# ------------------------------------------------------------------ #
JOHAN_BASE_HYPERPARAMS = {
    "restaurant-laptop": dict(
        learning_rate_dis=0.005, learning_rate_f=0.005,
        momentum_dis=0.80,       momentum_f=0.90,
        l2_dis=0.001,            l2_f=0.0001,
        keep_prob=0.3,           balance_lambda=0.6
    ),
    "restaurant-book": dict(
        learning_rate_dis=0.03,  learning_rate_f=0.01,
        momentum_dis=0.90,       momentum_f=0.80,
        l2_dis=0.0001,           l2_f=0.001,
        keep_prob=0.3,           balance_lambda=1.1
    ),
    "laptop-restaurant": dict(
        learning_rate_dis=0.03,  learning_rate_f=0.01,
        momentum_dis=0.90,       momentum_f=0.80,
        l2_dis=0.001,            l2_f=0.01,
        keep_prob=0.3,           balance_lambda=1.1
    ),
    "laptop-book": dict(
        learning_rate_dis=0.005, learning_rate_f=0.005,
        momentum_dis=0.85,       momentum_f=0.80,
        l2_dis=0.01,             l2_f=0.001,
        keep_prob=0.3,           balance_lambda=0.6
    ),
    "book-restaurant": dict(
        learning_rate_dis=0.01,  learning_rate_f=0.01,
        momentum_dis=0.80,       momentum_f=0.85,
        l2_dis=0.0001,           l2_f=0.001,
        keep_prob=0.3,           balance_lambda=0.6
    ),
    "book-laptop": dict(
        learning_rate_dis=0.01,  learning_rate_f=0.01,
        momentum_dis=0.80,       momentum_f=0.90,
        l2_dis=0.0001,             l2_f=0.01,
        keep_prob=0.3,           balance_lambda=0.6
    ),
}

# ------------------------------------------------------------------ #
# Johan's optimal CL hyperparameters — FIXED, not tuned.
# Source: Table 7 of Johan Verschoor's thesis (2025), full CL-XD-ABSA column.
# ------------------------------------------------------------------ #
JOHAN_CL_HYPERPARAMS = {
    "restaurant-laptop": dict(tau_d=0.07, tau_c=0.05, lambda_dcl=0.05, lambda_ccl=0.7),
    "restaurant-book":   dict(tau_d=0.2,  tau_c=0.2,  lambda_dcl=0.05, lambda_ccl=0.2),
    "laptop-restaurant": dict(tau_d=0.07, tau_c=0.05, lambda_dcl=0.5,  lambda_ccl=0.7),
    "laptop-book":       dict(tau_d=0.2,  tau_c=0.1,  lambda_dcl=0.5,  lambda_ccl=0.5),
    "book-restaurant":   dict(tau_d=0.07, tau_c=0.1,  lambda_dcl=0.1,  lambda_ccl=0.7),
    "book-laptop":       dict(tau_d=0.1,  tau_c=0.05, lambda_dcl=0.2,  lambda_ccl=0.7),
}

# ------------------------------------------------------------------ #
# Predefined HCL configurations — the ONLY things being tuned.
# Each entry is (m, [beta_1, ..., beta_m]).
# Restricted to m in {1, 2, 3} per revised experimental scope.
#
# The search space is a complete evenly-spaced simplex grid:
#   m=1:  1 config  (trivial)
#   m=2: 11 configs (step 0.1 — fine-grained, cheap to evaluate)
#   m=3: 21 configs (step 0.2 — coarser to keep compute manageable)
#   Total: 33 configs
# ------------------------------------------------------------------ #
def _build_hcl_configs():
    configs = []

    # m = 1: only one possibility
    configs.append((1, [1.0]))

    # m = 2: full simplex b1 + b2 = 1, step 0.1 (11 configs)
    # Keep fine-grained for m=2 since it is cheap to evaluate
    for _b1 in range(0, 11):
        b1 = round(_b1 / 10, 1)
        b2 = round(1.0 - b1, 1)
        configs.append((2, [b1, b2]))

    # m = 3: full simplex b1 + b2 + b3 = 1, step 0.2 (21 configs)
    # Coarser grid to keep total compute manageable while still
    # covering all regions of the weight space evenly.
    for _b1 in range(0, 6):
        for _b2 in range(0, 6 - _b1):
            b1 = round(_b1 / 5, 1)
            b2 = round(_b2 / 5, 1)
            b3 = round(1.0 - b1 - b2, 1)
            b3 = 0.0 if b3 == 0.0 else b3  # fix -0.0 float artifact
            configs.append((3, [b1, b2, b3]))

    return configs

HCL_CONFIGS = _build_hcl_configs()


def main():
    """
    Runs HCL hyperparameter tuning for the specified domain pairs.
    Only the HCL configuration (m + betas) is tuned.
    All other hyperparameters are fixed to Johan's optimal values.
    """
    runs = 10
    n_iter = 10

    # (source_name, source_year, target_name, target_year, batch_src, batch_tar, batch_te)
    rest_lapt    = ["restaurant", 2014, "laptop",     2014, 24, 15, 701]
    rest_book    = ["restaurant", 2014, "book",       2019, 24, 18, 804]
    lapt_rest    = ["laptop",     2014, "restaurant", 2014, 20, 24, 1122]
    lapt_book    = ["laptop",     2014, "book",       2019, 20, 24, 804]
    book_rest    = ["book",       2019, "restaurant", 2014, 18, 24, 1122]
    book_lapt    = ["book",       2019, "laptop",     2014, 24, 20, 701]

    domains = [lapt_rest]  # edit to run multiple pairs

    for domain in domains:
        run_hyper(
            source_domain=domain[0], source_year=domain[1],
            target_domain=domain[2], target_year=domain[3],
            batch_size_src=domain[4], batch_size_tar=domain[5],
            batch_size_te=domain[6],
            runs=runs, n_iter=n_iter
        )


def run_hyper(source_domain, source_year, target_domain, target_year,
              batch_size_src, batch_size_tar, batch_size_te, runs, n_iter):
    """
    Runs HCL hyperparameter tuning for one domain pair.

    :param source_domain: source domain name
    :param source_year:   year of source dataset
    :param target_domain: target domain name
    :param target_year:   year of target dataset
    :param batch_size_src: batch size for source domain
    :param batch_size_tar: batch size for target domain
    :param batch_size_te:  total test set size
    :param runs:          number of TPE optimisation runs
    :param n_iter:        training epochs per evaluation
    """
    pair_key = "{}-{}".format(source_domain, target_domain)
    path = "hyper_results/HCL/{}/{}/".format(source_domain, target_domain)

    FLAGS.source_domain   = source_domain
    FLAGS.source_year     = source_year
    FLAGS.target_domain   = target_domain
    FLAGS.target_year     = target_year
    FLAGS.batch_size_src  = batch_size_src
    FLAGS.batch_size_tar  = batch_size_tar
    FLAGS.batch_size_te   = batch_size_te
    FLAGS.n_iter          = n_iter

    FLAGS.train_data_source = "data/externalData/" + source_domain + "_train_" + str(source_year) + ".xml"
    FLAGS.train_data_target = "data/externalData/" + target_domain + "_train_" + str(target_year) + ".xml"
    FLAGS.test_data         = "data/externalData/" + target_domain + "_test_"  + str(target_year) + ".xml"

    FLAGS.train_path_source = "data/programGeneratedData/BERT/" + source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + source_domain + "_train_" + str(source_year) + "_BERT.txt"
    FLAGS.train_path_target = "data/programGeneratedData/BERT/" + target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + target_domain + "_train_" + str(target_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + target_domain + "_test_" + str(target_year) + "_BERT.txt"

    FLAGS.train_embedding_source = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + source_domain + "_" + str(
        source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.train_embedding_target = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + target_domain + "_" + str(
        target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + target_domain + "_" + str(
        target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    FLAGS.hyper_train_path_source = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hypertraindatasource' + "_" + source_domain + "_" + target_domain + ".txt"
    FLAGS.hyper_train_path_target = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hypertraindatatarget' + "_" + source_domain + "_" + target_domain + ".txt"
    FLAGS.hyper_eval_path_target  = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hyperevaldatatarget' + "_" + source_domain + "_" + target_domain + ".txt"

    FLAGS.results_file = "hyper_results/HCL/" + source_domain + "/" + target_domain + "/hyperresults.txt"

    _, _, _, _, _, _ = load_hyper_data(FLAGS, shuffle=False)

    # The search space is a categorical choice over the predefined HCL configurations.
    # hyperopt selects one index, which maps to (m, betas).
    hcl_space = hp.choice('hcl_config_idx', list(range(len(HCL_CONFIGS))))

    global eval_num, best_loss, best_hyperparams
    eval_num = 0
    best_loss = None
    best_hyperparams = None

    for i in range(runs):
        print("HCL Optimisation Run {} / {} for {} -> {}".format(
            i + 1, runs, source_domain, target_domain))
        run_a_trial(hcl_space, path, pair_key)
        plot_best_model(path)


def hcl_objective(hcl_config_idx, path, pair_key):
    """
    Objective function for TPE: trains the HCL model with the selected
    HCL configuration and returns the validation accuracy as loss.

    :param hcl_config_idx: integer index into HCL_CONFIGS
    :param path:           save path for results
    :param pair_key:       domain pair string e.g. "book-restaurant"
    :return: hyperopt result dict
    """
    global eval_num, best_loss, best_hyperparams
    eval_num += 1

    hcl_m, hcl_betas = HCL_CONFIGS[hcl_config_idx]
    base_hp = JOHAN_BASE_HYPERPARAMS[pair_key]
    cl_hp   = JOHAN_CL_HYPERPARAMS[pair_key]

    print("Eval {}: HCL config idx={} -> m={}, betas={}".format(
        eval_num, hcl_config_idx, hcl_m, hcl_betas))

    l, pred1, fw1, bw1, tl1, tr1 = DAT_LCR_Rot_hop_plus_plus.main(
        FLAGS.hyper_train_path_source,
        FLAGS.hyper_train_path_target,
        FLAGS.hyper_eval_path_target,
        learning_rate_dis = base_hp["learning_rate_dis"],
        learning_rate_f   = base_hp["learning_rate_f"],
        keep_prob         = base_hp["keep_prob"],
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
    tf.reset_default_graph()

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = (hcl_config_idx, hcl_m, hcl_betas)

    result = {
        'loss': -l,
        'status': STATUS_OK,
        'space': {
            'hcl_config_idx': hcl_config_idx,
            'hcl_m': hcl_m,
            'hcl_betas': hcl_betas,
        },
    }
    save_json_result(str(l), result, path)
    return result


def run_a_trial(hcl_space, path, pair_key):
    """
    Runs one TPE trial for the HCL configuration search.

    :param hcl_space: hyperopt search space (categorical over HCL_CONFIGS indices)
    :param path:      save path for trial results
    :param pair_key:  domain pair string
    """
    max_evals = nb_evals = 1
    print("Attempting to resume past training if it exists:")
    try:
        trials = pickle.load(open(path + "results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except Exception:
        trials = Trials()
        print("Starting from scratch: new trials.")

    partial_objective = partial(hcl_objective, path=path, pair_key=pair_key)

    fmin(
        fn=partial_objective,
        space=hcl_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(path + "results.pkl", "wb"))
    print("OPTIMISATION STEP COMPLETE.\n")


def save_json_result(model_name, result, path):
    """Save json result to path."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, result_name), 'w') as f:
        json.dump(result, f, default=json_util.default,
                  sort_keys=True, indent=4, separators=(',', ': '))


def load_json_result(best_result_name, path):
    """Load json from path."""
    result_path = os.path.join(path, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(f.read())


def load_best_hyperspace(path):
    """Load the best hyperparameter configuration found so far."""
    results = [f for f in sorted(os.listdir(path)) if 'json' in f]
    if not results:
        return None
    return load_json_result(results[-1], path)["space"]


def plot_best_model(path):
    """Print the best HCL configuration found so far."""
    space_best = load_best_hyperspace(path)
    if space_best is None:
        print("No best model to plot yet. Continuing...")
        return
    print("Best HCL config so far:")
    print(json.dumps(space_best, indent=4))


if __name__ == "__main__":
    main()