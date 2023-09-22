import os
import random
import numpy as np
import pandas as pd
import igraph as ig
import rpy2
import torch
import lingam
from matplotlib import pyplot as plt, ticker
from sklearn import preprocessing
from lingam.utils import make_dot
from numpy.random import laplace, uniform, normal
from scipy.stats import kstest
import rpy2.robjects as robj


def normalize_data(X):
    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)
    assert (normalized_X.std(axis=0).round(decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

    return normalized_X


def set_random_seed(seed):
    random.seed(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # set pytorch seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs.


def set_random_seed_R(seed):
    r = rpy2.robjects.r
    set_seed = r('set.seed')
    set_seed(seed)


def get_GPU_if_available():
    """
    prefers CUDA over MPS.
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        # it looks like MPS is slower than CPU.
        device = 'mps'
    return device


def draw_DAGs_using_LINGAM(file_name, adjacency_matrix, variable_names):
    # direction of the adjacency matrix needs to be transposed.
    # in LINGAM, the adjacency matrix is defined as column variable -> row variable
    # in NOTEARS, the W is defined as row variable -> column variable

    # there are bugs about using prior knowledge in the LINGAM package version 1.5.1, which is fixed in the version 1.5.2.
    assert lingam.__version__ == '1.5.3', 'current LINGAM package version: ' + lingam.__version__

    # the default value here was 0.01. Instead of not drawing edges smaller than 0.01, we eliminate edges
    # smaller than `w_threshold` from the estimated graph so that we can set the value here to 0.
    lower_limit = 0.0

    dot = make_dot(np.transpose(adjacency_matrix), labels=variable_names, lower_limit=lower_limit)

    dot.format = 'png'
    dot.render(file_name)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true, B_est, allow_cycles=False):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive (i.e. among the predicted edges,
            the percentage that are incorrect. The smaller the better.)
        precision: 1 - fdr, but base on the computation of fdr below, computing precision with fdr may produce
            problem when fdr=0. So compute precision without using fdr: (true positive) / prediction positive,
            (i.e. among the predicted edges, the percentage that are correct. The bigger the better.)
        tpr (recall): (true positive) / condition positive (i.e. among the true edges,
            the percentage that are predicted. The bigger the better.)
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not allow_cycles:
            if not is_dag(B_est):
                raise ValueError('B_est should be a DAG')

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Refer to Linear Notears 2018, D.2. Metrics
    # fdr: False discovery rate, FDR = (R + FP) / P, the smaller the better
    # tpr: True positive rate, TPR = TP / T, the bigger the better
    # fpr: False positive rate, FPR = (R + FP) / F, the smaller the better
    # shd: Structural Hamming distance, SHD = E + M + R, the smaller the better, (total number of edge additions, deletions, and reversals needed to convert the estimated DAG into the true DAG)
    # nnz: number of predicted positives

    precision = float(len(true_pos)) / max(pred_size, 1)
    recall = tpr

    # https://en.wikipedia.org/wiki/F-score#Definition
    # The highest possible value of an F-score is 1.0, indicating perfect precision and recall,
    # and the lowest possible value is 0, if either the precision or the recall is zero.
    if precision == 0 or recall == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return {'fdr': fdr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'precision': precision, 'recall': recall,
            'f1_score': f1_score}


def append_to_file(target_file, content_to_append):
    f = open(target_file, "a")
    f.write(content_to_append)
    f.close()


def make_configuration_path(root_directory, test_case, sem_type, noise_type, n, epochs, optimizer, learning_rate,
                            L2_penalty, method, KLD_weight, KLD_threshold, activation, hidden_dimensions,
                            reconstruction_weight, lambda1, lambda2, lambda3, sparsity_weight,
                            weight_threshold_incremental_step, extra_kld_weight, tau_A, lambda_A,
                            c_A, use_A_connect_loss, use_A_positiver_loss, alpha, weight_threshold, pruning, score,
                            initial_A, n_layers, n_hidden, prior_dist, net_class, architecture, scale_base, shift_base,
                            scale, train_test_split, batch_size, scheduler, weight_decay, beta1, amsgrad, seed=None):
    configuration_path = '{}/TestCase_{}/SemType_{}/NoiseType_{}/N_{}/Epochs_{}/' \
                         'Optimizer_{}/LearningRate_{}/L2Penalty_{}/' \
                         'Method_{}/Initial_A_{}/Weight_Threshold_Incremental_Step_{}/Reconstruction_Weight_{}/KLD_Weight_{}/' \
                         'Extra_KLD_Weight_{}/KLD_Threshold_{}/Sparsity_Weight_{}/' \
                         'Activation_{}/HiddenDimensions_{}/Lambda1_{}/Lambda2_{}/Lambda3_{}/' \
                         'tau_A_{}/lambda_A_{}/c_A_{}/use_A_connect_loss_{}/use_A_positiver_loss_{}/Alpha_{}/' \
                         'Weight_Threshold_{}/Pruning_{}/Score_{}/' \
                         'n_layers_{}/n_hidden_{}/prior_dist_{}/net_class_{}/architecture_{}/scale_base_{}/' \
                         'shift_base_{}/scale_{}/train_test_split_{}/batch_size_{}/scheduler_{}/weight_decay_{}/' \
                         'beta1_{}/amsgrad_{}/{}'.format(
        root_directory, test_case, sem_type, noise_type, n, epochs, optimizer, learning_rate, L2_penalty, method,
        initial_A, weight_threshold_incremental_step, reconstruction_weight, KLD_weight, extra_kld_weight,
        KLD_threshold, sparsity_weight, activation, hidden_dimensions, lambda1, lambda2, lambda3, tau_A, lambda_A, c_A,
        use_A_connect_loss, use_A_positiver_loss, alpha, weight_threshold, pruning, score, n_layers, n_hidden,
        prior_dist, net_class, architecture, scale_base, shift_base, scale, train_test_split, batch_size, scheduler,
        weight_decay, beta1, amsgrad, "Seed_{}/".format(seed) if seed is not None else "")

    return configuration_path


def plot_datapoints(result_folder, title, cause, effect, cause_label, effect_label):
    plt.title(title)

    plt.xlabel(cause_label)
    plt.ylabel(effect_label)

    plt.plot(cause, effect, 'o', markeredgecolor='black', markeredgewidth=2)

    # x and y axes have the same scale
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # plt.show()
    plt.savefig(os.path.join(result_folder, title))

    plt.close('all')  # close all plots, otherwise it would consume memory


def plot_data_distributions(result_folder, figure_title, variable1, v1_subfigure_title, variable2, v2_subfigure_title):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    fig.tight_layout(pad=3.0)  # add space between subplots
    fig.suptitle(figure_title)

    ax1.set_title(v1_subfigure_title)
    feature = variable1.reshape(-1, 1)
    ax1.hist(feature, bins=100)
    # ax1.set(yticklabels=[])  # remove the tick labels
    ax1.tick_params(left=False)  # remove the ticks
    ax1.set_ylabel("Counts")

    ax2.set_title(v2_subfigure_title)
    feature = variable2.reshape(-1, 1)
    ax2.hist(feature, bins=100)
    # ax2.set(yticklabels=[])  # remove the tick labels
    ax2.tick_params(left=False)  # remove the ticks
    ax2.set_ylabel("Counts")

    # plt.show()
    plt.savefig(os.path.join(result_folder, figure_title))

    plt.close('all')  # close all plots, otherwise it would consume memory


def plot_loss(loss_values, savefig, result_folder_to_save, figure_title, x_axis_label, y_axis_label,
              y_limit: list = None):
    plt.plot(loss_values)

    plt.title(figure_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    if y_limit is not None:
        plt.ylim(y_limit[0], y_limit[1])

    if savefig:
        plt.savefig(os.path.join(result_folder_to_save, figure_title))
    else:
        plt.show()

    plt.close('all')  # close all plots, otherwise it would consume memory


def compute_covariance(data, D):
    #############################################################
    # https://en.wikipedia.org/wiki/Correlation#Uncorrelatedness_and_independence_of_stochastic_processes
    # if Z's (no matter what distribution they are) are independent, they must have 0 covariance.
    # equivalently, if Z's have non-zero covariance, they must be dependent.

    # https://en.wikipedia.org/wiki/Normally_distributed_and_uncorrelated_does_not_imply_independent
    # for two random variables that are normally distributed, uncorrelatedness does not in general imply
    # independence, unless they are multivariate normal distribution.
    #############################################################

    if D != 2:
        raise Exception("current implementation supports only D=2. Currently, D={}".format(D))

    if data.T.shape[0] != D:
        raise Exception("Data passed to np.cov() must be in the shape of (D, n). Currently, data.T.shape={}"
                        .format(data.T.shape))

    # https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    covariance_matrix = np.cov(data.T)
    if covariance_matrix.shape[0] != D or covariance_matrix.shape[1] != D:
        raise Exception("what??? covariance_matrix.shape={}".format(covariance_matrix.shape))

    if covariance_matrix[0][1] != covariance_matrix[1][0]:
        raise Exception("what??? covariance_matrix={}".format(covariance_matrix))

    covariance = covariance_matrix[0][1]
    return covariance


def goodness_of_fit_KS_test(n, data, alpha=0.05):
    # use Kolmogorov-Smirnov Test for continuous data to test which distribution the data is most likely to follow.

    gaussian_result = kstest(normalize_data(data.reshape(-1, 1)).reshape(-1).tolist(),
                             normalize_data(normal(size=n).reshape(-1, 1)).reshape(-1).tolist())
    # print("p-value from gaussian: {}".format(gaussian_result.pvalue))

    laplace_result = kstest(normalize_data(data.reshape(-1, 1)).reshape(-1).tolist(),
                            normalize_data(laplace(size=n).reshape(-1, 1)).reshape(-1).tolist())
    # print("p-value from laplace: {}".format(laplace_result.pvalue))

    uniform_result = kstest(normalize_data(data.reshape(-1, 1)).reshape(-1).tolist(),
                            normalize_data(uniform(size=n).reshape(-1, 1)).reshape(-1).tolist())
    # print("p-value from uniform: {}".format(uniform_result.pvalue))

    distributions_not_reject = []
    if gaussian_result.pvalue > alpha:
        distributions_not_reject.append("gaussian")

    if laplace_result.pvalue > alpha:
        distributions_not_reject.append("laplace")

    if uniform_result.pvalue > alpha:
        distributions_not_reject.append("uniform")

    return distributions_not_reject


# Using R inside python
# https://stackoverflow.com/questions/55797564/how-to-import-r-packages-in-python
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.vectors import FloatVector


def call_dHSIC_from_R(dHSIC, x_python, y_python, kernels_python):
    '''
    Call R package dHSIC to detect dependency, using method dhsic().

    https://cran.r-project.org/web/packages/dHSIC/dHSIC.pdf
    '''

    # convert python variables to R variables
    x = FloatVector(x_python)
    y = FloatVector(y_python)
    kernels = StrVector(kernels_python)

    # call the R function, originally in R: ```dhsic(list(x,y),kernel=c("gaussian","gaussian"))```
    results = dHSIC.dhsic([x, y], kernel=kernels)

    return results


def call_dHSIC_test_from_R(dHSIC, x_python, y_python, dHSIC_test_method, kernels_python, result_file=None, alpha=0.05):
    '''
    Call R package dHSIC to detect dependency, using method dhsic.test().

    alpha=0.05 is the default used in the R package.

    https://cran.r-project.org/web/packages/dHSIC/dHSIC.pdf
    '''

    # convert python variables to R variables
    x = FloatVector(x_python)
    y = FloatVector(y_python)
    kernels = StrVector(kernels_python)

    # call the R function, originally in R: ```dhsic(list(x,y),method = "permutation", kernel=c("gaussian","gaussian"))```
    # https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740
    results_R = dHSIC.dhsic_test([x, y], alpha=alpha, method=dHSIC_test_method, kernel=kernels)

    p_value = results_R.rx2("p.value")[0]
    print("p_value: {}".format(p_value))
    if result_file is not None:
        append_to_file(result_file, "p_value: {}\n".format(p_value))

    test_statistic = results_R.rx2("statistic")[0]
    critical_value = results_R.rx2("crit.value")[0]
    print("test_statistic: {}".format(test_statistic))
    print("critical_value: {}".format(critical_value))
    if result_file is not None:
        append_to_file(result_file, "test_statistic: {}\n".format(test_statistic))
        append_to_file(result_file, "critical_value: {}\n".format(critical_value))

    # The null hypothesis (H_0: joint independence) is rejected if statistic is strictly greater than crit.value.
    if test_statistic > critical_value:
        print("test_statistic > critical_value")
        independence_conclusion = "dependent"
    else:
        print("test_statistic <= critical_value")
        independence_conclusion = "independent"

    return independence_conclusion


def convert_numpy_to_r(normalized_X):
    # convert numpy array to R matrix
    # https://stackoverflow.com/a/10037302
    nr, nc = normalized_X.shape
    xvec = robj.FloatVector(normalized_X.transpose().reshape((normalized_X.size)))
    normalized_X_R = robj.r.matrix(xvec, nrow=nr, ncol=nc)

    return normalized_X_R


def read_accuracy_results(data_type, default_result_file, result_file, split, x_label):
    dfault_df = pd.read_csv(default_result_file)
    dfault_df_08 = dfault_df[dfault_df[' train_test_split'].str.contains(split)]

    if dfault_df_08.empty == True:
        raise RuntimeError('table is empty???')

    print("columns: \n", dfault_df.columns)

    specific_df = pd.read_csv(result_file)
    specific_df_08 = specific_df[specific_df[' train_test_split'].str.contains(split)]

    if specific_df_08.empty == True:
        raise RuntimeError('table is empty???')

    print("default results: \n", dfault_df_08.to_string())
    print("specific results: \n", specific_df_08.to_string())

    total_df = pd.concat([dfault_df_08, specific_df_08])

    # cast column values from str to numbers
    if "prior" not in x_label:
        total_df[x_label] = pd.to_numeric(total_df[x_label])

    if data_type == "Tubingen_CEpairs_weighted":
        total_df[" Tubingen_CEpairs_CAREFL_LR_weighted_accuracy"] = pd.to_numeric(
            total_df[" Tubingen_CEpairs_CAREFL_LR_weighted_accuracy"])
        total_df[" Tubingen_CEpairs_CAREFL_IT_ZZ_weighted_accuracy"] = pd.to_numeric(
            total_df[" Tubingen_CEpairs_CAREFL_IT_ZZ_weighted_accuracy"])
        total_df[" Tubingen_CEpairs_CAREFL_IT_XZ_weighted_accuracy "] = pd.to_numeric(
            total_df[" Tubingen_CEpairs_CAREFL_IT_XZ_weighted_accuracy "])
    else:
        total_df[" CAREFL-LR accuracy"] = pd.to_numeric(total_df[" CAREFL-LR accuracy"])
        total_df[" CAREFL-IT-ZZ accuracy"] = pd.to_numeric(total_df[" CAREFL-IT-ZZ accuracy"])
        total_df[" CAREFL-IT-XZ accuracy"] = pd.to_numeric(total_df[" CAREFL-IT-XZ accuracy"])

    # sort the table by X-tick values
    if "prior" not in x_label:
        total_df = total_df.sort_values(by=[x_label])

    # for X ticks, cast column values from numbers to str
    if "prior" not in x_label:
        total_df[x_label] = total_df[x_label].astype(str)
    x_ticks = total_df[x_label].tolist()
    print("x_ticks: ", x_ticks)

    # read CAREFL-M accuracy
    if data_type == "Tubingen_CEpairs_weighted":
        y_values_M = total_df[" Tubingen_CEpairs_CAREFL_LR_weighted_accuracy"].tolist()
    else:
        y_values_M = total_df[" CAREFL-LR accuracy"].tolist()
    print("y_values_M: ", y_values_M)

    # read CAREFL-H accuracy
    y_values_H = []
    for index, row in total_df.iterrows():

        if data_type == "Tubingen_CEpairs_weighted":
            row_ZZ = row[' Tubingen_CEpairs_CAREFL_IT_ZZ_weighted_accuracy']
            row_XZ = row[' Tubingen_CEpairs_CAREFL_IT_XZ_weighted_accuracy ']
        else:
            row_ZZ = row[' CAREFL-IT-ZZ accuracy']
            row_XZ = row[' CAREFL-IT-XZ accuracy']

        if row_XZ >= row_ZZ:
            y_values_H.append(row_XZ)
        else:
            y_values_H.append(row_ZZ)
    print("y_values_H: ", y_values_H)

    return x_ticks, y_values_M, y_values_H


def make_figures(data_type, root_directory, figure_name, y_range):
    default_file = os.path.join(root_directory, "_default/file_all_results_default.csv")
    nh_file = os.path.join(root_directory, "_nhs/file_all_results_nhs.csv")
    nl_file = os.path.join(root_directory, "_nls/file_all_results_nls.csv")
    epoch_file = os.path.join(root_directory, "_epochs/file_all_results_epochs.csv")
    prior_file = os.path.join(root_directory, "_priors/file_all_results_priors.csv")
    weight_decay_file = os.path.join(root_directory, "_weight_decays/file_all_results_weight_decays.csv")

    x_labels_08_nh, y_values_M_08_nh, y_values_H_08_nh = read_accuracy_results(
        data_type, default_file, nh_file, "0.8", " carefl_nh")
    x_labels_08_nl, y_values_M_08_nl, y_values_H_08_nl = read_accuracy_results(
        data_type, default_file, nl_file, "0.8", " carefl_nl")
    x_labels_08_epoch, y_values_M_08_epoch, y_values_H_08_epoch = read_accuracy_results(
        data_type, default_file, epoch_file, "0.8", " carefl_epochs")
    _, y_values_M_08_prior, y_values_H_08_prior = read_accuracy_results(
        data_type, default_file, prior_file, "0.8", " carefl_prior_dist")
    x_labels_08_weight_decay, y_values_M_08_weight_decay, y_values_H_08_weight_decay = read_accuracy_results(
        data_type, default_file, weight_decay_file, "0.8", " carefl_weight_decay")

    x_labels_1_nh, y_values_M_1_nh, y_values_H_1_nh = read_accuracy_results(
        data_type, default_file, nh_file, "1", " carefl_nh")
    x_labels_1_nl, y_values_M_1_nl, y_values_H_1_nl = read_accuracy_results(
        data_type, default_file, nl_file, "1", " carefl_nl")
    x_labels_1_epoch, y_values_M_1_epoch, y_values_H_1_epoch = read_accuracy_results(
        data_type, default_file, epoch_file, "1", " carefl_epochs")
    x_labels_1_prior, y_values_M_1_prior, y_values_H_1_prior = read_accuracy_results(
        data_type, default_file, prior_file, "1", " carefl_prior_dist")
    x_labels_1_weight_decay, y_values_M_1_weight_decay, y_values_H_1_weight_decay = read_accuracy_results(
        data_type, default_file, weight_decay_file, "1", " carefl_weight_decay")

    fig, axs = plt.subplots(1, 5, figsize=(11, 2))

    # line figures
    markers = ["o", "^", "s", "x"]
    axs[0].plot(x_labels_08_nh, y_values_M_08_nh, label="CAREFL-M (0.8)", marker=markers[0])
    axs[0].plot(x_labels_08_nh, y_values_H_08_nh, label="CAREFL-H (0.8)", marker=markers[1])
    axs[0].plot(x_labels_1_nh, y_values_M_1_nh, label="CAREFL-M (1.0)", marker=markers[2])
    axs[0].plot(x_labels_1_nh, y_values_H_1_nh, label="CAREFL-H (1.0)", marker=markers[3])
    axs[0].set(xlabel="Number of Hidden Neurons", ylabel="Accuracy")
    axs[0].set_ylim(y_range)

    axs[1].plot(x_labels_08_nl, y_values_M_08_nl, label="CAREFL-M (0.8)", marker=markers[0])
    axs[1].plot(x_labels_08_nl, y_values_H_08_nl, label="CAREFL-H (0.8)", marker=markers[1])
    axs[1].plot(x_labels_1_nl, y_values_M_1_nl, label="CAREFL-M (1.0)", marker=markers[2])
    axs[1].plot(x_labels_1_nl, y_values_H_1_nl, label="CAREFL-H (1.0)", marker=markers[3])
    axs[1].set(xlabel="Number of Sub-Flows")
    axs[1].set_ylim(y_range)

    axs[2].plot(x_labels_08_epoch, y_values_M_08_epoch, label="CAREFL-M (0.8)", marker=markers[0])
    axs[2].plot(x_labels_08_epoch, y_values_H_08_epoch, label="CAREFL-H (0.8)", marker=markers[1])
    axs[2].plot(x_labels_1_epoch, y_values_M_1_epoch, label="CAREFL-M (1.0)", marker=markers[2])
    axs[2].plot(x_labels_1_epoch, y_values_H_1_epoch, label="CAREFL-H (1.0)", marker=markers[3])
    axs[2].set(xlabel="Number of Epochs")
    axs[2].set_ylim(y_range)

    axs[3].plot(x_labels_08_weight_decay, y_values_M_08_weight_decay, label="CAREFL-M (0.8)", marker=markers[0])
    axs[3].plot(x_labels_08_weight_decay, y_values_H_08_weight_decay, label="CAREFL-H (0.8)", marker=markers[1])
    axs[3].plot(x_labels_1_weight_decay, y_values_M_1_weight_decay, label="CAREFL-M (1.0)", marker=markers[2])
    axs[3].plot(x_labels_1_weight_decay, y_values_H_1_weight_decay, label="CAREFL-H (1.0)", marker=markers[3])
    axs[3].set(xlabel="L2-Penalty")
    axs[3].set_ylim(y_range)

    # bar figure
    patterns = ["x", "o", "/", "*"]
    X_axis = np.arange(len(x_labels_1_prior))
    axs[4].bar(X_axis - 0.3, y_values_M_08_prior, label="CAREFL-M (0.8)", edgecolor='black', width=0.2,
               hatch=patterns[0])
    axs[4].bar(X_axis - 0.1, y_values_H_08_prior, label="CAREFL-H (0.8)", edgecolor='black', width=0.2,
               hatch=patterns[1])
    axs[4].bar(X_axis + 0.1, y_values_M_1_prior, label="CAREFL-M (1.0)", edgecolor='black', width=0.2,
               hatch=patterns[2])
    axs[4].bar(X_axis + 0.3, y_values_H_1_prior, label="CAREFL-H (1.0)", edgecolor='black', width=0.2,
               hatch=patterns[3])
    axs[4].set_xticks(X_axis)
    axs[4].set_xticklabels(x_labels_1_prior)
    axs[4].set(xlabel="Priors")
    axs[4].set_ylim(y_range)

    # To show legends, uncomment the following line, then comment out the 2nd and 3rd sub-plots, increase figure size
    # fig.legend(loc="lower center", ncol=4, prop={'size': 20}, framealpha=1.0)

    fig.tight_layout()

    # plt.show()
    png_file = os.path.join(root_directory, figure_name + '.png')
    plt.savefig(png_file)
    plt.close('all')  # close all plots, otherwise it would consume memory


def make_all_figures_in_experiments():
    ########################################################################
    # create figures for simulated data
    ########################################################################

    SEMs = ["LSNM-tanh-exp-cosine", "LSNM-sine-tanh", "LSNM-sigmoid-sigmoid"]
    noises = ["uniform", "beta0505", "continuous-bernoulli", "exp", "standard-gaussian", "laplace"]
    Ns = [500, 5000]

    for sem in SEMs:
        for noise in noises:
            for n in Ns:
                root_directory = "../CAREFL_H/temp_results/results_one_hyperparameter/simulated/" + sem + "/" + noise \
                                 + "/" + str(n)
                figure_name = sem + "_" + noise + "_" + str(n)
                make_figures("simulated", root_directory, figure_name, [0.0, 1.1])

    ########################################################################
    # create figures for SIM benchmarks
    ########################################################################

    dataset_types = ["SIM_benchmark_SIM", "SIM_benchmark_SIM-c", "SIM_benchmark_SIM-G", "SIM_benchmark_SIM-ln"]

    for sim_type in dataset_types:
        root_directory = "../CAREFL_H/temp_results/results_one_hyperparameter/SIM/" + sim_type
        make_figures("SIMs", root_directory, sim_type, [0.0, 1.1])

    ########################################################################
    # create figures for Tubingen_CEpairs
    ########################################################################

    root_directory = "../CAREFL_H/temp_results/results_one_hyperparameter/Tubingen_CEpairs"

    figure_name = "unweighted"
    make_figures("Tubingen_CEpairs_unweighted", root_directory, figure_name, [0.4, 0.9])

    figure_name = "weighted"
    make_figures("Tubingen_CEpairs_weighted", root_directory, figure_name, [0.4, 0.9])


def make_likelihood_vs_mcv_plot(SEM, noise, key_str, label):
    """
    plot: causal_MCV - anti_causal_MCV vs causal_likelihood - anti_causal_likelihood
    """

    minus_math_symbol = u"\u2212"  # looks more like munis than "-"
    divide_math_symbol = "/"  # looks more like munis than "-"

    causal_vs_anti_criterion_differences = []
    causal_vs_anti_mcv_quotients = []

    root_directory_path = "../CAREFL_H/temp_results/simulated/vary_conditional_variance/CAREFL/split_1.0/prior_gaussian/" \
                          + SEM + "/" + noise + "/epoch_750/_/Ground-truth_SEM_" + SEM + "/Noise_" + noise

    for g_magnitude_name in os.listdir(root_directory_path):
        print("g_magnitude_name: ", g_magnitude_name)
        g_magnitude_path = os.path.join(root_directory_path, g_magnitude_name)

        if os.path.isdir(g_magnitude_path):  # if it is a folder

            seed_parent_path = os.path.join(g_magnitude_path,
                                            "n_10000/fix_AffineCL_forward_1/nl_4_nh_5/prior_dist_gaussian/"
                                            "batch_size_-1/weight_decay_0.0/epochs_750/split_1.0")
            for seed_name in os.listdir(seed_parent_path):

                seed_path = os.path.join(seed_parent_path, seed_name)

                if os.path.isdir(seed_path):
                    print("seed_name: ", seed_name)

                    ######################################################################
                    # go through True_Conditional_Variances.txt line by line
                    ######################################################################

                    result_file_path = os.path.join(seed_path, "True_Conditional_Variances.txt")

                    f = open(result_file_path, "r")
                    file_lines = f.readlines()

                    mcv_causal = None
                    mcv_anti_causal = None
                    for i, line in enumerate(file_lines):
                        if "mean_var_causal_direction" in line:
                            mcv_causal = float(line.split(":")[-1])
                        elif "mean_var_anti_causal_direction" in line:
                            mcv_anti_causal = float(line.split(":")[-1])

                    causal_vs_anti_mcv_quotients.append(mcv_causal / mcv_anti_causal)

                    ######################################################################
                    # go through result.txt line by line
                    ######################################################################

                    result_file_path = os.path.join(seed_path, "result.txt")

                    f = open(result_file_path, "r")
                    file_lines = f.readlines()
                    causal_log_criterion = None
                    anti_causal_log_criterion = None
                    for i, line in enumerate(file_lines):
                        # print(line)

                        if "B_true:" in line:  # read next two lines which gives the true direction:
                            B_true_str = file_lines[i + 1].rstrip() + file_lines[i + 2].rstrip()
                            print("B_true_str:", B_true_str)

                            if "[[0. 1.] [0. 0.]]" in B_true_str:
                                causal_direction = "X1->X2"
                            elif "[[0. 0.] [1. 0.]]" in B_true_str:
                                causal_direction = "X2->X1"
                            else:
                                raise Exception("what??? B_true_str={}".format(B_true_str))

                            print("causal_direction: ", causal_direction)

                        elif "Fitting CAREFL in the direction" in line:
                            if "X1->X2" in line:
                                estimating_drecton = "X1->X2"
                            elif "X2->X1" in line:
                                estimating_drecton = "X2->X1"
                            else:
                                raise Exception("what??? line={}".format(line))

                        elif key_str in line:
                            criterion_value = float(line.split(":")[-1])
                            print(criterion_value)

                            if causal_direction not in ["X1->X2", "X2->X1"]:
                                raise Exception("what??? causal_direction={}".format(causal_direction))

                            if estimating_drecton not in ["X1->X2", "X2->X1"]:
                                raise Exception("what??? estimating_drecton={}".format(estimating_drecton))

                            if causal_direction == estimating_drecton:
                                causal_log_criterion = criterion_value
                            else:
                                anti_causal_log_criterion = criterion_value

                    print("causal {}: {}".format(label, causal_log_criterion))
                    print("anti_causal {}: {}".format(label, anti_causal_log_criterion))

                    # use difference instead of ratio, because the values are negative
                    causal_vs_anti_criterion_differences.append(causal_log_criterion - anti_causal_log_criterion)

    # print(len(causal_vs_anti_criterion_differences))
    # print(len(causal_vs_anti_mcv_quotients))

    # MCV vs Likelihood
    plt.plot(causal_vs_anti_mcv_quotients, causal_vs_anti_criterion_differences, 'o')
    plt.xlabel("Mean CV: Causal " + divide_math_symbol + " Anti-Causal", fontsize=20)
    plt.ylabel(label.replace(" ", "-") + ": Causal " + minus_math_symbol + " Anti-Causal", fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axhline(y=0.0, color='r', linestyle='--')  # a horizontal line at y=0.0

    plt.tight_layout()
    png_file = os.path.join(root_directory_path,
                            'mcv_vs_{}_'.format(label.replace(" ", "_")) + SEM + '_' + noise + '.png')
    plt.savefig(png_file)
    plt.close('all')  # close all plots, otherwise it would consume memory


def extract_R_versions(package_data):
    return dict(zip(
        package_data.rx(True, 'Package'),  # get Package column
        package_data.rx(True, 'Version')  # get Version column
    ))


def make_MisleadingCVs_vs_Accuracy_plot(SEM, noise, file_root_path, x_label, y_label, x_key_str, y_key_str_dic,
                                        marker_style_dic, marker_size_dic, marker_color_dic, methods):
    """
    plot: Percentage Datasets With Misleading CVs vs. Accuracy
    """

    for method in methods:
        y_key_str = y_key_str_dic[method]
        marker_style = marker_style_dic[method]
        marker_size = marker_size_dic[method]
        marker_color = marker_color_dic[method]

        full_file_name = os.path.join(file_root_path, "{}/file_all_results_{}.csv".format(method, SEM))

        dfault_df = pd.read_csv(full_file_name)
        dfault_df_noise = dfault_df[dfault_df[' noise'].str.contains(noise)]

        x_values = pd.to_numeric(dfault_df_noise[x_key_str]).tolist()
        y_values = pd.to_numeric(dfault_df_noise[y_key_str]).tolist()

        # plot
        # for example, change 1 to 10%
        plt.plot([i * 10 for i in x_values], y_values, marker_style, markersize=marker_size, color=marker_color)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.1, 1.1)
    # make x-axis show integer values only
    # https://stackoverflow.com/questions/52229875/how-to-force-matplotlib-to-show-values-on-x-axis-as-integers
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # every 10 integers

    lgnd = plt.legend(methods, fontsize=10)

    # make the marker the same size on the legend
    for legend_handle in lgnd.legendHandles:
        legend_handle.set_markersize(10)

    plt.tight_layout()
    png_file = os.path.join("/Users/xiangyus/Downloads", 'misleading_CVs_vs_accuracy_{}_{}.png'.format(SEM, noise))
    plt.savefig(png_file)
    plt.close('all')  # close all plots, otherwise it would consume memory


def make_ANM_plots(folder_name, methods, y_key_str_dic, marker_style_dic, marker_size_dic, marker_color_dic, SEMs):
    file_root_path = "/Users/xiangyus/Development/My_Useful_Utilities/CAREFL_H/saved_results/" + folder_name

    x_label = "Percentage of Datasets With Misleading CVs (%)"
    y_label = "Accuracy"
    x_key_str = " number_causal_bigger_CV"

    for SEM in SEMs:
        noise = "uniform"
        make_MisleadingCVs_vs_Accuracy_plot(SEM, noise, file_root_path, x_label, y_label, x_key_str, y_key_str_dic,
                                            marker_style_dic, marker_size_dic, marker_color_dic, methods)

        noise = "standard-gaussian"
        make_MisleadingCVs_vs_Accuracy_plot(SEM, noise, file_root_path, x_label, y_label, x_key_str, y_key_str_dic,
                                            marker_style_dic, marker_size_dic, marker_color_dic, methods)


def make_all_MisleadingCVs_vs_Accuracy_plots():
    ##############################
    # ANMs
    ##############################

    folder_name = "ANM_data"

    methods = ["CAREFL-ANM-M", "CAREFL-ANM-Sigma-M", "CAM"]

    SEMs = ["ANM-sine", "ANM-tanh", "ANM-sigmoid"]

    marker_size_dic = {
        "CAREFL-ANM-M": 10,
        "CAREFL-ANM-Sigma-M": 20,
        "CAM": 20
    }

    # https://matplotlib.org/stable/api/markers_api.html
    marker_style_dic = {
        "CAREFL-ANM-M": "o",
        "CAREFL-ANM-Sigma-M": "+",
        "CAM": "x"
    }

    marker_color_dic = {
        "CAREFL-ANM-M": "blue",
        "CAREFL-ANM-Sigma-M": "green",
        "CAM": "orange"
    }

    y_key_str_dic = {
        "CAREFL-ANM-M": " CAREFL-LR accuracy",
        "CAREFL-ANM-Sigma-M": " CAREFL-LR accuracy",
        "CAM": " CAM accuracy"
    }

    make_ANM_plots(folder_name, methods, y_key_str_dic, marker_style_dic, marker_size_dic, marker_color_dic, SEMs)

    ##############################
    # LSNMs
    ##############################

    folder_name = "LSNM_data"

    methods = ["CAREFL-M", "LOCI-M"]

    SEMs = ["LSNM-tanh-exp-cosine", "LSNM-sine-tanh", "LSNM-sigmoid-sigmoid"]

    # size decreases in the order of `methods`, so that the later ones do not cover the earlier ones.
    marker_size_dic = {
        "CAREFL-M": 20,
        "LOCI-M": 10
    }

    marker_style_dic = {
        "CAREFL-M": "x",
        "LOCI-M": "o"
    }

    marker_color_dic = {
        "CAREFL-M": "blue",
        "LOCI-M": "orange"
    }

    y_key_str_dic = {
        "CAREFL-M": " CAREFL-LR accuracy",
        "LOCI-M": " LOCI_NN accuracy"
    }

    make_ANM_plots(folder_name, methods, y_key_str_dic, marker_style_dic, marker_size_dic, marker_color_dic, SEMs)


if __name__ == "__main__":
    make_all_figures_in_experiments()

    # teaser plots
    SEM = "LSNM-sine-tanh"
    noise = "uniform"
    key_str = "Testing log p(x1, x2)"
    label = "Log Likelihood"
    make_likelihood_vs_mcv_plot(SEM, noise, key_str, label)

    SEM = "LSNM-sine-tanh"
    noise = "standard-gaussian"
    key_str = "Testing log p(x1, x2)"
    label = "Log Likelihood"
    make_likelihood_vs_mcv_plot(SEM, noise, key_str, label)

    SEM = "LSNM-sine-tanh"
    noise = "uniform"
    key_str = "estimated dHSIC value between X and reconstructed Z"
    label = "HSIC Value"
    make_likelihood_vs_mcv_plot(SEM, noise, key_str, label)

    SEM = "LSNM-sine-tanh"
    noise = "standard-gaussian"
    key_str = "estimated dHSIC value between X and reconstructed Z"
    label = "HSIC Value"
    make_likelihood_vs_mcv_plot(SEM, noise, key_str, label)

    make_all_MisleadingCVs_vs_Accuracy_plots()
