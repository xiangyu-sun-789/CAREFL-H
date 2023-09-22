import argparse
import math
import os
import random
import sys
import networkx as nx
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import SplineTransformer

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../algorithms/carefl"))
sys.path.append(os.path.abspath("../algorithms/loci"))

from Utilities.data_generator import simulate_dag, simulate_nonlinear_sem, SEM_Functionals, SEM_Noises, \
    read_Tubingen_data_file, is_bivariate_Tubingen_dataset
from Utilities.util_functions import set_random_seed, draw_DAGs_using_LINGAM, normalize_data, plot_data_distributions, \
    append_to_file, call_dHSIC_from_R, plot_datapoints, set_random_seed_R, extract_R_versions, convert_numpy_to_r
from algorithms.loci.run_individual import run_LinReg_Convex_LSNM, run_NN_LSNM, infer_direction_results

# Using R inside python
# https://stackoverflow.com/questions/55797564/how-to-import-r-packages-in-python
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import r

from algorithms.carefl.models import CAREFL


def run_carefl_in_one_direction(direction, normalized_X, result_folder, recons_error_figure_title, result_file,
                                B_true, updated_batch_size, split, fix_AffineCL_forward, dHSIC, dHSIC_kernels,
                                result_folder_method, carefl_nl, carefl_nh, carefl_prior_dist, carefl_epochs,
                                device, carefl_weight_decay, flow_SEM_type):
    ############################################
    # train CAREFL to fit the data
    ############################################

    n = normalized_X.shape[0]

    carefl_model = create_carefl_model(split, True if fix_AffineCL_forward else False, updated_batch_size,
                                       result_folder_method, carefl_nl, carefl_nh, carefl_prior_dist, carefl_epochs,
                                       device, carefl_weight_decay, flow_SEM_type)

    train_set, _, test_dset_numpy, dim, _ = carefl_model._get_datasets(normalized_X)
    carefl_model.dim = dim

    torch.manual_seed(carefl_model.config.training.seed)

    # fit the model in the provided direction
    if direction == "X1->X2":
        # Conditional Flow Model: X->Y
        trained_flows, total_losses, prior0_logprobs, prior1_logprobs, log_det0s, log_det1s = carefl_model._train(
            train_set)

    elif direction == "X2->X1":
        # Conditional Flow Model: Y->X
        trained_flows, total_losses, prior0_logprobs, prior1_logprobs, log_det0s, log_det1s = carefl_model._train(
            train_set, parity=True)
    else:
        raise Exception("What??? direction={}".format(direction))

    if len(trained_flows) != 1:
        raise Exception("What??? len(trained_flows)={}".format(len(trained_flows)))

    if len(total_losses) != 1:
        raise Exception("What??? len(total_losses)={}".format(len(total_losses)))

    # save training loss
    training_total_loss = total_losses[0]
    training_prior0_logprob = prior0_logprobs[0]
    training_prior1_logprob = prior1_logprobs[0]
    training_log_det0 = log_det0s[0]
    training_log_det1 = log_det1s[0]
    training_total_loss_log = os.path.join(result_folder, 'carefl_training_total_loss_log.txt')
    training_prior0_logprob_log = os.path.join(result_folder, 'carefl_training_prior0_logprob_log.txt')
    training_prior1_logprob_log = os.path.join(result_folder, 'carefl_training_prior1_logprob_log.txt')
    training_log_det0_log = os.path.join(result_folder, 'carefl_training_log_det0_log.txt')
    training_log_det1_log = os.path.join(result_folder, 'carefl_training_log_det1_log.txt')
    # to be consistent with notears, save the average loss, not summed loss
    np.savetxt(training_total_loss_log, np.array(training_total_loss) / n, delimiter=',', fmt='%f')
    np.savetxt(training_prior0_logprob_log, np.array(training_prior0_logprob) / n, delimiter=',', fmt='%f')
    np.savetxt(training_prior1_logprob_log, np.array(training_prior1_logprob) / n, delimiter=',', fmt='%f')
    np.savetxt(training_log_det0_log, np.array(training_log_det0) / n, delimiter=',', fmt='%f')
    np.savetxt(training_log_det1_log, np.array(training_log_det1) / n, delimiter=',', fmt='%f')
    append_to_file(result_file, "Training -log p(x1, x2) [-5:] / n: {}\n"
                   .format(np.array(training_total_loss[-5:]) / n))
    append_to_file(result_file, "Training log p(z1) [-5:] / n: {}\n"
                   .format(np.array(training_prior0_logprob[-5:]) / n))
    append_to_file(result_file, "Training log p(z2) [-5:] / n: {}\n"
                   .format(np.array(training_prior1_logprob[-5:]) / n))
    append_to_file(result_file, "Training log |Jacob det 1| [-5:] / n: {}\n"
                   .format(np.array(training_log_det0[-5:]) / n))
    append_to_file(result_file, "Training log |Jacob det 2| [-5:] / n: {}\n"
                   .format(np.array(training_log_det1[-5:]) / n))
    print("Training -log p(x1, x2) [-5:] / n: {}".format(np.array(training_total_loss[-5:]) / n))
    print("Training log p(z1) [-5:] / n: {}".format(np.array(training_prior0_logprob[-5:]) / n))
    print("Training log p(z2) [-5:] / n: {}".format(np.array(training_prior1_logprob[-5:]) / n))
    print("Training log |Jacob det 1| [-5:] / n: {}".format(np.array(training_log_det0[-5:]) / n))
    print("Training log |Jacob det 2| [-5:] / n: {}".format(np.array(training_log_det1[-5:]) / n))

    carefl_model.flow = trained_flows[0]

    # save testing log-likelihoods
    if direction == "X1->X2":
        _, test_log_likelihood_mean, _, _, test_log_p_x1_mean, test_log_p_x2_mean, test_log_det1, test_log_det2, \
            test_log_p_z0_mean, test_log_p_z1_mean = carefl_model._evaluate([carefl_model.flow], test_dset_numpy)
    elif direction == "X2->X1":
        _, test_log_likelihood_mean, _, _, test_log_p_x1_mean, test_log_p_x2_mean, test_log_det1, test_log_det2, \
            test_log_p_z0_mean, test_log_p_z1_mean = carefl_model._evaluate([carefl_model.flow], test_dset_numpy,
                                                                            parity=True)
    else:
        raise Exception("What??? direction={}".format(direction))

    test_log_likelihood_log = os.path.join(result_folder, 'carefl_test_log_likelihood_log.txt')
    np.savetxt(test_log_likelihood_log, [test_log_likelihood_mean], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log p(x1, x2): {}\n".format(test_log_likelihood_mean))
    print("Testing log p(x1, x2): {}".format(test_log_likelihood_mean))

    test_log_p_x1_log = os.path.join(result_folder, 'carefl_test_log_p_x1_log.txt')
    np.savetxt(test_log_p_x1_log, [test_log_p_x1_mean], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log p(x1) or log p(x1|x2): {}\n".format(test_log_p_x1_mean))
    print("Testing log p(x1) or log p(x1|x2): {}".format(test_log_p_x1_mean))

    test_log_p_x2_log = os.path.join(result_folder, 'carefl_test_log_p_x2_log.txt')
    np.savetxt(test_log_p_x2_log, [test_log_p_x2_mean], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log p(x2) or log p(x2|x1): {}\n".format(test_log_p_x2_mean))
    print("Testing log p(x2) or log p(x2|x1): {}".format(test_log_p_x2_mean))

    test_log_p_z0_log = os.path.join(result_folder, 'carefl_test_log_p_z0_log.txt')
    np.savetxt(test_log_p_z0_log, [test_log_p_z0_mean], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log p(z1): {}\n".format(test_log_p_z0_mean))
    print("Testing log p(z1): {}".format(test_log_p_z0_mean))

    test_log_p_z1_log = os.path.join(result_folder, 'carefl_test_log_p_z1_log.txt')
    np.savetxt(test_log_p_z1_log, [test_log_p_z1_mean], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log p(z2): {}\n".format(test_log_p_z1_mean))
    print("Testing log p(z2): {}".format(test_log_p_z1_mean))

    test_log_det1_log = os.path.join(result_folder, 'carefl_test_log_det1_log.txt')
    np.savetxt(test_log_det1_log, [test_log_det1.mean()], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log |Jacob det 1|: {}\n".format(test_log_det1.mean()))
    print("Testing log |Jacob det 1|: {}".format(test_log_det1.mean()))

    test_log_det2_log = os.path.join(result_folder, 'carefl_test_log_det2_log.txt')
    np.savetxt(test_log_det2_log, [test_log_det2.mean()], delimiter=',', fmt='%f')
    append_to_file(result_file, "Testing log |Jacob det 2|: {}\n".format(test_log_det2.mean()))
    print("Testing log |Jacob det 2|: {}".format(test_log_det2.mean()))

    # follow the code in CAREFL,
    # it calls ```z_obs = self._forward_flow(x_obs)``` in ```CAREFL.predict_counterfactual()```
    recons_Z = carefl_model._forward_flow(test_dset_numpy)

    compute_Var_E_given_C(carefl_model, direction, recons_Z, result_file, test_dset_numpy, test_log_det1, test_log_det2)

    ##################################################################################################################
    # compute the dependency between reconstructed residual of hypothesized effect and hypothesized cause using dHSIC
    ##################################################################################################################

    with torch.no_grad():
        if direction == "X1->X2":
            hypothesized_cause = test_dset_numpy[:, 0]
            recons_Z_hypothesized_effect = recons_Z[:, 1]
        elif direction == "X2->X1":
            hypothesized_cause = test_dset_numpy[:, 1]
            recons_Z_hypothesized_effect = recons_Z[:, 0]
        else:
            raise Exception("What??? direction={}".format(direction))

        results_R = call_dHSIC_from_R(dHSIC, hypothesized_cause, recons_Z_hypothesized_effect, dHSIC_kernels)
        dHSIC_value_X_Z = results_R.rx2("dHSIC")[0]
        print("estimated dHSIC value between the X and reconstructed Z: {}".format(dHSIC_value_X_Z))
        append_to_file(result_file, "estimated dHSIC value between X and reconstructed Z: {}\n".format(dHSIC_value_X_Z))

    #######################################################################
    # compute the dependency between reconstructed residuals using dHSIC
    #######################################################################

    with torch.no_grad():
        recons_Z1 = recons_Z[:, 0]
        recons_Z2 = recons_Z[:, 1]

        results_R = call_dHSIC_from_R(dHSIC, recons_Z1, recons_Z2, dHSIC_kernels)
        dHSIC_value_Z_Z = results_R.rx2("dHSIC")[0]
        print("estimated dHSIC value between the reconstructed Z's: {}".format(dHSIC_value_Z_Z))
        append_to_file(result_file, "estimated dHSIC value between the reconstructed Z's: {}\n".format(dHSIC_value_Z_Z))

        plot_data_distributions(result_folder, recons_error_figure_title, recons_Z1, "reconstructed Z1", recons_Z2,
                                "reconstructed Z2")

        if flow_SEM_type == "affine":
            # compute and plot the Reconstructed X using the Reconstructed Z
            recons_X = carefl_model._backward_flow(recons_Z)
            recons_X1 = recons_X[:, 0]
            recons_X2 = recons_X[:, 1]
            if np.all(B_true == np.array([[0, 1], [0, 0]])):
                # ground truth direction: X1->X2
                plot_title = "Reconstructed X"
                plot_datapoints(result_folder, plot_title, recons_X1, recons_X2, "Reconstructed X1", "Reconstructed X2")
            elif np.all(B_true == np.array([[0, 0], [1, 0]])):
                # ground truth direction: X2->X1
                plot_title = "Reconstructed X"
                plot_datapoints(result_folder, plot_title, recons_X2, recons_X1, "Reconstructed X2", "Reconstructed X1")
            else:
                raise Exception("What??? B_true=\n{}".format(B_true))

        recons_Z1_mean = recons_Z1.mean()
        recons_Z2_mean = recons_Z2.mean()
        recons_Z1_var = recons_Z1.var()
        recons_Z2_var = recons_Z2.var()
        print("mean of reconstructed Z1: {}".format(recons_Z1_mean))
        print("mean of reconstructed Z2: {}".format(recons_Z2_mean))
        print("variance of reconstructed Z1: {}".format(recons_Z1_var))
        print("variance of reconstructed Z2: {}".format(recons_Z2_var))
        append_to_file(result_file, "mean of reconstructed Z1: {}\n".format(recons_Z1_mean))
        append_to_file(result_file, "mean of reconstructed Z2: {}\n".format(recons_Z2_mean))
        append_to_file(result_file, "variance of reconstructed Z1: {}\n".format(recons_Z1_var))
        append_to_file(result_file, "variance of reconstructed Z2: {}\n".format(recons_Z2_var))

    return dHSIC_value_Z_Z, test_log_likelihood_mean, dHSIC_value_X_Z


def compute_Var_E_given_C(carefl_model, direction, recons_Z, result_file, test_dset_numpy, test_log_det1,
                          test_log_det2):
    """
    compute Var(Y|X) using Jacobian determinants.
    """
    # compute Var(Y|X) = g^2(x) * Var(Zy) = (dY/dZy)^2 * Var(Zy) = (dZy/dY)^-2 * Var(Zy) = (Jacob_Det)^-2 * Var(Zy)
    # log Var(Y|X) = log [(Jacob_Det)^-2 * Var(Zy)] = log [(Jacob_Det)^-2] + log [Var(Zy)]
    # = -2 * log [Jacob_Det] + log [Var(Zy)]

    var_Z = np.var(recons_Z, axis=0)

    if direction == "X1->X2":
        var_Ze = var_Z[1]
        log_jac_det = test_log_det2
    elif direction == "X2->X1":
        var_Ze = var_Z[0]
        log_jac_det = test_log_det1
    else:
        raise Exception("What??? direction={}".format(direction))

    print("log [Jacob_Det] mean: ", log_jac_det.mean())
    print("e^(-2*log [Jacob_Det]) mean: ", np.exp(-2 * log_jac_det).mean())
    print("max log [Jacob_Det]: ", log_jac_det.max())
    print("min log [Jacob_Det]: ", log_jac_det.min())
    print("max e^(-2*log [Jacob_Det]): ", np.exp(-2 * log_jac_det).max())
    print("min e^(-2*log [Jacob_Det]): ", np.exp(-2 * log_jac_det).min())
    print("Var(Zy): ", var_Ze)
    append_to_file(result_file, "log [Jacob_Det] mean: {}\n".format(log_jac_det.mean()))
    append_to_file(result_file, "e^(-2*log [Jacob_Det]) mean: {}\n".format(np.exp(-2 * log_jac_det).mean()))
    append_to_file(result_file, "max log [Jacob_Det]: {}\n".format(log_jac_det.max()))
    append_to_file(result_file, "min log [Jacob_Det]: {}\n".format(log_jac_det.min()))
    append_to_file(result_file, "max e^(-2*log [Jacob_Det]): {}\n".format(np.exp(-2 * log_jac_det).max()))
    append_to_file(result_file, "min e^(-2*log [Jacob_Det]): {}\n".format(np.exp(-2 * log_jac_det).min()))
    append_to_file(result_file, "Var(Zy): {}\n".format(var_Ze))

    log_var_Xe_given_Xc = -2 * log_jac_det + math.log(var_Ze)
    var_Xe_given_Xc = np.exp(log_var_Xe_given_Xc)
    print("Mean Var(Xe|Xc) = e^(-2 * log [Jacob_Det] + log [Var(Zy)]) : ", var_Xe_given_Xc.mean())
    append_to_file(result_file, "Mean Var(Xe|Xc) = e^(-2 * log [Jacob_Det] + log [Var(Zy)]) : {}\n"
                   .format(var_Xe_given_Xc.mean()))
    var_log = result_file.split(".txt")[0] + "_Estimated_Conditional_Variance_{}.txt".format(direction)
    append_to_file(var_log, "Mean Var(Xe|Xc) = e^(-2 * log [Jacob_Det] + log [Var(Zy)]) : {}\n"
                            "Var(Xe|Xc) = e^(-2 * log [Jacob_Det] + log [Var(Zy)]) : \n{}\n"
                   .format(var_Xe_given_Xc.mean(), var_Xe_given_Xc.tolist()))


def create_carefl_model(train_test_split, fix_AffineCL_forward, batch_size, result_folder, carefl_nl, carefl_nh,
                        carefl_prior_dist, carefl_epochs, device, weight_decay, flow_SEM_type):
    config = argparse.Namespace()

    '''
    By looking at the AffineCL class in the flows.py:
    ```scale_base``` controls s0
    ```shift_base``` controls t0
    ```scale``` controls s1
    ```shift``` controls t1
    If it is False, then the term is always 0 and not learned.
    '''

    config_flow = argparse.Namespace()
    # the type of neural networks to model a flow, # of hidden layers = MLP_X-2
    setattr(config_flow, "net_class", "mlp4")
    setattr(config_flow, "nl", carefl_nl)  # number of sub-flows per normalizing flow
    setattr(config_flow, "nh", carefl_nh)  # number of hidden neurons for the neural network of s() and t()
    setattr(config_flow, "prior_dist", carefl_prior_dist)  # prior distribution of a flow, P_z()
    setattr(config_flow, "architecture", 'CL')  # if 'cl' or 'realnvp', then AffineCL is used as the flow
    setattr(config_flow, "scale_base", True)
    setattr(config_flow, "shift_base", True)
    setattr(config_flow, "scale", True)
    setattr(config_flow, "shift", True)
    setattr(config_flow, "fix_AffineCL_forward", fix_AffineCL_forward)
    setattr(config_flow, "flow_SEM_type", flow_SEM_type)
    setattr(config, "flow", config_flow)

    config_training = argparse.Namespace()
    setattr(config_training, "epochs", carefl_epochs)
    setattr(config_training, "split", train_test_split)
    setattr(config_training, "seed", 1)
    setattr(config_training, "batch_size", batch_size)
    setattr(config_training, "verbose", False)
    setattr(config, "training", config_training)

    config_optim = argparse.Namespace()
    setattr(config_optim, "lr", 0.001)
    setattr(config_optim, "weight_decay", weight_decay)
    setattr(config_optim, "beta1", 0.9)
    setattr(config_optim, "optimizer", "Adam")
    setattr(config_optim, "amsgrad", False)
    setattr(config_optim, "scheduler",
            False)  # set to False so that the learning rate is not reduced with patience=3
    setattr(config, "optim", config_optim)

    setattr(config, "device", device)

    append_to_file(os.path.join(result_folder, 'CAREFL_Model_Config.txt'),
                   "config_flow: {} \nconfig_training:{}\nconfig_optim:{}"
                   .format(config_flow, config_training, config_optim))

    if carefl_nl is None or carefl_nh is None or carefl_prior_dist is None or fix_AffineCL_forward is None \
            or carefl_epochs is None or train_test_split is None or batch_size is None or weight_decay is None \
            or batch_size == -1:
        raise Exception(
            "what??? carefl_nl={},carefl_nh={},carefl_prior_dist={},fix_AffineCL_forward={},carefl_epochs={},"
            "train_test_split={},batch_size={}, weight_decay={}".format(
                carefl_nl, carefl_nh, carefl_prior_dist, fix_AffineCL_forward, carefl_epochs, train_test_split,
                batch_size, weight_decay))

    model = CAREFL(config)

    return model


def generate_DAG_and_data(sem_type, noise_type, result_folder_sem, D, EXPECTED_EDGES, variable_names,
                          normalize_datasets, n, g_magnitude):
    B_true = simulate_dag(D, EXPECTED_EDGES, 'ER')
    np.savetxt(os.path.join(result_folder_sem, 'W_true.csv'), B_true, delimiter=',')

    X, Z = simulate_nonlinear_sem(B_true, n, sem_type, noise_type, None, g_magnitude)
    draw_DAGs_using_LINGAM(os.path.join(result_folder_sem, "W_true_DAG"), B_true, variable_names)

    np.savetxt(os.path.join(result_folder_sem, 'X.csv'), X, delimiter=',')
    np.savetxt(os.path.join(result_folder_sem, 'Z.csv'), Z, delimiter=',')

    #############################################################
    # Plot the raw data
    #############################################################

    if np.all(B_true == np.array([[0, 1], [0, 0]])):
        # X1->X2
        plot_title = "ground-truth direction X1 -> X2"
        plot_datapoints(result_folder_sem, plot_title, X[:, 0], X[:, 1], "Raw X1", "Raw X2")
    elif np.all(B_true == np.array([[0, 0], [1, 0]])):
        # X2->X1
        plot_title = "ground-truth direction X2 -> X1"
        plot_datapoints(result_folder_sem, plot_title, X[:, 1], X[:, 0], "Raw X2", "Raw X1")
    else:
        raise Exception("What??? B_true=\n{}".format(B_true))

    if normalize_datasets:
        normalized_X = normalize_data(X)
        X = None  # make sure the data used to train the model is normalized.

        if np.all(B_true == np.array([[0, 1], [0, 0]])):
            # X1->X2
            plot_title = "ground-truth direction X1 -> X2 (normalized)"
            plot_datapoints(result_folder_sem, plot_title, normalized_X[:, 0], normalized_X[:, 1], "Normalized X1",
                            "Normalized X2")
        elif np.all(B_true == np.array([[0, 0], [1, 0]])):
            # X2->X1
            plot_title = "ground-truth direction X2 -> X1 (normalized)"
            plot_datapoints(result_folder_sem, plot_title, normalized_X[:, 1], normalized_X[:, 0], "Normalized X2",
                            "Normalized X1")
        else:
            raise Exception("What??? B_true=\n{}".format(B_true))

    else:
        normalized_X = X

    print("B_true: \n{}\n".format(B_true))

    return normalized_X, Z, B_true


def compute_GT_mean_conditional_variance(B_true, normalized_X, result_folder_sem, true_data_variance_logger):
    number_of_bins = 10
    n = normalized_X.shape[0]

    # X1->X2
    X1_min = np.min(normalized_X[:, 0])
    X1_max = np.max(normalized_X[:, 0])
    bin_boundaries = np.linspace(X1_min, X1_max, number_of_bins + 1)

    # to make all lower bounds exclusive later
    bin_boundaries[0] = bin_boundaries[0] - 1

    if len(bin_boundaries) != number_of_bins + 1:
        raise Exception("what??? len(bin_boundaries)={}".format(len(bin_boundaries)))

    total_points_all_bins = 0
    all_bins = [[] for _ in range(number_of_bins)]
    for b in range(0, len(bin_boundaries) - 1):
        points_in_bin = []
        bin_boundary_low = bin_boundaries[b]
        bin_boundary_high = bin_boundaries[b + 1]
        for i in range(len(normalized_X)):
            presumed_cause = normalized_X[i, 0]
            presumed_effect = normalized_X[i, 1]
            if bin_boundary_low < presumed_cause and presumed_cause <= bin_boundary_high:
                points_in_bin.append(presumed_effect)
                total_points_all_bins += 1

        all_bins[b] = points_in_bin

    if total_points_all_bins != n:
        raise Exception("what??? total_points_all_bins={}".format(total_points_all_bins))

    mean_var_X2_given_X1 = 0
    for bin in all_bins:
        if len(bin) > 0:
            mean_var_X2_given_X1 += (len(bin) / total_points_all_bins) * np.array(bin).var()
            # print("np.array(bin).var(): ", np.array(bin).var())
            # print("len(bin): ", len(bin))

    # print("mean_var_X2_given_X1: ", mean_var_X2_given_X1)

    # X2->X1
    X2_min = np.min(normalized_X[:, 1])
    X2_max = np.max(normalized_X[:, 1])
    bin_boundaries = np.linspace(X2_min, X2_max, number_of_bins + 1)

    # to make all lower bounds exclusive later
    bin_boundaries[0] = bin_boundaries[0] - 1

    if len(bin_boundaries) != number_of_bins + 1:
        raise Exception("what??? len(bin_boundaries)={}".format(len(bin_boundaries)))

    total_points_all_bins = 0
    all_bins = [[] for _ in range(number_of_bins)]
    for b in range(0, len(bin_boundaries) - 1):
        points_in_bin = []
        bin_boundary_low = bin_boundaries[b]
        bin_boundary_high = bin_boundaries[b + 1]
        for i in range(len(normalized_X)):
            presumed_cause = normalized_X[i, 1]
            presumed_effect = normalized_X[i, 0]
            if bin_boundary_low < presumed_cause and presumed_cause <= bin_boundary_high:
                points_in_bin.append(presumed_effect)
                total_points_all_bins += 1

        all_bins[b] = points_in_bin

    if total_points_all_bins != n:
        raise Exception("what??? total_points_all_bins={}".format(total_points_all_bins))

    mean_var_X1_given_X2 = 0
    for bin in all_bins:
        if len(bin) > 0:
            mean_var_X1_given_X2 += (len(bin) / total_points_all_bins) * np.array(bin).var()
            # print("np.array(bin).var(): ", np.array(bin).var())
            # print("len(bin): ", len(bin))

    # print("mean_var_X2_given_X1: ", mean_var_X1_given_X2)

    if np.all(B_true == np.array([[0, 1], [0, 0]])):
        # X1->X2
        mean_var_causal_direction = mean_var_X2_given_X1
        mean_var_anti_causal_direction = mean_var_X1_given_X2
    elif np.all(B_true == np.array([[0, 0], [1, 0]])):
        # X2->X1
        mean_var_causal_direction = mean_var_X1_given_X2
        mean_var_anti_causal_direction = mean_var_X2_given_X1
    else:
        raise Exception("What??? B_true=\n{}".format(B_true))

    print("Mean conditional variance in causal direction: ", mean_var_causal_direction)
    print("Mean conditional variance in anti-causal direction: ", mean_var_anti_causal_direction)

    if true_data_variance_logger is not None:
        true_data_variance_logger["number_of_datasets"] += 1
        true_data_variance_logger["causal_direction_total_conditional_variance"] += mean_var_causal_direction
        true_data_variance_logger["anti_causal_direction_total_conditional_variance"] += mean_var_anti_causal_direction

        if mean_var_causal_direction > mean_var_anti_causal_direction:
            true_data_variance_logger["number_causal_bigger_CV"] += 1
        elif mean_var_causal_direction < mean_var_anti_causal_direction:
            true_data_variance_logger["number_anti-causal_bigger_CV"] += 1

    append_to_file(os.path.join(result_folder_sem, 'True_Conditional_Variances.txt'),
                   "Computed from data using binning.\n"
                   "number_of_bins={}.\n"
                   "mean_var_causal_direction: {}\n"
                   "mean_var_anti_causal_direction: {}\n"
                   .format(number_of_bins, mean_var_causal_direction, mean_var_anti_causal_direction))


def run_single_test(method_to_test, counter, seed, dataset_type, SEMs, noises,
                    result_folder_root, carefl_nl, carefl_nh, carefl_prior_dist, batch_size, carefl_epochs,
                    training_file, training_directory, gt_x1_x2_directions, gt_x2_x1_directions, normalize_datasets, D,
                    split, fix_AffineCL_forward, variable_names, EXPECTED_EDGES, n, dHSIC, dHSIC_kernels,
                    training_dataset_index, df_training, df_labels, device, g_magnitude, true_data_variance_logger,
                    carefl_weight_decay):
    print("\n\n\n############################################")
    print("Seed: {}".format(seed))
    set_random_seed(seed)
    set_random_seed_R(seed)

    if dataset_type == "simulated":
        random_SEM = random.sample(SEMs, 1)[0]
        random_noise = random.sample(noises, 1)[0]

        if random_noise == SEM_Noises.standard_gaussian:
            gt_noise_type = SEM_Noises.standard_gaussian
        elif random_noise == SEM_Noises.gaussian0_4:
            gt_noise_type = SEM_Noises.gaussian0_4
        elif random_noise == SEM_Noises.gaussian0_25:
            gt_noise_type = SEM_Noises.gaussian0_25
        elif random_noise == SEM_Noises.gaussian0_100:
            gt_noise_type = SEM_Noises.gaussian0_100
        elif random_noise == SEM_Noises.gaussian0_400:
            gt_noise_type = SEM_Noises.gaussian0_400
        elif random_noise == SEM_Noises.laplace:
            gt_noise_type = SEM_Noises.laplace
        elif random_noise == SEM_Noises.uniform:
            gt_noise_type = SEM_Noises.uniform
        elif random_noise == SEM_Noises.exp:
            gt_noise_type = SEM_Noises.exp
        elif random_noise == SEM_Noises.beta0505:
            gt_noise_type = SEM_Noises.beta0505
        elif random_noise == SEM_Noises.continuous_bernoulli:
            gt_noise_type = SEM_Noises.continuous_bernoulli
        else:
            raise Exception("Need to add the noise {}.".format(random_noise))

        if random_SEM == SEM_Functionals.LSNM_tanh_exp_cosine:
            gt_sem_type = SEM_Functionals.LSNM_tanh_exp_cosine
        elif random_SEM == SEM_Functionals.LSNM_sine_tanh:
            gt_sem_type = SEM_Functionals.LSNM_sine_tanh
        elif random_SEM == SEM_Functionals.LSNM_sigmoid_sigmoid:
            gt_sem_type = SEM_Functionals.LSNM_sigmoid_sigmoid
        elif random_SEM == SEM_Functionals.ANM_sine:
            gt_sem_type = SEM_Functionals.ANM_sine
        elif random_SEM == SEM_Functionals.ANM_tanh:
            gt_sem_type = SEM_Functionals.ANM_tanh
        elif random_SEM == SEM_Functionals.ANM_sigmoid:
            gt_sem_type = SEM_Functionals.ANM_sigmoid
        else:
            raise Exception("What??? SEM: {}".format(random_SEM))

        result_folder_args = result_folder_root + "Ground-truth_SEM_{}/Noise_{}/g_magnitude_{}/n_{}/fix_AffineCL_forward_{}/" \
                                                  "nl_{}_nh_{}/prior_dist_{}/batch_size_{}/weight_decay_{}/" \
                                                  "epochs_{}/split_{}/Seed_{}/" \
            .format(gt_sem_type, gt_noise_type, g_magnitude, n, fix_AffineCL_forward, carefl_nl, carefl_nh,
                    carefl_prior_dist, batch_size, carefl_weight_decay, carefl_epochs, split, seed)

    elif dataset_type == "Tubingen_CEpairs" or dataset_type.startswith("SIM_benchmark"):
        result_folder_args = result_folder_root + "fix_AffineCL_forward_{}/nl_{}_nh_{}/prior_dist_{}/batch_size_{}/" \
                                                  "weight_decay_{}/epochs_{}/split_{}/training_file_{}/" \
            .format(fix_AffineCL_forward, carefl_nl, carefl_nh, carefl_prior_dist, batch_size, carefl_weight_decay,
                    carefl_epochs, split, training_file)

    else:
        raise Exception("what??? dataset_type={}".format(dataset_type))

    os.makedirs(result_folder_args, exist_ok=True)

    result_file = os.path.join(result_folder_args, 'result.txt')

    append_to_file(result_file, "\n\n\n############################################\n")

    print("result_folder: {}".format(result_folder_args))
    append_to_file(result_file, "result_folder: {}\n".format(result_folder_args))

    dataset_weight = None
    if dataset_type == "simulated":

        ############################################
        # Generate DAG and data
        ############################################

        print("\nStarting ground-truth SEM {} with ground-truth noise {}...".format(gt_sem_type, gt_noise_type))
        append_to_file(result_file, "\nStarting ground-truth SEM {} with ground-truth noise {}...\n\n"
                       .format(gt_sem_type, gt_noise_type))

        normalized_X, Z, B_true = generate_DAG_and_data(gt_sem_type, gt_noise_type, result_folder_args, D,
                                                        EXPECTED_EDGES, variable_names, normalize_datasets, n,
                                                        g_magnitude)

        append_to_file(result_file, "\nB_true: \n{}\n\n".format(B_true))

        #############################################################
        # compute dependence between ground-truth Z's
        #############################################################

        # normalized_Z1 = normalize_data(Z[:, 0].reshape(-1, 1)).reshape(-1, )
        # normalized_Z2 = normalize_data(Z[:, 1].reshape(-1, 1)).reshape(-1, )
        # independence_conclusion = call_dHSIC_test_from_R(dHSIC, normalized_Z1, normalized_Z2, dHSIC_test_method,
        #                                                  dHSIC_kernels, result_file)
        # print("independence conclusion between the ground-truth z1 and ground-truth z2 using critical_value: {}".format(
        #     independence_conclusion))
        # append_to_file(result_file,
        #                "independence conclusion between the ground-truth z1 and ground-truth z2 using critical_value: {}\n"
        #                .format(independence_conclusion))

        plot_data_distributions(result_folder_args, "Z-{}-{}".format(gt_sem_type, gt_noise_type), Z[:, 0],
                                "Ground-truth Z1", Z[:, 1], "Ground-truth Z2")

        normalized_X1 = normalized_X[:, 0]
        normalized_X2 = normalized_X[:, 1]
        plot_data_distributions(result_folder_args, "X-{}-{}".format(gt_sem_type, gt_noise_type), normalized_X1,
                                "Ground-truth X1", normalized_X2, "Ground-truth X2")

        # the CAREFL-IT just compares the two HSIC values directly, it does not use p-values from the HSIC test anymore.
        # therefore, we can skip the following.
        # if independence_conclusion == "dependent":
        #     print("The random generated ground-truth noises should be independent. Skip the dataset.")
        #     return

        normalized_X1 = None
        normalized_X2 = None

    elif dataset_type.startswith("SIM_benchmark") or dataset_type == "Tubingen_CEpairs":

        ############################################
        # Read data
        ############################################

        if dataset_type == "Tubingen_CEpairs":
            data, B_true, plot_title, dataset_weight = read_Tubingen_data_file(training_directory, training_file)

            print("dataset_weight: \n{}\n".format(dataset_weight))
            append_to_file(result_file, "dataset_weight: \n{}\n".format(dataset_weight))

            if data.shape[1] != D:
                print("data.shape: {}".format(data.shape))
                # according to the meta file, all the datasets here are already valid bivariate datasets with
                # cause and effect indexes being 1 or 2, if D!=2, then get rid of the 3rd column
                data = data[:, 0:2]
                print("data.shape: {}".format(data.shape))

        else:
            data, B_true, plot_title, _ = read_Tubingen_data_file(training_directory, training_file)

        print("B_true: \n{}\n".format(B_true))
        append_to_file(result_file, "B_true: \n{}\n".format(B_true))

        if data.shape[1] != D:
            raise Exception("what??? data.shape={}".format(data.shape))

        print("data.shape: {}".format(data.shape))
        append_to_file(result_file, "data.shape: {}\n".format(data.shape))

        ####################################################################################
        # Plot the raw data points (looks the same as normalized data points)
        ####################################################################################

        if "X1 -> X2" in plot_title:
            plot_datapoints(result_folder_args, plot_title, data[:, 0], data[:, 1], "Raw X1", "Raw X2")
        elif "X2 -> X1" in plot_title:
            plot_datapoints(result_folder_args, plot_title, data[:, 1], data[:, 0], "Raw X2", "Raw X1")
        else:
            raise Exception("What??? plot_title={}".format(plot_title))

        ############################################
        # Normalize data
        ############################################
        if normalize_datasets:
            normalized_X = normalize_data(data)
            data = None  # make sure the data used to train the model is normalized.
        else:
            normalized_X = data

    else:
        raise Exception("what??? dataset_type={}".format(dataset_type))

    #################################################################################
    # compute the true Var(Y|X) and Var(X|Y) of the data using binning.
    #################################################################################

    compute_GT_mean_conditional_variance(B_true, normalized_X, result_folder_args, true_data_variance_logger)

    if batch_size == -1:
        # if batch_size==-1, meaning all data in one batch
        updated_batch_size = normalized_X.shape[0]
    else:
        updated_batch_size = batch_size

    counter["total"] += 1

    if dataset_type == "Tubingen_CEpairs":
        counter["total_Tubingen_CEpairs_weight_sum"] += dataset_weight

    if method_to_test == "LOCI":
        ###############################################################################
        # LOCI
        ###############################################################################
        print("\nRunning LOCI...")

        result_folder_method = os.path.join(result_folder_args, "LOCI")
        os.makedirs(result_folder_method, exist_ok=True)

        run_LOCI(B_true, counter, dHSIC, dHSIC_kernels, device, normalized_X, result_folder_method, seed,
                 updated_batch_size)

    if "CAREFL" in method_to_test:

        if method_to_test == "CAREFL-LSNM":
            flow_SEM_type = "affine"
        elif method_to_test == "CAREFL-ANM":
            flow_SEM_type = "ANM"
        else:
            raise Exception("what??? method_to_test: {}".format(method_to_test))

        ###############################################################################
        # estimate the causal direction using a CAREFL model
        ###############################################################################

        result_folder_method = os.path.join(result_folder_args, "CAREFL")
        os.makedirs(result_folder_method, exist_ok=True)

        #################################################################################
        # Fit a model in the direction of X1->X2
        #################################################################################

        if dataset_type == "simulated":
            recons_error_figure_title = "{}_recons_z".format(gt_sem_type)
        elif dataset_type == "Tubingen_CEpairs":
            recons_error_figure_title = "{}_recons_z".format(training_file.replace(".", ""))
        elif dataset_type.startswith("SIM_benchmark"):
            recons_error_figure_title = "{}_recons_z".format(training_file.replace(".", ""))
        else:
            raise Exception("what??? dataset_type={}".format(dataset_type))

        direction = "X1->X2"
        print("\nFitting {} in the direction {}...".format("CAREFL", direction))
        append_to_file(result_file, "\nFitting {} in the direction {}...\n".format("CAREFL", direction))

        result_folder_direction = os.path.join(result_folder_method, "{}".format(direction))
        os.makedirs(result_folder_direction, exist_ok=True)

        dHSIC_value_ZZ_x1_x2, test_log_likelihood_x1_x2, dHSIC_value_XZ_x1_x2 = run_carefl_in_one_direction(
            direction, normalized_X, result_folder_direction, recons_error_figure_title, result_file, B_true,
            updated_batch_size, split, fix_AffineCL_forward, dHSIC, dHSIC_kernels, result_folder_method, carefl_nl,
            carefl_nh, carefl_prior_dist, carefl_epochs, device, carefl_weight_decay, flow_SEM_type)

        #################################################################################
        # Fit a model in the direction of X2->X1
        #################################################################################

        direction = "X2->X1"
        print("\nFitting {} in the direction {}...".format("CAREFL", direction))
        append_to_file(result_file, "\nFitting {} in the direction {}...\n".format("CAREFL", direction))

        result_folder_direction = os.path.join(result_folder_method, "{}".format(direction))
        os.makedirs(result_folder_direction, exist_ok=True)

        dHSIC_value_ZZ_x2_x1, test_log_likelihood_x2_x1, dHSIC_value_XZ_x2_x1 = run_carefl_in_one_direction(
            direction, normalized_X, result_folder_direction, recons_error_figure_title, result_file, B_true,
            updated_batch_size, split, fix_AffineCL_forward, dHSIC, dHSIC_kernels, result_folder_method, carefl_nl,
            carefl_nh, carefl_prior_dist, carefl_epochs, device, carefl_weight_decay, flow_SEM_type)

        #################################################################################
        # compare the testing log-likelihoods
        #################################################################################

        if test_log_likelihood_x1_x2 > test_log_likelihood_x2_x1:  # X1->X2
            estimated_direction_LR = np.array([[0, 1], [0, 0]])
        elif test_log_likelihood_x1_x2 < test_log_likelihood_x2_x1:  # X2->X1
            estimated_direction_LR = np.array([[0, 0], [1, 0]])
        else:  # no conclusion
            estimated_direction_LR = np.array([[0, 0], [0, 0]])

        print("\nA_estimated using CAREFL-LR: \n{}".format(estimated_direction_LR))
        append_to_file(result_file, "A_estimated using CAREFL-LR: \n{}\n".format(estimated_direction_LR))

        np.savetxt(os.path.join(result_folder_method, 'W_estimated_DAG_LR.csv'), estimated_direction_LR,
                   delimiter=',')
        draw_DAGs_using_LINGAM(os.path.join(result_folder_method, "W_estimated_DAG_LR"), estimated_direction_LR,
                               variable_names)

        if np.all(B_true == estimated_direction_LR):
            correct_direction_estimation = 1
        else:
            correct_direction_estimation = 0

        counter["correct_CAREFL_LR"] += correct_direction_estimation

        if dataset_type == "Tubingen_CEpairs":
            if correct_direction_estimation == 1:
                counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"] += dataset_weight

        #################################################################################
        # compare the HSIC values between Z's
        #################################################################################

        if dHSIC_value_ZZ_x1_x2 < dHSIC_value_ZZ_x2_x1:  # X1->X2
            estimated_direction_HSIC = np.array([[0, 1], [0, 0]])
        elif dHSIC_value_ZZ_x1_x2 > dHSIC_value_ZZ_x2_x1:  # X2->X1
            estimated_direction_HSIC = np.array([[0, 0], [1, 0]])
        else:  # no conclusion
            estimated_direction_HSIC = np.array([[0, 0], [0, 0]])

        print("\nA_estimated using CAREFL-IT-ZZ: \n{}".format(estimated_direction_HSIC))
        append_to_file(result_file, "A_estimated using CAREFL-IT-ZZ: \n{}\n".format(estimated_direction_HSIC))

        np.savetxt(os.path.join(result_folder_method, 'W_estimated_DAG_IT_ZZ.csv'), estimated_direction_HSIC,
                   delimiter=',')
        draw_DAGs_using_LINGAM(os.path.join(result_folder_method, "W_estimated_DAG_IT_ZZ"), estimated_direction_HSIC,
                               variable_names)

        if np.all(B_true == estimated_direction_HSIC):
            correct_direction_estimation = 1
        else:
            correct_direction_estimation = 0

        counter["correct_CAREFL_IT_ZZ"] += correct_direction_estimation

        if dataset_type == "Tubingen_CEpairs":
            if correct_direction_estimation == 1:
                counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"] += dataset_weight

        #################################################################################
        # compare the HSIC values between X and Z
        #################################################################################

        if dHSIC_value_XZ_x1_x2 < dHSIC_value_XZ_x2_x1:  # X1->X2
            estimated_direction_HSIC = np.array([[0, 1], [0, 0]])
        elif dHSIC_value_XZ_x1_x2 > dHSIC_value_XZ_x2_x1:  # X2->X1
            estimated_direction_HSIC = np.array([[0, 0], [1, 0]])
        else:  # no conclusion
            estimated_direction_HSIC = np.array([[0, 0], [0, 0]])

        print("\nA_estimated using CAREFL-IT-XZ: \n{}".format(estimated_direction_HSIC))
        append_to_file(result_file, "A_estimated using CAREFL-IT-XZ: \n{}\n".format(estimated_direction_HSIC))

        np.savetxt(os.path.join(result_folder_method, 'W_estimated_DAG_IT_XZ.csv'), estimated_direction_HSIC,
                   delimiter=',')
        draw_DAGs_using_LINGAM(os.path.join(result_folder_method, "W_estimated_DAG_IT_XZ"), estimated_direction_HSIC,
                               variable_names)

        if np.all(B_true == estimated_direction_HSIC):
            correct_direction_estimation = 1
        else:
            correct_direction_estimation = 0

        counter["correct_CAREFL_IT_XZ"] += correct_direction_estimation

        if dataset_type == "Tubingen_CEpairs":
            if correct_direction_estimation == 1:
                counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"] += dataset_weight

    print("\nSo far, Number of datasets tested: {}".format(counter["total"]))

    if "CAREFL" in method_to_test:
        print("So far, Number of directions correctly detected by CAREFL-LR: {}".format(counter["correct_CAREFL_LR"]))
        print("So far, Number of directions correctly detected by CAREFL-IT-ZZ: {}"
              .format(counter["correct_CAREFL_IT_ZZ"]))
        print("So far, Number of directions correctly detected by CAREFL-IT-XZ: {}"
              .format(counter["correct_CAREFL_IT_XZ"]))

        if dataset_type == "Tubingen_CEpairs":
            print("So far, sum of dataset weights: {}".format(counter["total_Tubingen_CEpairs_weight_sum"]))
            print("So far, sum of correct dataset weights by CAREFL-LR: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"]))
            print("So far, sum of correct dataset weights by CAREFL-IT-ZZ: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"]))
            print("So far, sum of correct dataset weights by CAREFL-IT-XZ: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"]))

    if method_to_test == "LOCI":
        print("So far, Number of directions correctly detected by LOCI_LinReg: {}"
              .format(counter["correct_LOCI_LinReg"]))
        print("So far, Number of directions correctly detected by LOCI_LinReg_HSIC: {}"
              .format(counter["correct_LOCI_LinReg_HSIC"]))
        print("So far, Number of directions correctly detected by LOCI_NN: {}".format(counter["correct_LOCI_NN"]))
        print("So far, Number of directions correctly detected by LOCI_NN_HSIC: {}".format(
            counter["correct_LOCI_NN_HSIC"]))

    print()

    return counter


def run_LOCI(B_true, counter, dHSIC, dHSIC_kernels, device, normalized_X, result_folder_method, seed,
             updated_batch_size):
    loci_result_id = 0
    df = pd.DataFrame(index=[loci_result_id])

    X1_numpy = normalized_X[:, 0].reshape(-1, 1)  # numpy (N, 1)
    X2_numpy = normalized_X[:, 1].reshape(-1, 1)  # numpy (N, 1)

    # run linear models with spline feature maps
    spline_trans = SplineTransformer(n_knots=25, degree=5)
    Phi_X1 = spline_trans.fit_transform(X1_numpy)
    Phi_X2 = spline_trans.fit_transform(X2_numpy)

    X1_tensor = torch.from_numpy(X1_numpy)  # Tensor (N, 1)
    X2_tensor = torch.from_numpy(X2_numpy)  # Tensor (N, 1)

    run_LinReg_Convex_LSNM(df, loci_result_id, device, X1_tensor, X2_tensor, Phi_X1, Phi_X2, dHSIC, dHSIC_kernels)

    n_data = X1_tensor.shape[0]
    d = X1_tensor.shape[1]

    map_kwargs = dict(
        scheduler='cos',
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=5000,
        nu_noise_init=0.5,
    )

    run_NN_LSNM(df, loci_result_id, device, seed, updated_batch_size, X1_tensor.float(), X2_tensor.float(), n_data,
                d, dHSIC, dHSIC_kernels, map_kwargs)

    infer_direction_results(df, loci_result_id)

    df.to_csv("{}/{}.csv".format(result_folder_method, "results"))

    for column in df:
        if "Result:" in df[column].name:
            print("\nDirection estimated using {}: \n{}".format(df[column].name.split(": ")[-1], df[column][0]))

    if df.loc[loci_result_id, 'Result: LinReg_Convex_LSNM'] == "X1->X2":
        estimated_direction = np.array([[0, 1], [0, 0]])
    elif df.loc[loci_result_id, 'Result: LinReg_Convex_LSNM'] == "X2->X1":
        estimated_direction = np.array([[0, 0], [1, 0]])
    else:  # no conclusion
        estimated_direction = np.array([[0, 0], [0, 0]])

    if np.all(B_true == estimated_direction):
        correct_direction_estimation = 1
    else:
        correct_direction_estimation = 0

    counter["correct_LOCI_LinReg"] += correct_direction_estimation

    if df.loc[loci_result_id, 'Result: LinReg_Convex_LSNM_HSIC'] == "X1->X2":
        estimated_direction = np.array([[0, 1], [0, 0]])
    elif df.loc[loci_result_id, 'Result: LinReg_Convex_LSNM_HSIC'] == "X2->X1":
        estimated_direction = np.array([[0, 0], [1, 0]])
    else:  # no conclusion
        estimated_direction = np.array([[0, 0], [0, 0]])

    if np.all(B_true == estimated_direction):
        correct_direction_estimation = 1
    else:
        correct_direction_estimation = 0

    counter["correct_LOCI_LinReg_HSIC"] += correct_direction_estimation

    if df.loc[loci_result_id, 'Result: NN_LSNM'] == "X1->X2":
        estimated_direction = np.array([[0, 1], [0, 0]])
    elif df.loc[loci_result_id, 'Result: NN_LSNM'] == "X2->X1":
        estimated_direction = np.array([[0, 0], [1, 0]])
    else:  # no conclusion
        estimated_direction = np.array([[0, 0], [0, 0]])

    if np.all(B_true == estimated_direction):
        correct_direction_estimation = 1
    else:
        correct_direction_estimation = 0

    counter["correct_LOCI_NN"] += correct_direction_estimation

    if df.loc[loci_result_id, 'Result: NN_LSNM_HSIC'] == "X1->X2":
        estimated_direction = np.array([[0, 1], [0, 0]])
    elif df.loc[loci_result_id, 'Result: NN_LSNM_HSIC'] == "X2->X1":
        estimated_direction = np.array([[0, 0], [1, 0]])
    else:  # no conclusion
        estimated_direction = np.array([[0, 0], [0, 0]])

    if np.all(B_true == estimated_direction):
        correct_direction_estimation = 1
    else:
        correct_direction_estimation = 0

    counter["correct_LOCI_NN_HSIC"] += correct_direction_estimation


def run_multiple_times(args, result_folder_root, result_summary_file):
    D, H = 2, 0.5
    EXPECTED_EDGES = int(D * H)

    method_to_test = args.method_to_test
    dataset_type = args.dataset_type
    normalize_datasets = args.normalize_datasets
    n = args.n
    number_of_datasets = args.number_of_datasets
    SEM = args.SEM
    noise = args.noise
    g_magnitude = args.g_magnitude
    # dHSIC_test_method = args.dHSIC_test_method
    dHSIC_kernel1 = args.dHSIC_kernel1
    dHSIC_kernel2 = args.dHSIC_kernel2
    dHSIC_kernels = [dHSIC_kernel1, dHSIC_kernel2]
    carefl_nl = args.carefl_nl
    carefl_nh = args.carefl_nh
    carefl_prior_dist = args.carefl_prior_dist
    fix_AffineCL_forward = args.fix_AffineCL_forward
    carefl_weight_decay = args.carefl_weight_decay
    carefl_epochs = args.carefl_epochs
    batch_size = args.batch_size
    split = args.split
    device = args.device

    if dataset_type != "simulated":
        n = None
        number_of_datasets = None
        SEM = None
        noise = None
        H = None
        EXPECTED_EDGES = None
        g_magnitude = None

    print("method_to_test: ", method_to_test)
    print("dataset_type: ", dataset_type)
    print("normalize_datasets: ", normalize_datasets)
    print('n: ', n)
    print('number_of_datasets: ', number_of_datasets)
    print('SEM: ', SEM)
    print('noise: ', noise)
    print('g_magnitude', g_magnitude)
    # print('dHSIC_test_method: ', dHSIC_test_method)
    print('dHSIC_kernels: ', dHSIC_kernels)
    print('carefl_nl: ', carefl_nl)
    print('carefl_nh: ', carefl_nh)
    print('carefl_prior_dist: ', carefl_prior_dist)
    print('fix_AffineCL_forward: ', fix_AffineCL_forward)
    print("carefl_weight_decay: ", carefl_weight_decay)
    print('carefl_epochs: ', carefl_epochs)
    print("batch_size: ", batch_size)
    print("split: ", split)
    print("device: ", device)
    print("result_folder: ", args.result_folder)
    print("result_folder_suffix: ", args.result_folder_suffix)

    variable_names = ['X{}'.format(j) for j in range(1, D + 1)]

    ############################################
    # Call R package dHSIC to detect dependency
    ############################################

    # Install the dHSIC package
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # choose a CRAN mirror
    utils.install_packages("dHSIC")  # https://cran.r-project.org/web/packages/dHSIC/dHSIC.pdf
    dHSIC = importr('dHSIC')  # Load the dHSIC package, i.e. ```library("dHSIC")``` in R

    i = extract_R_versions(utils.installed_packages())
    package_name = 'dHSIC'
    print("dHSIC Version: ", i[package_name])

    r("""cat(paste("R version: ",R.version.string))""")  # print R version

    print("\nPytorch version: ", torch.__version__)

    ############################################
    # Start the experiments
    ############################################

    true_data_variance_logger = {
        "number_of_datasets": 0,
        "causal_direction_total_conditional_variance": 0,
        "anti_causal_direction_total_conditional_variance": 0,
        "number_causal_bigger_CV": 0,
        "number_anti-causal_bigger_CV": 0,
    }

    if dataset_type == "simulated":
        if SEM == 'any':
            SEMs = ["LSNM-tanh-exp-cosine", "LSNM-sine-tanh", "LSNM-sigmoid-sigmoid"]
        else:
            SEMs = [SEM]

        if noise == 'any':
            noises = ['laplace', 'standard-gaussian', "gaussian0_4", "gaussian0_25", "gaussian0_100",
                      "gaussian0_400", 'uniform', 'exp', 'beta0505', 'continuous-bernoulli']
        else:
            noises = [noise]

    elif dataset_type.startswith("SIM_benchmark") or dataset_type == "Tubingen_CEpairs":

        if dataset_type == "SIM_benchmark_SIM":  # default
            training_directory = "../data/SIM/SIM"
        elif dataset_type == "SIM_benchmark_SIM-c":  # with confounder
            training_directory = "../data/SIM/SIM-c"
        elif dataset_type == "SIM_benchmark_SIM-G":  # approximate Gaussian
            training_directory = "../data/SIM/SIM-G"
        elif dataset_type == "SIM_benchmark_SIM-ln":  # low noise levels
            training_directory = "../data/SIM/SIM-ln"
        elif dataset_type == "Tubingen_CEpairs":
            # https://webdav.tuebingen.mpg.de/cause-effect/
            training_directory = "../data/Tubingen_CEpairs/"
        else:
            raise Exception("what??? dataset_type={}.".format(dataset_type))

        discrete_datasets_to_remove = [47, 70, 107]  # follow previous works
        training_files = []
        for filename in os.listdir(training_directory):
            if filename.startswith("pair") and "meta" not in filename and "_des" not in filename:

                if dataset_type == "Tubingen_CEpairs":
                    pair_id = int(filename.replace("pair", "").replace(".txt", ""))
                    is_bivariate = is_bivariate_Tubingen_dataset(training_directory, pair_id)

                    if is_bivariate and (pair_id not in discrete_datasets_to_remove):
                        training_files.append(filename)
                    else:
                        if not is_bivariate:
                            print("Skip data file {} because it's multivariate.".format(filename))

                        if (pair_id in discrete_datasets_to_remove):
                            print("Skip data file {} because it's discrete.".format(filename))
                else:
                    training_files.append(filename)

        training_files.sort()
        print("{} training files: {}".format(len(training_files), training_files))

        if dataset_type == "Tubingen_CEpairs" and len(training_files) != 99:
            raise Exception("what??? len(training_files) should be 99, but is {}".format(len(training_files)))

    else:
        raise Exception("what??? dataset_type={}".format(dataset_type))

    counter = {
        "total": 0,
        "correct_CAREFL_LR": None,
        "correct_CAREFL_IT_ZZ": None,
        "correct_CAREFL_IT_XZ": None,
        "correct_LOCI_LinReg": None,
        "correct_LOCI_LinReg_HSIC": None,
        "correct_LOCI_NN": None,
        "correct_LOCI_NN_HSIC": None,
        "correct_Tubingen_CEpairs_CAREFL_LR_weight_sum": None,
        "correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum": None,
        "correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum": None,
        "total_Tubingen_CEpairs_weight_sum": None
    }

    if "CAREFL" in method_to_test:
        counter["correct_CAREFL_LR"] = 0
        counter["correct_CAREFL_IT_ZZ"] = 0
        counter["correct_CAREFL_IT_XZ"] = 0

    if method_to_test == "LOCI":
        counter["correct_LOCI_LinReg"] = 0
        counter["correct_LOCI_LinReg_HSIC"] = 0
        counter["correct_LOCI_NN"] = 0
        counter["correct_LOCI_NN_HSIC"] = 0

    if dataset_type == "simulated":
        for seed in range(1, number_of_datasets + 1):
            counter = run_single_test(
                method_to_test, counter, seed, dataset_type, SEMs, noises,
                result_folder_root, carefl_nl, carefl_nh, carefl_prior_dist, batch_size, carefl_epochs, None, None,
                None, None, normalize_datasets, D, split, fix_AffineCL_forward, variable_names, EXPECTED_EDGES, n,
                dHSIC, dHSIC_kernels, None, None, None, device, g_magnitude, true_data_variance_logger,
                carefl_weight_decay)

    elif dataset_type == "Tubingen_CEpairs":
        seed = 2

        counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"] = 0
        counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"] = 0
        counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"] = 0
        counter["total_Tubingen_CEpairs_weight_sum"] = 0

        if method_to_test != "CAREFL":
            raise Exception("counter entries for other methods with weighted accuracy are not implemented yet.")

        for training_file in training_files:
            counter = run_single_test(
                method_to_test, counter, seed, dataset_type, None, None,
                result_folder_root, carefl_nl, carefl_nh, carefl_prior_dist, batch_size, carefl_epochs, training_file,
                training_directory, None, None, normalize_datasets, D, split,
                fix_AffineCL_forward, variable_names, EXPECTED_EDGES, n, dHSIC, dHSIC_kernels, None, None, None, device,
                None, true_data_variance_logger, carefl_weight_decay)

    elif dataset_type.startswith("SIM_benchmark"):
        seed = 2
        for training_file in training_files:
            counter = run_single_test(
                method_to_test, counter, seed, dataset_type, None, None,
                result_folder_root, carefl_nl, carefl_nh, carefl_prior_dist, batch_size, carefl_epochs, training_file,
                training_directory, None, None, normalize_datasets, D, split,
                fix_AffineCL_forward, variable_names, EXPECTED_EDGES, n, dHSIC, dHSIC_kernels, None, None, None, device,
                None, true_data_variance_logger, carefl_weight_decay)

    else:
        raise Exception("what??? dataset_type={}".format(dataset_type))

    print("\n\n\nDone.")

    print("\nNumber of datasets tested: {}".format(counter["total"]))
    if "CAREFL" in method_to_test:
        print("Number of directions correctly detected by CAREFL-LR: {}".format(counter["correct_CAREFL_LR"]))
        print("Number of directions correctly detected by CAREFL-IT-ZZ: {}".format(counter["correct_CAREFL_IT_ZZ"]))
        print("Number of directions correctly detected by CAREFL-IT-XZ: {}".format(counter["correct_CAREFL_IT_XZ"]))

        if dataset_type == "Tubingen_CEpairs":
            print("Sum of dataset weights: {}".format(counter["total_Tubingen_CEpairs_weight_sum"]))
            print("Sum of correct dataset weights by CAREFL-LR: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"]))
            print("Sum of correct dataset weights by CAREFL-IT-ZZ: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"]))
            print("Sum of correct dataset weights by CAREFL-IT-XZ: {}"
                  .format(counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"]))

    if method_to_test == "LOCI":
        print("Number of directions correctly detected by LOCI_LinReg: {}".format(counter["correct_LOCI_LinReg"]))
        print("Number of directions correctly detected by LOCI_LinReg_HSIC: {}"
              .format(counter["correct_LOCI_LinReg_HSIC"]))
        print("Number of directions correctly detected by LOCI_NN: {}".format(counter["correct_LOCI_NN"]))
        print("Number of directions correctly detected by LOCI_NN_HSIC: {}".format(counter["correct_LOCI_NN_HSIC"]))

    print()

    append_to_file(result_summary_file,
                   "dataset_type, SEM, g_magnitude, noise, n, method_to_test, carefl_nl, carefl_nh, carefl_prior_dist, "
                   "carefl_epochs, carefl_weight_decay, train_test_split, batch_size, number_datasets, "
                   "CAREFL-LR correct_count, CAREFL-IT-ZZ correct_count, CAREFL-IT-XZ correct_count, "
                   "LOCI_LinReg correct_count, LOCI_LinReg_HSIC correct_count, LOCI_NN correct_count, "
                   "LOCI_NN_HSIC correct_count, CAREFL-LR accuracy, CAREFL-IT-ZZ accuracy, CAREFL-IT-XZ accuracy,"
                   "LOCI_LinReg accuracy, LOCI_LinReg_HSIC accuracy, LOCI_NN accuracy, LOCI_NN_HSIC accuracy, "
                   "causal_direction_mean_conditional_variance, anti_causal_direction_mean_conditional_variance, "
                   "number_causal_bigger_CV, number_anti-causal_bigger_CV, total_Tubingen_CEpairs_weight_sum, "
                   "Tubingen_CEpairs_CAREFL_LR_weighted_accuracy, Tubingen_CEpairs_CAREFL_IT_ZZ_weighted_accuracy, "
                   "Tubingen_CEpairs_CAREFL_IT_XZ_weighted_accuracy \n"
                   "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
                   "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n"
                   .format(dataset_type, SEM, g_magnitude, noise, n, method_to_test, carefl_nl, carefl_nh,
                           carefl_prior_dist, carefl_epochs, carefl_weight_decay, split, batch_size, counter["total"],
                           counter["correct_CAREFL_LR"], counter["correct_CAREFL_IT_ZZ"],
                           counter["correct_CAREFL_IT_XZ"], counter["correct_LOCI_LinReg"],
                           counter["correct_LOCI_LinReg_HSIC"], counter["correct_LOCI_NN"],
                           counter["correct_LOCI_NN_HSIC"],
                           counter["correct_CAREFL_LR"] / counter["total"]
                           if counter["correct_CAREFL_LR"] is not None else None,
                           counter["correct_CAREFL_IT_ZZ"] / counter["total"]
                           if counter["correct_CAREFL_IT_ZZ"] is not None else None,
                           counter["correct_CAREFL_IT_XZ"] / counter["total"]
                           if counter["correct_CAREFL_IT_XZ"] is not None else None,
                           counter["correct_LOCI_LinReg"] / counter["total"]
                           if counter["correct_LOCI_LinReg"] is not None else None,
                           counter["correct_LOCI_LinReg_HSIC"] / counter["total"]
                           if counter["correct_LOCI_LinReg_HSIC"] is not None else None,
                           counter["correct_LOCI_NN"] / counter["total"]
                           if counter["correct_LOCI_NN"] is not None else None,
                           counter["correct_LOCI_NN_HSIC"] / counter["total"]
                           if counter["correct_LOCI_NN_HSIC"] is not None else None,
                           true_data_variance_logger["causal_direction_total_conditional_variance"] /
                           true_data_variance_logger["number_of_datasets"]
                           if true_data_variance_logger is not None else None,
                           true_data_variance_logger["anti_causal_direction_total_conditional_variance"] /
                           true_data_variance_logger["number_of_datasets"]
                           if true_data_variance_logger is not None else None,
                           true_data_variance_logger["number_causal_bigger_CV"]
                           if true_data_variance_logger is not None else None,
                           true_data_variance_logger["number_anti-causal_bigger_CV"]
                           if true_data_variance_logger is not None else None,

                           counter["total_Tubingen_CEpairs_weight_sum"],

                           counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"]
                           / counter["total_Tubingen_CEpairs_weight_sum"]
                           if counter["correct_Tubingen_CEpairs_CAREFL_LR_weight_sum"] is not None else None,

                           counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"]
                           / counter["total_Tubingen_CEpairs_weight_sum"]
                           if counter["correct_Tubingen_CEpairs_CAREFL_IT_ZZ_weight_sum"] is not None else None,

                           counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"]
                           / counter["total_Tubingen_CEpairs_weight_sum"]
                           if counter["correct_Tubingen_CEpairs_CAREFL_IT_XZ_weight_sum"] is not None else None,

                           ))


if __name__ == "__main__":
    """
    First, run CAREFL with likelihood ratio.
    Then, run CAREFL with independence test.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--method_to_test', required=False, type=str, choices=["CAREFL-LSNM", "LOCI", "CAREFL-ANM", ],
                        default="CAREFL-LSNM")

    parser.add_argument('--dataset_type', required=False, type=str,
                        choices=["simulated", "Tubingen_CEpairs", "SIM_benchmark_SIM", "SIM_benchmark_SIM-G",
                                 "SIM_benchmark_SIM-ln", "SIM_benchmark_SIM-c"],
                        default="simulated")

    parser.add_argument('--normalize_datasets', required=False, type=int, choices=[0, 1], default=1)

    parser.add_argument('--number_of_datasets', required=False, type=int, default=10)

    parser.add_argument('--SEM', required=False, type=str, default="any")
    parser.add_argument('--noise', required=False, type=str, default="any")
    parser.add_argument('--g_magnitude', required=False, type=float, default=1.0)

    parser.add_argument('--n', required=False, type=int,
                        choices=[-1, 25, 50, 75, 100, 150, 250, 500, 1000, 5000, 10000], default=10000)

    # https://cran.r-project.org/web/packages/dHSIC/dHSIC.pdf
    # parser.add_argument('--dHSIC_test_method', required=False, type=str, default="gamma",
    #                     choices=["permutation", "gamma"])
    parser.add_argument('--dHSIC_kernel1', required=False, type=str, choices=["gaussian", "discrete"],
                        default="gaussian")
    parser.add_argument('--dHSIC_kernel2', required=False, type=str, choices=["gaussian", "discrete"],
                        default="gaussian")

    parser.add_argument('--carefl_nl', required=False, type=int, default=4)
    parser.add_argument('--carefl_nh', required=False, type=int, default=5)
    parser.add_argument('--carefl_prior_dist', required=False, type=str,
                        choices=['laplace', 'gaussian', 'None'], default='laplace')
    parser.add_argument('--fix_AffineCL_forward', required=False, type=int, choices=[0, 1], default=1)

    parser.add_argument('--carefl_weight_decay', required=False, type=float, default=0.000)

    parser.add_argument('--carefl_epochs', required=False, type=int, default=750)
    parser.add_argument('--batch_size', required=False, type=int, choices=[-1, 128], default=-1,
                        help="if -1, meaning all data in one batch, which is faster.")
    # 0.8: training_set = 0.8 * whole_set, testing_set = 0.2 * whole_set.
    # 1.0: training_set = testing_set = whole_set
    parser.add_argument('--split', required=True, type=float, choices=[0.8, 1.0])

    parser.add_argument('--result_folder', required=False, type=str, default="temp_results")
    parser.add_argument('--result_folder_suffix', required=False, type=str, default="")

    # since the data size is not too big, it looks like cpu is faster than cuda
    parser.add_argument('--device', required=False, type=str, choices=['cpu'], default='cpu')

    args = parser.parse_args()

    result_folder_root = "./{}_{}/".format(args.result_folder, args.result_folder_suffix)

    if args.method_to_test == "simulated":
        result_summary_file = result_folder_root + "file_all_results_{}_{}_{}_{}.csv".format(
            args.SEM, args.noise, args.g_magnitude, args.result_folder_suffix)
    else:
        result_summary_file = result_folder_root + "file_all_results_{}.csv".format(args.result_folder_suffix)

    run_multiple_times(args, result_folder_root, result_summary_file)
