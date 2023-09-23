import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../algorithms/carefl"))
sys.path.append(os.path.abspath("../algorithms/loci"))

from Utilities.data_generator import SEM_Functionals, SEM_Noises
from Utilities.util_functions import set_random_seed, append_to_file, normalize_data, set_random_seed_R
from CAREFL_H.main_carefl_comparison_multiple_times import generate_DAG_and_data, create_carefl_model, \
    compute_GT_mean_conditional_variance


def run(B_true, X, Z, target_direction, n):
    print()
    append_to_file(result_file, "\n")

    normalized_X = normalize_data(X)
    X = None

    true_residuals = Z

    if np.any(B_true != np.array([[0, 0], [1, 0]])):
        raise Exception("B_true is {}.".format(B_true))

    result_folder_method = result_folder_root + "CAREFL/N={}/".format(n)

    if target_direction == "anti-causal":
        #################################################################################
        # Fit a model in the direction of X1->X2
        #################################################################################

        direction = "X1->X2"

    elif target_direction == "causal":
        #################################################################################
        # Fit the CAREFL model in the causal direction of X2->X1
        #################################################################################

        direction = "X2->X1"

    result_folder_direction = os.path.join(result_folder_method, "{}".format(direction))
    os.makedirs(result_folder_direction, exist_ok=True)

    compute_GT_mean_conditional_variance(B_true, normalized_X, result_folder_direction, None)

    n = normalized_X.shape[0]
    print("n: ", n)
    append_to_file(result_file, "\nn: {}\n".format(n))

    carefl_model = create_carefl_model(split, True if fix_AffineCL_forward else False, updated_batch_size,
                                       result_folder_method, carefl_nl, carefl_nh, carefl_prior_dist, carefl_epochs,
                                       "cpu", weight_decay, "affine")

    train_set, _, test_dset_numpy, dim, train_dset_numpy = carefl_model._get_datasets(normalized_X)
    print("training set shape: ", train_dset_numpy.shape)
    print("testing set shape: ", test_dset_numpy.shape)
    append_to_file(result_file, "\ntraining set shape: {}\n".format(train_dset_numpy.shape))
    append_to_file(result_file, "\ntesting set shape: {}\n".format(test_dset_numpy.shape))

    carefl_model.dim = dim

    torch.manual_seed(carefl_model.config.training.seed)

    print("Start training the CAREFL model in the direction {}...".format(direction))
    append_to_file(result_file, "\nStart training the CAREFL model in the direction {}...\n".format(direction))

    # fit the model in the provided direction
    if direction == "X1->X2":
        # Conditional Flow Model: X->Y
        trained_flows, losses, _, _, _, _ = carefl_model._train(train_set)

    elif direction == "X2->X1":
        # Conditional Flow Model: Y->X
        trained_flows, losses, _, _, _, _ = carefl_model._train(train_set, parity=True)

    else:
        raise Exception("What??? direction={}".format(direction))

    if len(trained_flows) != 1:
        raise Exception("What??? len(trained_flows)={}".format(len(trained_flows)))

    if len(losses) != 1:
        raise Exception("What??? len(losses)={}".format(len(losses)))

    training_loss = losses[0]
    training_loss_log = os.path.join(result_folder_direction, 'carefl_training_loss_log.txt')
    # to be consistent with notears, save the average loss, not summed loss
    np.savetxt(training_loss_log, np.array(training_loss) / n, delimiter=',', fmt='%f')

    carefl_model.flow = trained_flows[0]

    with torch.no_grad():
        print("true_residuals.shape: ", true_residuals.shape)
        append_to_file(result_file, "\ntrue_residuals.shape: {}\n".format(true_residuals.shape))

        predicted_residuals = carefl_model._forward_flow(normalized_X)  # against both training and testing sets

        print("predicted_residuals.shape: ", predicted_residuals.shape)
        append_to_file(result_file, "\npredicted_residuals.shape: {}\n".format(predicted_residuals.shape))

        suitable_value = np.mean(1 / n * np.power(predicted_residuals - true_residuals, 2), axis=0)
        print("suitable_value for (Z1, Z2): ", suitable_value)
        append_to_file(result_file, "\nsuitable_value for (Z1, Z2): {}\n".format(suitable_value))


if __name__ == "__main__":
    """
    Answer the question: Is CAREFL under prior misspecification suitable? 
    
    See Definition 18 of the Mooij benchmark paper.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--SEM', required=False, type=str, default="any")
    parser.add_argument('--noise', required=False, type=str, default="any")

    parser.add_argument('--result_folder', required=False, type=str, default="temp_results/suitability/")

    args = parser.parse_args()

    np.set_printoptions(precision=5, suppress=True)

    seed = 1
    set_random_seed(seed)
    set_random_seed_R(seed)

    result_folder_root = args.result_folder
    os.makedirs(result_folder_root, exist_ok=True)

    result_file = os.path.join(result_folder_root, 'result.txt')

    D = 2
    EXPECTED_EDGES = 1

    variable_names = ['X{}'.format(j) for j in range(1, D + 1)]

    normalize_datasets = 1

    random_SEM = args.SEM
    random_noise = args.noise
    print('SEM: ', random_SEM)
    print('noise: ', random_noise)
    print('result_folder_root: ', result_folder_root)

    if random_noise == SEM_Noises.uniform:
        gt_noise_type = SEM_Noises.uniform
    elif random_noise == SEM_Noises.exp:
        gt_noise_type = SEM_Noises.exp
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
    else:
        raise Exception("What??? SEM: {}".format(random_SEM))

    g_magnitude = 1

    ############################################
    # Default CAREFL Hyperparameters
    ############################################
    # "In the data splitting case, any regression method that is weakly consistent for P(X,Y) is suitable."
    # In the paper, it uses the same N for both training set and testing set.
    split = 0.5

    fix_AffineCL_forward = 1
    updated_batch_size = 128
    carefl_nl = 4
    carefl_nh = 5
    carefl_prior_dist = 'laplace'
    carefl_epochs = 750
    weight_decay = 0

    ############################################
    # Generate DAG and data
    ############################################

    n_max = 50000

    print("\nStarting ground-truth SEM {} with ground-truth noise {}...".format(gt_sem_type, gt_noise_type))
    append_to_file(result_file, "\nStarting ground-truth SEM {} with ground-truth noise {}...\n\n"
                   .format(gt_sem_type, gt_noise_type))

    all_X, all_true_residuals, B_true = generate_DAG_and_data(gt_sem_type, gt_noise_type,
                                                              result_folder_root, D,
                                                              EXPECTED_EDGES, variable_names,
                                                              normalize_datasets, n_max, g_magnitude)

    if np.all(B_true == np.array([[0, 0], [1, 0]])):
        print("\nCausal Direction: X2->X1")
        print("Anti-Causal Direction: X1->X2\n")
        append_to_file(result_file, "\nCausal Direction: X2->X1\n")
        append_to_file(result_file, "\nAnti-Causal Direction: X1->X2\n")
    elif np.all(B_true == np.array([[0, 1], [0, 0]])):
        print("\nCausal Direction: X1->X2")
        print("Anti-Causal Direction: X2->X1\n")
        append_to_file(result_file, "\nCausal Direction: X1->X2\n")
        append_to_file(result_file, "\nAnti-Causal Direction: X2->X1\n")
    else:
        raise Exception("What??? B_true: {}".format(B_true))

    # increasing N and see if the suitable_value approaches 0.
    for n in [100, 1000, 2000, 10000]:
        X = np.genfromtxt(result_folder_root + "X.csv", delimiter=',', max_rows=n)
        Z = np.genfromtxt(result_folder_root + "Z.csv", delimiter=',', max_rows=n)
        run(B_true, X, Z, "causal", n)

    # # increasing N and see if the suitable_value approaches 0.
    # for n in [100, 1000, 2000, 10000]:
    #     X = np.genfromtxt(result_folder_root + "X.csv", delimiter=',', max_rows=n)
    #     Z = np.genfromtxt(result_folder_root + "Z.csv", delimiter=',', max_rows=n)
    #     run(B_true, X, Z, "anti-causal", n)
