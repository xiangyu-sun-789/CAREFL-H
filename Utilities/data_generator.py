import os
import numpy as np
import igraph as ig
import pandas as pd
import torch
from scipy.special import expit as sigmoid  # same as notears
from Utilities.util_functions import is_dag, draw_DAGs_using_LINGAM


class SEM_Noises:
    standard_gaussian = "standard-gaussian"
    gaussian0_4 = "gaussian0_4"
    gaussian0_25 = "gaussian0_25"
    gaussian0_100 = "gaussian0_100"
    gaussian0_400 = "gaussian0_400"
    uniform = "uniform"
    exp = "exp"
    laplace = "laplace"
    beta0505 = "beta0505"
    continuous_bernoulli = "continuous-bernoulli"


class SEM_Functionals:
    LSNM_tanh_exp_cosine = "LSNM-tanh-exp-cosine"
    LSNM_sine_tanh = "LSNM-sine-tanh"
    LSNM_sigmoid_sigmoid = "LSNM-sigmoid-sigmoid"
    ANM_tanh = "ANM-tanh"
    ANM_sine = "ANM-sine"
    ANM_sigmoid = "ANM-sigmoid"


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_linear_parameters(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate linear SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


# def simulate_linear_sem(B, n, noise_type, noise_scale=None):
#     """Simulate samples from linear SEM with specified type of noise.
#
#     For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
#
#     Args:
#         W (np.ndarray): [d, d] weighted adj matrix of DAG
#         n (int): num of samples, n=inf mimics population risk
#         noise_type (str): gauss, exp, gumbel, uniform, logistic, poisson
#         noise_scale (np.ndarray): scale parameter of additive noise, default all ones
#
#     Returns:
#         X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
#         W: the simulated DAG with adjacency weights
#     """
#
#     def _simulate_single_equation(X, w, scale):
#         """X: [n, num of parents], w: [num of parents], x: [n]"""
#         if noise_type == SEM_Noises.standard_gaussian:
#             z = np.random.normal(size=n)
#             x = X @ w + z
#
#         elif noise_type == SEM_Noises.uniform:
#             z = np.random.uniform(low=-1 * scale, high=scale, size=n)
#             x = X @ w + z
#
#         elif noise_type == SEM_Noises.laplace:
#             z = np.random.laplace(loc=0.0, scale=1.0, size=n)
#             x = X @ w + z
#
#         elif noise_type == SEM_Noises.exp:
#             # on Wikipedia, lambda is the inverse scale, so lambda = 1/scale, scale = 1/lambda
#             lambdaa = 1
#             z = np.random.exponential(scale=1.0 / lambdaa, size=n)
#             x = X @ w + z
#         else:
#             raise ValueError('unknown noise type')
#         return x, z
#
#     W = simulate_linear_parameters(B)
#
#     d = W.shape[0]
#     if noise_scale is None:
#         scale_vec = np.ones(d)
#     elif np.isscalar(noise_scale):
#         scale_vec = noise_scale * np.ones(d)
#     else:
#         if len(noise_scale) != d:
#             raise ValueError('noise scale must be a scalar or has length d')
#         scale_vec = noise_scale
#     if not is_dag(W):
#         raise ValueError('W must be a DAG')
#     if np.isinf(n):  # population risk for linear gauss SEM
#         if noise_type == 'gauss':
#             # make 1/d X'X = true cov
#             X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
#             return X
#         else:
#             raise ValueError('population risk not available')
#     # empirical risk
#     G = ig.Graph.Weighted_Adjacency(W.tolist())
#     ordered_vertices = G.topological_sorting()
#     assert len(ordered_vertices) == d
#     X = np.zeros([n, d])
#     Z = np.zeros([n, d])
#     for j in ordered_vertices:
#         parents = G.neighbors(j, mode=ig.IN)
#         X[:, j], Z[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
#     return X, Z, None, W


def simulate_nonlinear_sem(B, n, sem_type, noise_type, noise_scale=None, g_magnitude=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    def _simulate_single_equation(X_parents, scale):
        """
        X_parents: [n, num of parents],
        x: [n]
        """

        if noise_type == SEM_Noises.standard_gaussian:
            z = np.random.normal(loc=0.0, scale=1.0, size=n)  # N(0, 1)

        elif noise_type == SEM_Noises.gaussian0_4:
            z = np.random.normal(loc=0.0, scale=2.0, size=n)

        elif noise_type == SEM_Noises.gaussian0_25:
            z = np.random.normal(loc=0.0, scale=5.0, size=n)

        elif noise_type == SEM_Noises.gaussian0_100:
            z = np.random.normal(loc=0.0, scale=10.0, size=n)

        elif noise_type == SEM_Noises.gaussian0_400:
            z = np.random.normal(loc=0.0, scale=20.0, size=n)

        elif noise_type == SEM_Noises.uniform:
            z = np.random.uniform(low=-1 * scale, high=scale, size=n)

        elif noise_type == SEM_Noises.exp:
            # on Wikipedia, lambda is the inverse scale, so lambda = 1/scale, scale = 1/lambda
            lambdaa = 1
            z = np.random.exponential(scale=1.0 / lambdaa, size=n)

        elif noise_type == SEM_Noises.laplace:
            z = np.random.laplace(loc=0.0, scale=1.0, size=n)

        elif noise_type == SEM_Noises.beta0505:
            z = np.random.beta(a=0.5, b=0.5, size=n)

        elif noise_type == SEM_Noises.continuous_bernoulli:
            cb = torch.distributions.continuous_bernoulli.ContinuousBernoulli(0.9)
            z = cb.sample(torch.Size([n])).numpy()

        else:
            raise Exception("What??? Invalid noise type {}".format(noise_type))

        # generate values for the source nodes
        pa_size = X_parents.shape[1]
        if pa_size == 0:
            return z, z

        # generate values for other nodes given their parent nodes
        # ```@``` is matrix dot product
        # ```*``` is matrix element-wise multiplication
        if sem_type == SEM_Functionals.LSNM_tanh_exp_cosine:
            # See paper https://arxiv.org/abs/2011.02268
            # Affine flow: equation 5
            # Identifiability: theorem 1
            # if "gaussian" not in noise_type or B.shape[0] != 2:
            #     raise Exception("Affine model is proven to be identifiable with Gaussian noise in the bivariate case.")

            hidden = 1
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1
            # x2 = e^{s2(x1)} * z2 + t2(x1), if s()=0, then it is SEM_Nonlinear_Functions.ANM_tanh
            # s() is cosine
            # t() is tanh (invertible)
            g = g_magnitude * np.power(np.e, np.cos(X_parents @ W1) @ W2)
            x = g * z + np.tanh(X_parents @ W3) @ W4

        elif sem_type == SEM_Functionals.LSNM_sine_tanh:
            # Y = f(X) + g(X) * Z
            # g() must be positive

            hidden = 1
            W1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1
            W2 = np.random.uniform(low=1.0, high=2.0, size=pa_size)

            # if s()=1, then it is SEM_Nonlinear_Functions.ANM_sine
            # s() = tanh + U[1,2] is positive,
            # f() is sine

            g = g_magnitude * (np.tanh(X_parents @ W1) + W2)
            x = np.sin(X_parents @ W3) @ W4 + g * z

            if np.any(g <= 0):
                raise Exception("g(X) in LSNM must be positive.")

        elif sem_type == SEM_Functionals.LSNM_sigmoid_sigmoid:
            # Y = f(X) + g(X) * Z
            # g() must be positive

            hidden = 1
            W1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1

            # if s()=1, then it is SEM_Nonlinear_Functions.ANM_sine
            # s() = sigmoid is positive,
            # f() is sine

            g = g_magnitude * sigmoid(X_parents @ W1)
            x = sigmoid(X_parents @ W3) @ W4 + g * z

            if np.any(g <= 0):
                raise Exception("g(X) in LSNM must be positive.")

        elif sem_type == SEM_Functionals.ANM_tanh:
            hidden = 1
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1
            x = g_magnitude * z + np.tanh(X_parents @ W3) @ W4

        elif sem_type == SEM_Functionals.ANM_sine:
            hidden = 1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1
            x = np.sin(X_parents @ W3) @ W4 + g_magnitude * z

        elif sem_type == SEM_Functionals.ANM_sigmoid:
            hidden = 1
            W3 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W3[np.random.rand(*W3.shape) < 0.5] *= -1
            W4 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W4[np.random.rand(hidden) < 0.5] *= -1

            x = sigmoid(X_parents @ W3) @ W4 + g_magnitude * z

        else:
            raise ValueError('unknown sem type')

        if "LSNM" in sem_type:
            print("max g(): ", g.max())
            print("min g(): ", g.min())
            print("mean g(): ", g.mean())

        return x, z

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    Z = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()

    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j], Z[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X, Z


def plot_ground_truth_independent_noise_model(file_name, B_true, variable_names):
    # in NOTEARS, the W is defined as row variable -> column variable

    D_X = len(variable_names)

    variable_names_plus_z = [str(X_names).replace("X", "Z").replace("x", "z") for X_names in
                             variable_names] + variable_names

    Zs = np.hstack((np.zeros((D_X, D_X)), np.eye(D_X)))  # Zi -> Xi
    Xs = np.hstack((np.zeros((D_X, D_X)), B_true))
    B_true_plus_z = np.vstack((Zs, Xs))

    draw_DAGs_using_LINGAM(file_name, B_true_plus_z, variable_names_plus_z)


def plot_ground_truth_confounder_noise_model(file_name, B_true, variable_names):
    # a confounder noise that goes to every X

    # in NOTEARS, the W is defined as row variable -> column variable

    D_X = len(variable_names)

    variable_names_plus_z = ["Z_confounder"] + [str(X_names).replace("X", "Z").replace("x", "z") for X_names in
                                                variable_names] + variable_names

    independent_Zs = np.hstack((np.zeros((D_X, D_X)), np.eye(D_X)))  # Zi -> Xi
    Xs = np.hstack((np.zeros((D_X, D_X)), B_true))
    B_true_plus_independent_Zs = np.vstack((independent_Zs, Xs))

    # add Z_confounder -> every_Xi
    confounderZ_to_X = [0] * (len(independent_Zs) + 1) + [1] * len(Xs)
    B_true_plus_z = np.vstack((np.array(confounderZ_to_X), np.hstack(
        (np.array([[0] for _ in range(len(independent_Zs) + len(Xs))]), B_true_plus_independent_Zs))))

    draw_DAGs_using_LINGAM(file_name, B_true_plus_z, variable_names_plus_z)


def process_description_text(text):
    # the description file of the cause effect pairs data are not very well organized. Remove all the whitespaces.
    return text.lower().replace(":", "").replace(" ", "").replace("\t", "").replace("\n", "")


def is_bivariate_Tubingen_dataset(directory, pair_id):
    meta = pd.read_csv(directory + '/pairmeta.txt', delim_whitespace=True,
                       header=None,
                       names=['id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                       index_col=0).astype(float)

    if meta.loc[pair_id, 'cause_start'] != meta.loc[pair_id, 'cause_end']:
        return False

    if meta.loc[pair_id, 'effect_start'] != meta.loc[pair_id, 'effect_end']:
        return False

    return True


def read_Tubingen_meta_file(directory, pair_id):
    meta = pd.read_csv(directory + '/pairmeta.txt', delim_whitespace=True,
                       header=None,
                       names=['id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                       index_col=0).astype(float)

    if meta.loc[pair_id, 'cause_start'] != meta.loc[pair_id, 'cause_end']:
        raise Exception("What??? pair_id={}, cause_start {} != cause_end {}."
                        .format(pair_id, meta.loc[pair_id, 'cause_start'], meta.loc[pair_id, 'cause_end']))

    if meta.loc[pair_id, 'effect_start'] != meta.loc[pair_id, 'effect_end']:
        raise Exception("What??? pair_id={}, effect_start {} != effect_end {}."
                        .format(pair_id, meta.loc[pair_id, 'effect_start'], meta.loc[pair_id, 'effect_end']))

    cause_index = meta.loc[pair_id, 'cause_start']
    effect_index = meta.loc[pair_id, 'effect_start']
    weight = meta.loc[pair_id, 'weight']

    return meta, cause_index, effect_index, weight


def read_Tubingen_data_file(directory, training_file):
    # copied and adapted from LOCI

    pair_id = int(training_file.replace("pair", "").replace(".txt", ""))

    df = pd.read_csv(directory + f'/pair{pair_id:04d}.txt', delim_whitespace=True, header=None)

    X = df.to_numpy()

    meta, cause_index, effect_index, dataset_weight = read_Tubingen_meta_file(directory, pair_id)

    if cause_index == 1 and effect_index == 2:
        # "X1->X2"
        B_true = np.array([[0, 1], [0, 0]])
        plot_title = "ground-truth direction: X1 -> X2"

    elif cause_index == 2 and effect_index == 1:
        # "X2->X1"
        B_true = np.array([[0, 0], [1, 0]])
        plot_title = "ground-truth direction: X2 -> X1"

    else:
        raise Exception("What??? cause_start: {}, cause_end: {}, ".
                        format(meta.loc[pair_id, 'cause_start'], meta.loc[pair_id, 'cause_end']))

    return X, B_true, plot_title, dataset_weight


if __name__ == "__main__":

    d, n, sem_type = 2, 2000, 'mim'

    # ----- generate random data to create datasets -----
    result_folder = './results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # true direction: X1 -> X2
    B_true = np.array([[0, 1], [0, 0]])
    np.savetxt(os.path.join(result_folder, 'W_true.csv'), B_true, delimiter=',')

    variable_names = ['X{}'.format(j) for j in range(1, d + 1)]
    draw_DAGs_using_LINGAM(os.path.join(result_folder, "W_true_DAG"), B_true, variable_names)

    X = simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt(os.path.join(result_folder, 'X.csv'), X, delimiter=',')
