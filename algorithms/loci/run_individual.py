from os.path import exists
import numpy as np
import pandas as pd
import torch
from torch import nn
import gin
import argparse
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, SplineTransformer
from sklearn.linear_model import LinearRegression

from causa.datasets import AN, LS, MNU, SIMG, ANs, CausalDataset, Tuebingen, SIM, LSs
from causa.ml import map_optimization
from causa.heci import HECI
from causa.hsic import HSIC
from causa.het_ridge import convex_fgls
from causa.utils import TensorDataLoader

from Utilities.util_functions import call_dHSIC_from_R

import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def build_network(in_dim=1, out_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, out_dim)
    )


class HetSpindlyHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(1, 1, bias=False)
        self.lin2 = nn.Linear(1, 1, bias=False)
        self.lin2.weight.data.fill_(0.0)

    def forward(self, input):
        out1 = self.lin1(input[:, 0].unsqueeze(-1))
        out2 = torch.exp(self.lin2(input[:, 1].unsqueeze(-1)))
        return torch.cat([out1, out2], 1)


def build_het_network(in_dim=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 2),
        HetSpindlyHead()
    )


def test_indep_fgls(Phi, Psi, x, y_true, w_1, w_2, dHSIC, dHSIC_kernels, convex=False):
    eta_1 = Phi @ w_1
    if convex:
        eta_2 = - torch.abs(Psi) @ w_2
    else:
        eta_2 = - 0.5 * torch.exp(Psi @ w_2)
    scale = torch.sqrt(- 0.5 / eta_2)
    loc = - 0.5 * eta_1 / eta_2
    residuals = (y_true.flatten() - loc) / scale
    # dhsic_res = HSIC(residuals.flatten().cpu().numpy(), x)
    # return dhsic_res
    results_R = call_dHSIC_from_R(dHSIC, residuals.flatten().cpu().numpy(), x, dHSIC_kernels)
    dHSIC_value = results_R.rx2("dHSIC")[0]
    return dHSIC_value


def test_indep_nn(model, x, y_true, dHSIC, dHSIC_kernels, mode='homo', nu_noise=None):
    # modes: 'homo', 'het', 'het_noexp'
    y_true = y_true.flatten()
    with torch.no_grad():
        f = model(x)
    if mode == 'homo':
        residuals = (f.flatten() - y_true)
    else:
        if mode == 'het':
            eta_2 = nu_noise * f[:, 1]
        elif mode == 'het_noexp':
            eta_2 = - 0.5 * torch.exp(f[:, 1])
        scale = torch.sqrt(- 0.5 / eta_2)
        loc = - 0.5 * f[:, 0] / eta_2
        residuals = (y_true - loc) / scale
    # return HSIC(residuals.cpu().flatten().numpy(), x.cpu().flatten().numpy())
    results_R = call_dHSIC_from_R(dHSIC, residuals.cpu().flatten().numpy(), x.cpu().flatten().numpy(), dHSIC_kernels)
    dHSIC_value = results_R.rx2("dHSIC")[0]
    return dHSIC_value


@gin.configurable
def experiment(experiment_name,
               pair_id: int,
               benchmark: CausalDataset = gin.REQUIRED,
               device: str = 'cpu',
               double: bool = True,
               seed=711,
               result_dir: str = 'results',
               skip_baselines=True):
    if exists(f'{result_dir}/{experiment_name}_{pair_id}.csv'):
        print('Run already completed. Aborting...')
        exit()

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    map_kwargs = dict(
        scheduler='cos',
        lr=1e-2,
        lr_min=1e-6,
        n_epochs=5000,
        nu_noise_init=0.5,
    )
    print(f'Experiment name: {experiment_name} pair {pair_id}')
    df = pd.DataFrame(index=[pair_id])
    df.index.name = 'pair_id'
    dataset = benchmark(pair_id, double=double)
    if double:
        torch.set_default_dtype(torch.float64)

    n_data = dataset.cause.shape[0]
    if dataset.cause.shape[1] > 1 or dataset.effect.shape[1] > 1:
        print('Skip dataset with pair_id', pair_id, 'because of multidimensionality')
        exit()
    df.loc[pair_id, 'n_data'] = int(n_data)

    # True direction
    d, k = dataset.cause.shape[1], dataset.effect.shape[1]
    # in benchmark(), it always set dataset.cause as the true cause and dataset.effect as the true effect
    df.loc[pair_id, 'Ground_Truth_Direction'] = "X1->X2"

    batch_size = len(dataset.cause)
    print(f'd={d}, k={k}, n={batch_size}')

    # NOTE: all scores such that higher is better (log ml, log lik) gives score!

    x_true = dataset.cause.numpy().flatten()
    x_false = dataset.effect.numpy().flatten()

    if not skip_baselines:
        # HECI baseline
        heci_cause = x_true.tolist()
        heci_effect = x_false.tolist()
        _, score_true, score_false = HECI(heci_cause, heci_effect)
        df.loc[pair_id, 'heci_X1->X2'] = -score_true
        df.loc[pair_id, 'heci_X2->X1'] = -score_false

    # run linear models with spline feature maps
    spline_trans = SplineTransformer(n_knots=25, degree=5)
    Phi_true = spline_trans.fit_transform(dataset.cause.numpy())
    y_true = dataset.effect.numpy().flatten()
    Phi_false = spline_trans.fit_transform(dataset.effect.numpy())
    y_false = dataset.cause.numpy().flatten()

    if not skip_baselines:
        # Linear Regression (ML) baseline
        model = LinearRegression()
        model.fit(Phi_true, y_true)
        y_pred = torch.from_numpy(model.predict(Phi_true)).flatten()
        lik = torch.distributions.Normal(loc=y_pred, scale=torch.ones_like(y_pred))
        df.loc[pair_id, 'LinReg_ANM_X1->X2'] = lik.log_prob(dataset.effect.flatten()).mean().item()
        residuals_true = (y_pred.numpy() - y_true).flatten()
        df.loc[pair_id, 'LinReg_ANM_HSIC_X1->X2'] = HSIC(residuals_true, x_true)

        model.fit(Phi_false, y_false)
        y_pred = torch.from_numpy(model.predict(Phi_false)).flatten()
        lik = torch.distributions.Normal(loc=y_pred, scale=torch.ones_like(y_pred))
        df.loc[pair_id, 'LinReg_ANM_X2->X1'] = lik.log_prob(dataset.cause.flatten()).mean().item()
        residuals_false = (y_pred.numpy() - y_false).flatten()
        df.loc[pair_id, 'LinReg_ANM_HSIC_X2->X1'] = HSIC(residuals_false, x_false)

    # Convex Heteroscedastic Linear Regression (ML) LSNM estimator
    run_LinReg_Convex_LSNM(df, pair_id, device, dataset.cause, dataset.effect, Phi_true, Phi_false, dHSIC,
                           dHSIC_kernels)

    # Neural Network estimators

    ## Homoscedastic

    # Forward direction
    x_true, y_true = dataset.cause.to(device), dataset.effect.to(device)

    if not skip_baselines:
        loader_ordered = TensorDataLoader(
            dataset.cause.to(device), dataset.effect.to(device), batch_size=batch_size
        )
        set_seed(seed)
        model, losses, _, _, _ = map_optimization(
            build_network(d, k).to(device),
            loader_ordered,
            likelihood='regression',
            prior_prec=0.0,
            **map_kwargs
        )
        df.loc[pair_id, 'NN_ANM_X1->X2'] = - np.nanmin(losses) / n_data
        df.loc[pair_id, 'NN_ANM_HSIC_X1->X2'] = test_indep_nn(model, x_true, y_true, mode='homo')

    # Backward direction
    x_false, y_false = dataset.effect.to(device), dataset.cause.to(device)

    if not skip_baselines:
        loader_reversed = TensorDataLoader(
            dataset.effect.to(device), dataset.cause.to(device), batch_size=batch_size
        )
        set_seed(seed)
        model, losses, _, _, _ = map_optimization(
            build_network(d, k).to(device),
            loader_reversed,
            likelihood='regression',
            prior_prec=0.0,
            **map_kwargs
        )
        df.loc[pair_id, 'NN_ANM_X2->X1'] = - np.nanmin(losses) / n_data
        df.loc[pair_id, 'NN_ANM_HSIC_X2->X1'] = test_indep_nn(model, x_false, y_false, mode='homo')

    ## Heteroscedastic
    run_NN_LSNM(df, pair_id, device, seed, batch_size, dataset.cause, dataset.effect, n_data, d, dHSIC, dHSIC_kernels,
                map_kwargs)

    infer_direction_results(df, pair_id)

    print(df.loc[pair_id])

    df.to_csv(f'{result_dir}/{experiment_name}_{pair_id}.csv')


def run_LinReg_Convex_LSNM(df, pair_id, device, X1, X2, Phi_X1, Phi_X2, dHSIC, dHSIC_kernels):
    # convert to torch
    Phi_X1 = torch.from_numpy(Phi_X1).to(device)
    # y_true = dataset.effect.to(device)
    Phi_X2 = torch.from_numpy(Phi_X2).to(device)
    # y_false = dataset.cause.to(device)

    # X1 -> X2

    w_1, w_2, _, loglik = convex_fgls(Phi_X1, Phi_X1.abs(), X2.to(device), delta_Phi=1e-5, delta_Psi=1e-5)
    df.loc[pair_id, 'LinReg_Convex_LSNM_X1->X2'] = - loglik
    df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X1->X2'] = test_indep_fgls(
        Phi_X1, Phi_X1, X1.numpy().flatten(), X2.to(device), w_1, w_2, dHSIC, dHSIC_kernels, convex=True)

    # X2 -> X1

    w_1, w_2, _, loglik = convex_fgls(Phi_X2, Phi_X2.abs(), X1, delta_Phi=1e-5, delta_Psi=1e-5)
    df.loc[pair_id, 'LinReg_Convex_LSNM_X2->X1'] = - loglik
    df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X2->X1'] = test_indep_fgls(
        Phi_X2, Phi_X2, X2.numpy().flatten(), X1.to(device), w_1, w_2, dHSIC, dHSIC_kernels, convex=True)


def run_NN_LSNM(df, pair_id, device, seed, batch_size, X1, X2, n_data, d, dHSIC, dHSIC_kernels, map_kwargs):
    # X1 -> X2

    loader_ordered = TensorDataLoader(
        X1.to(device), X2.to(device).flatten(), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network(d).to(device),
        loader_ordered,
        likelihood='heteroscedastic_regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'NN_LSNM_X1->X2'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'NN_LSNM_HSIC_X1->X2'] = test_indep_nn(
        model, X1.to(device), X2.to(device), dHSIC, dHSIC_kernels, mode='het', nu_noise=-0.5)

    # X2 -> X1

    loader_reversed = TensorDataLoader(
        X2.to(device), X1.to(device).flatten(), batch_size=batch_size
    )
    set_seed(seed)
    model, losses, _, _, _ = map_optimization(
        build_het_network(d).to(device),
        loader_reversed,
        likelihood='heteroscedastic_regression',
        prior_prec=0.0,
        **map_kwargs
    )
    df.loc[pair_id, 'NN_LSNM_X2->X1'] = - np.nanmin(losses) / n_data
    df.loc[pair_id, 'NN_LSNM_HSIC_X2->X1'] = test_indep_nn(
        model, X2.to(device), X1.to(device), dHSIC, dHSIC_kernels, mode='het', nu_noise=-0.5)


def infer_direction_results(df, pair_id):
    # The non-HSIC methods give likelihood or negative loss, so the bigger the better
    # It looks like the HSIC methods give HSIC values instead of p-values, so the smaller the better (more independent)

    if df.loc[pair_id, 'LinReg_Convex_LSNM_X1->X2'] > df.loc[pair_id, 'LinReg_Convex_LSNM_X2->X1']:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM'] = "X1->X2"
    elif df.loc[pair_id, 'LinReg_Convex_LSNM_X1->X2'] < df.loc[pair_id, 'LinReg_Convex_LSNM_X2->X1']:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM'] = "X2->X1"
    else:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM'] = "no conclusion"

    if df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X1->X2'] < df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X2->X1']:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM_HSIC'] = "X1->X2"
    elif df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X1->X2'] > df.loc[pair_id, 'LinReg_Convex_LSNM_HSIC_X2->X1']:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM_HSIC'] = "X2->X1"
    else:
        df.loc[pair_id, 'Result: LinReg_Convex_LSNM_HSIC'] = "no conclusion"

    if df.loc[pair_id, 'NN_LSNM_X1->X2'] > df.loc[pair_id, 'NN_LSNM_X2->X1']:
        df.loc[pair_id, 'Result: NN_LSNM'] = "X1->X2"
    elif df.loc[pair_id, 'NN_LSNM_X1->X2'] < df.loc[pair_id, 'NN_LSNM_X2->X1']:
        df.loc[pair_id, 'Result: NN_LSNM'] = "X2->X1"
    else:
        df.loc[pair_id, 'Result: NN_LSNM'] = "no conclusion"

    if df.loc[pair_id, 'NN_LSNM_HSIC_X1->X2'] < df.loc[pair_id, 'NN_LSNM_HSIC_X2->X1']:
        df.loc[pair_id, 'Result: NN_LSNM_HSIC'] = "X1->X2"
    elif df.loc[pair_id, 'NN_LSNM_HSIC_X1->X2'] > df.loc[pair_id, 'NN_LSNM_HSIC_X2->X1']:
        df.loc[pair_id, 'Result: NN_LSNM_HSIC'] = "X2->X1"
    else:
        df.loc[pair_id, 'Result: NN_LSNM_HSIC'] = "no conclusion"


if __name__ == '__main__':
    # Install the dHSIC package
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # choose a CRAN mirror
    utils.install_packages("dHSIC")  # https://cran.r-project.org/web/packages/dHSIC/dHSIC.pdf
    dHSIC = importr('dHSIC')  # Load the dHSIC package, i.e. ```library("dHSIC")``` in R

    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_id', type=int, help='pair id to run')
    parser.add_argument('--config', type=str, help='gin-config for the run.')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--dHSIC_kernel1', required=False, type=str, choices=["gaussian", "discrete"],
                        default="gaussian")
    parser.add_argument('--dHSIC_kernel2', required=False, type=str, choices=["gaussian", "discrete"],
                        default="gaussian")
    args = parser.parse_args()

    dHSIC_kernel1 = args.dHSIC_kernel1
    dHSIC_kernel2 = args.dHSIC_kernel2
    dHSIC_kernels = [dHSIC_kernel1, dHSIC_kernel2]

    gin.external_configurable(StandardScaler)
    gin.external_configurable(MinMaxScaler)
    gin.parse_config_file(args.config)
    experiment(experiment_name=args.config.split('/')[-1].split('.')[0],
               pair_id=args.pair_id, double=True,
               result_dir=args.result_dir)
