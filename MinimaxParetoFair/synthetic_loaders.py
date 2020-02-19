import pandas as pd
import numpy as np
from . import *
import sys
sys.path.append(".")
sys.path.append("..")
from MinimaxParetoFair.network import *
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def toy_dataybin_gen(x_gen, a_gen, input_dic, seed=42):
    y_gen_prob = np.ones(x_gen.shape)
    a_unique = np.unique(a_gen)

    for ix_a in np.arange(a_unique.shape[0]):
        input_opt = input_dic[ix_a]
        a_val = a_unique[ix_a]
        x_a = x_gen[a_gen == a_val]

        pa_y = np.ones(y_gen_prob[a_gen == a_val].shape) * input_opt[2, 0]
        pa_y[x_a >= input_opt[0, 1]] = input_opt[2, 1]

        y_gen_prob[a_gen == a_val] = pa_y
        print(np.unique(pa_y), input_opt[0, 1])

    from scipy.stats import bernoulli
    np.random.seed(seed)
    y_gen = bernoulli.rvs(y_gen_prob)
    return y_gen, y_gen_prob

def toy_datax_gen(mean_a_array, pa_array, n_samples, seed=42):
    np.random.seed(seed)
    x = np.random.randn(n_samples)
    a_cat = np.random.multinomial(1, pa_array, size=n_samples)
    a = np.argmax(a_cat, axis=1)
    x += np.sum(a_cat * mean_a_array[np.newaxis, :], axis=1)

    return x, a, a_cat

def plot_simulations(x_gen, y_gen, a_gen, y_gen_prob):
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    for a_val in np.arange(a_gen.max() + 1):
        ix_a = np.argsort(x_gen[a_gen == a_val])
        u_plot = y_gen[a_gen == a_val]
        x_plot = x_gen[a_gen == a_val]
        plt.plot(x_plot[ix_a], u_plot[ix_a], '.')
    plt.ylabel(r'$u$', fontsize=15)

    plt.subplot(3, 1, 2)
    for a_val in np.arange(a_gen.max() + 1):
        ix_a = np.argsort(x_gen[a_gen == a_val])
        u_plot = y_gen_prob[a_gen == a_val]
        x_plot = x_gen[a_gen == a_val]
        plt.plot(x_plot[ix_a], u_plot[ix_a], '-.')
    plt.ylabel(r'$p(u=1|x)$', fontsize=15)

    plt.subplot(3, 1, 3)
    for a_val in np.arange(a_gen.max() + 1):
        plt.hist(x_gen[a_gen == a_val], 25)
    plt.xlabel(r'$x$', fontsize=15)
    plt.show()

def get_step_function(input_est, input_opt):
    x_i0 = input_est[0, :]
    x_i1 = input_est[1, :]
    value = input_est[2, :]

    xc_i0 = input_opt[0, :]
    xc_i1 = input_opt[1, :]
    valuec = input_opt[2, :]

    x_full0 = np.concatenate([x_i0, xc_i0, xc_i1], axis=0)
    x_full0[x_full0 == np.inf] = -1 * np.inf
    x_full1 = np.concatenate([x_i1, xc_i0, xc_i1], axis=0)
    x_full1[x_full1 == -np.inf] = np.inf
    x_full0 = np.sort(np.unique(x_full0))
    x_full1 = np.sort(np.unique(x_full1))

    ix_ini = x_full0[np.newaxis, :] >= x_i0[:, np.newaxis]
    ix_fin = x_full1[np.newaxis, :] <= x_i1[:, np.newaxis]
    value_full = np.sum(value[:, np.newaxis] * (ix_ini * ix_fin), axis=0)

    ix_ini = x_full0[np.newaxis, :] >= xc_i0[:, np.newaxis]
    ix_fin = x_full1[np.newaxis, :] <= xc_i1[:, np.newaxis]
    valuec_full = np.sum(valuec[:, np.newaxis] * (ix_ini * ix_fin), axis=0)

    output = np.zeros([4, value_full.shape[0]])
    output[0, :] = x_full0
    output[1, :] = x_full1
    output[2, :] = value_full
    output[3, :] = valuec_full

    return output

def bin_onedgauss_risk(output, mean, std):
    from scipy.stats import norm
    coef = (output[3, :]) * (output[2, :] - 1) ** 2 + (output[2, :] ** 2) * (1 - output[3, :])
    CDF1 = norm.cdf(((output[1, :]) - mean) / std)
    CDF0 = norm.cdf(((output[0, :]) - mean) / std)
    return (CDF1 - CDF0) * coef

def simul_gauss3groups(mean_array=[-1, 0, 1], std_array=[1, 1, 1],
                       pa_array=[1 / 3, 1 / 3, 1 / 3], low_rho_array=[0.4, 0.4, 0.4],
                       high_rho_array=[0.9, 0.9, 0.9], transitions_array=[-0.5, 0, 0.5],
                       n_samples=60000, seed=42):
    # to numpy
    mean_array = np.asarray(mean_array)
    std_array = np.asarray(std_array)
    pa_array = np.asarray(pa_array)
    low_rho_array = np.asarray(low_rho_array)
    high_rho_array = np.asarray(high_rho_array)
    transitions_array = np.asarray(transitions_array)
    print(transitions_array)

    input_dic = {}
    y_x = np.zeros([mean_array.shape[0], 2])
    for i in np.arange(y_x.shape[0]):
        y_x[i, :] = np.array([low_rho_array[i], high_rho_array[i]])

        ## input opt
        input_opt = np.zeros([3, 2])
        input_opt[0, :] = np.array([-np.inf, transitions_array[i]])
        input_opt[1, :] = np.array([transitions_array[i], np.inf])
        input_opt[2, :] = y_x[i, :]

        ## input dictionary
        input_dic[i] = input_opt

    ## analize risks vertices
    risks_trades = np.zeros([y_x.shape[0], y_x.shape[0]])
    for i in np.arange(y_x.shape[0]):
        for j in np.arange(y_x.shape[0]):
            output = get_step_function(input_dic[j], input_dic[i])
            risk = bin_onedgauss_risk(output, mean_array[i], std_array[i])
            #         print(risk)
            risks_trades[i, j] = np.sum(risk)
        print('for class ' + str(i) + '; risks tradeoffs ', risks_trades[i, :])

    x_gen, a_gen, a_gen_cat = toy_datax_gen(mean_array, pa_array, n_samples, seed=seed)

    #     for a in np.unique(a_gen):
    #         plt.hist(x_gen[a_gen == a],50)
    #         print(np.mean(x_gen[a_gen == a]))
    #     plt.show()

    y_gen, y_gen_prob = toy_dataybin_gen(x_gen, a_gen, input_dic, seed=seed)
    #plot_simulations(x_gen, y_gen, a_gen, y_gen_prob)

    ## Train, val & test split
    ntrain = 0.6 * n_samples
    nval = 0.2 * n_samples
    ntest = 0.2 * n_samples

    x_train = np.array(x_gen[0:int(ntrain)])
    y_train = np.array(y_gen[0:int(ntrain)])
    a_train = np.array(a_gen[0:int(ntrain)])
    #     plot_simulations(x_train, y_train, a_train, y_gen_prob[0:int(ntrain)])

    x_val = np.array(x_gen[int(ntrain):int(ntrain + nval)])
    y_val = np.array(y_gen[int(ntrain):int(ntrain + nval)])
    a_val = np.array(a_gen[int(ntrain):int(ntrain + nval)])
    #     plot_simulations(x_val, y_val, a_val, y_gen_prob[int(ntrain):int(ntrain+nval)])

    x_test = np.array(x_gen[int(ntrain + nval):])
    y_test = np.array(y_gen[int(ntrain + nval):])
    a_test = np.array(a_gen[int(ntrain + nval):])
    #     plot_simulations(x_test, y_test, a_test, y_gen_prob[int(ntrain+nval):])

    # Dataframes
    # pd_data = np.concatenate([x_gen[:, np.newaxis], y_gen[:, np.newaxis]], axis=1)
    # pd_data = np.concatenate([pd_data, a_gen[:, np.newaxis]], axis=1)
    # pd_data = pd.DataFrame(pd_data, columns=['x', 'u', 's'])

    train_pd = np.concatenate([x_train[:, np.newaxis], y_train[:, np.newaxis]], axis=1)
    train_pd = np.concatenate([train_pd, a_train[:, np.newaxis]], axis=1)
    train_pd = pd.DataFrame(train_pd, columns=['x', 'u', 's'])

    val_pd = np.concatenate([x_val[:, np.newaxis], y_val[:, np.newaxis]], axis=1)
    val_pd = np.concatenate([val_pd, a_val[:, np.newaxis]], axis=1)
    val_pd = pd.DataFrame(val_pd, columns=['x', 'u', 's'])

    test_pd = np.concatenate([x_test[:, np.newaxis], y_test[:, np.newaxis]], axis=1)
    test_pd = np.concatenate([test_pd, a_test[:, np.newaxis]], axis=1)
    test_pd = pd.DataFrame(test_pd, columns=['x', 'u', 's'])

    return train_pd, val_pd, test_pd

def get_dataloaders_gauss3groups(config, sampler=True, secret_tag='s', utility_tag='u', balanced_tag='s',mean_array=[-1, 0, 1], std_array=[1, 1, 1],
                       pa_array=[1 / 3, 1 / 3, 1 / 3], low_rho_array=[0.4, 0.4, 0.4],
                       high_rho_array=[0.9, 0.9, 0.9], transitions_array=[-0.5, 0, 0.5]):
    from .misc import get_weight_dict

    train_pd, val_pd, test_pd = simul_gauss3groups(seed = config.split,mean_array=mean_array, std_array=std_array,
                       pa_array=pa_array, low_rho_array=low_rho_array,
                       high_rho_array=high_rho_array, transitions_array=transitions_array)

    n_utility = train_pd[utility_tag].nunique()
    n_secret = train_pd[secret_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_secret = n_secret  # depends on dataset
    config.size = 10000 # depends on dataset
    config.cov_tags = ['x']

    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 32
    if config.LEARNING_RATE == 0:
        if config.type_loss == 0:
            config.LEARNING_RATE = 5e-4
        else:
            config.LEARNING_RATE = 1e-3
    if config.patience == 0:
        config.patience = 15

    train_pd['secret_cat'] = train_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    test_pd['secret_cat'] = test_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    val_pd['secret_cat'] = val_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))

    train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))

    # get prior of subgroups
    config.p_secret = train_pd['secret_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = None  # Tabular data

    if sampler:

        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=config.cov_tags,
                                                         utility_tag='utility_cat', secret_tag='secret_cat',
                                                         transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd,cov_list = config.cov_tags,
                                                         utility_tag = 'utility_cat', secret_tag = 'secret_cat',
                                                         transform=composed),
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=True, num_workers=config.n_dataloader, pin_memory = True)

    val_dataloader = DataLoader(TablePandasDataset(pd=val_pd, cov_list=config.cov_tags,
                                                   utility_tag='utility_cat', secret_tag='secret_cat',
                                                   transform=composed),
                                batch_size=config.BATCH_SIZE,
                                shuffle=True, num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(TablePandasDataset(pd=test_pd, cov_list=config.cov_tags,
                                                    utility_tag='utility_cat', secret_tag='secret_cat',
                                                    transform=composed),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=config.n_dataloader, pin_memory=True)

    ## NETWORK ##
    torch.manual_seed(config.seed)

    if (config.nhidden is None) | (config.shidden == 0):
        config.nhidden = 0
        config.shidden = 0

    if (config.nhidden == 0):
        hidden_units = ()
    else:
        hidden_units = tuple([config.shidden for _ in range(config.nhidden)])

    classifier_network = VanillaNet(config.n_utility * config.ensemble,
                                    FCBody(state_dim=len(config.cov_tags), hidden_units=hidden_units,
                                           gate=F.elu)).to(config.DEVICE)
    print(classifier_network)

    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config




