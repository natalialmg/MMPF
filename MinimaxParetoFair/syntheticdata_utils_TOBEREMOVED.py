
import numpy as np
from . import *
import sys
from scipy.stats import bernoulli,norm
import torch
# from MinimaxParetoFair.network import *
from torch.utils.data import Dataset, DataLoader
sys.path.append(".")
sys.path.append("..")
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

def get_bs_optimal(x_array, p_xa, p_yxa, mua, type = 'MSE'):
    ##########################
    # THIS IS FOR BINARY Y!!!
    #########################

    dx = np.max(x_array) - np.min(x_array)
    dx /= x_array.shape[0]

    h = np.sum(p_yxa * p_xa * mua[np.newaxis, :], axis=1) / np.sum(p_xa * mua[np.newaxis, :], axis=1)

    if type == 'MSE':
        risk = p_yxa * 2 * ((1 - h[:, np.newaxis]) ** 2) + (1 - p_yxa) * 2 * ((h[:, np.newaxis]) ** 2)
        risk *= p_xa
        risk = np.sum(risk, axis=0) * dx
    else:
        risk = -p_yxa*np.log(h[:, np.newaxis]) - (1-p_yxa)*np.log((1-h[:, np.newaxis]))
        risk *=  p_xa
        risk = np.sum(risk, axis=0) * dx


    aux = (p_yxa - h[:, np.newaxis]) * p_xa / np.sum(p_xa * mua)
    aux2 = (p_yxa - h[:, np.newaxis])
    drisk = (-4 * aux[:, np.newaxis, :] * aux2[:, :, np.newaxis]) * p_xa[:, :,
                                                                    np.newaxis]  # -4 cause is binary so its X2
    drisk = np.sum(drisk, axis=0) * dx

    return h, risk, drisk

def get_probabilities(x_array, param_pa, param_pyxa, param_pxa):
    p_xa = np.zeros([x_array.shape[0], param_pa.shape[0]])
    p_yxa = np.zeros([x_array.shape[0], param_pa.shape[0]])

    for i in np.arange(param_pa.shape[0]):
        z_i = (x_array - param_pxa[i, 0]) / param_pxa[i, 1]
        p_xa[:, i] = norm.pdf(z_i)
        step = np.zeros(x_array.shape[0])
        step[x_array >= param_pyxa[i, 0]] = 1
        p_yxa[:, i] = step * param_pyxa[i, 2]
        p_yxa[:, i] += (1 - step) * param_pyxa[i, 1]
    return p_xa, p_yxa

def get_probabilities_extended(x_array, param_pa, param_pyxa, param_pxa):
    p_xa = np.zeros([x_array.shape[0], param_pa.shape[0]])
    p_yxa = np.zeros([x_array.shape[0], param_pa.shape[0]])

    for i in np.arange(param_pa.shape[0]):
        z_i = (x_array - param_pxa[i, 0]) / param_pxa[i, 1]
        p_xa[:, i] = norm.pdf(z_i)

        ## build p_yxa
        params = param_pyxa[i]
        th = params[0]
        values = params[1]
        aux = np.zeros([x_array.shape[0]])
        for j in np.arange(th.shape[0]):
            aux[x_array >= th[j]] += 1

        for j in np.arange(values.shape[0]):
            # print(values[j])
            p_yxa[aux == j, i] = values[j]
    return p_xa, p_yxa


def pareto_minmax_GD(bs_optimal ,mua_ini ,niter = 100 ,max_patience = 20 ,eps_step = 1,
                  eps_max = 5e-4 ,ceps_up = 1.5 ,ceps_down = 2 ,up_eps_update = 2):
    i = 0
    i_patience = 0
    up_update = 0

    mu_i = mua_ini
    step_mu_i = np.ones(mu_i.shape ) /mu_i.shape[0]

    ##outputs
    risk_list = []
    mu_list = []
    risk_best_list = []
    mu_best_list = []
    params_list = []

    K = 1
    while (( i <=niter) & ( i_patience <=max_patience)):

        # get h optiman and max risks
        h ,risk, drisk = bs_optimal(mu_i)
        risk_max = np.max(risk)

        # argmax_risks
        argrisk_max = np.arange(risk.shape[0])
        argrisk_max = argrisk_max[((risk_max - risk ) /risk_max) < eps_max]
        mask_max = np.zeros(risk.shape[0])
        mask_max[argrisk_max] = 1

        print('delta_mu : ', step_mu_i, step_mu_i * eps_step, mu_i)
        print('eps_step : ', eps_step, '; K : ', K)

        if i == 0:

            # Initialization#
            risk_max_best = risk_max + 1
            argrisk_max_best = argrisk_max
            mu_best = mu_i + 0
            mask_max_best = mask_max + 0

        # improved risk
        if risk_max_best > risk_max:

            print(drisk)
            drisk = drisk / np.sqrt(np.sum((drisk)**2, axis=1)[:, np.newaxis])
            if argrisk_max.shape[0] > 1:
                step_mu_i = np.mean(drisk[argrisk_max,: ], axis=0) #This should be rethink for >2 elements in minmax
                print('MORE THAN ONE:::')
                print(drisk)
                print(step_mu_i/np.sum(np.abs(step_mu_i)))

                for j in np.arange(argrisk_max.shape[0]):
                    print(np.sum(step_mu_i*drisk[argrisk_max[j],:])/np.sum(np.abs(step_mu_i)))
            else:
                step_mu_i = np.mean(drisk[argrisk_max, :], axis=0)
            step_mu_i = -1*step_mu_i / np.sum(np.abs(step_mu_i))



            eps_th = np.max(mu_i / step_mu_i)


            ## Increase step size (epsilon)
            up_update += 1
            if (up_update >= up_eps_update) & (i > 0) & (np.sum(np.abs(mask_max_best-mask_max)) == 0) :
                eps_step = np.minimum(eps_step * ceps_up,eps_th)

            # update best risk
            risk_max_best = risk_max + 0
            argrisk_max_best = argrisk_max
            risk_best = risk + 0
            mu_best = mu_i + 0
            mask_max_best = mask_max + 0




            ## resets
            i_patience = 0
            type_step = 0

            print('Iteration : ', i, ' Improve risk : ', argrisk_max_best, risk_max_best)

        else:  # no risk improvement

            eps_step = np.minimum(np.maximum(eps_step / ceps_down, 1e-10),eps_th)

            # improve others
            print('Iteration : ', i, ' non improve risk (arg/ max, arg/ max best)', argrisk_max,
                  risk[argrisk_max], argrisk_max_best, risk_max_best)

            i_patience += 1
            type_step = 1



        ##save lists
        params_list.append([type_step, K, eps_step])
        risk_list.append(risk)
        mu_list.append(mu_i)
        risk_best_list.append(risk_best)
        mu_best_list.append(risk_best)

        ## Update mu
        mu_i = mu_best + step_mu_i * eps_step
        mu_i = mu_i / np.sum(mu_i)
        i += 1

        print('-------------------------')

    print('patience , iterations', i_patience, i)
    return risk_list, mu_list, risk_best_list, mu_best_list, params_list

def sample_ybin_xgmm(param_pa, param_pxa, param_pyxa, seed=42, n_samples=10000):
    np.random.seed(seed)
    a_cat = np.random.multinomial(1, param_pa, size=n_samples)
    a_gen = np.argmax(a_cat, axis=1)
    x_gen = np.random.randn(n_samples) * (np.sum(a_cat * param_pxa[:, 1][np.newaxis, :], axis=1))
    x_gen += np.sum(a_cat * param_pxa[:, 0][np.newaxis, :], axis=1)

    y_gen_prob = np.ones(x_gen.shape)
    a_unique = np.unique(a_gen)

    for ix_a in np.arange(a_unique.shape[0]):
        input_opt = param_pyxa[ix_a, :]
        a_val = a_unique[ix_a]
        x_a = x_gen[a_gen == a_val]

        pa_y = np.ones(y_gen_prob[a_gen == a_val].shape) * input_opt[1]
        pa_y[x_a >= input_opt[0]] = input_opt[2]

        y_gen_prob[a_gen == a_val] = pa_y
    # print(np.unique(pa_y), input_opt[0])

    from scipy.stats import bernoulli
    np.random.seed(seed)
    y_gen = bernoulli.rvs(y_gen_prob)

    return x_gen, a_gen, y_gen, y_gen_prob

def sample_ybin_xgmm_extended(param_pa, param_pxa, param_pyxa, seed=42, n_samples=10000):
    # np.random.seed(seed)
    a_cat = np.random.multinomial(1, param_pa, size=n_samples)
    a_gen = np.argmax(a_cat, axis=1)
    x_gen = np.random.randn(n_samples) * (np.sum(a_cat * param_pxa[:, 1][np.newaxis, :], axis=1))
    x_gen += np.sum(a_cat * param_pxa[:, 0][np.newaxis, :], axis=1)

    y_gen_prob = np.ones(x_gen.shape)
    a_unique = np.unique(a_gen)

    for ix_a in np.arange(a_unique.shape[0]):
        a_val = a_unique[ix_a]
        x_a = x_gen[a_gen == a_val]
        pa_y = np.zeros(x_a.shape)

        ## build p_yxa
        params = param_pyxa[ix_a]
        th = params[0]
        values = params[1]
        aux = np.zeros([x_a.shape[0]])
        for j in np.arange(th.shape[0]):
            aux[x_a >= th[j]] += 1

        for j in np.arange(values.shape[0]):
            # print(values[j])
            pa_y[aux == j] = values[j]
        y_gen_prob[a_gen == a_val] = pa_y

    from scipy.stats import bernoulli
    # np.random.seed(seed)
    y_gen = bernoulli.rvs(y_gen_prob)

    return x_gen, a_gen, y_gen, y_gen_prob



def get_pandas_data(x_gen,a_gen,y_gen,ratios=[0.6,0.2,0.2]):
    ratios = ratios/np.sum(ratios) #just in case
    n_samples = x_gen.shape[0]

    ## Train, val & test split
    ntrain = ratios[0]* n_samples
    nval = ratios[1]*n_samples

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

    columns = []
    if len(x_gen.shape)>1:
        for i in np.arange(x_gen.shape[1]):
            columns.append('x'+str(i))
    else:
        columns.append('x')
    columns.append('y')
    columns.append('s')

    train_pd = np.concatenate([x_train[:, np.newaxis], y_train[:, np.newaxis]], axis=1)
    train_pd = np.concatenate([train_pd, a_train[:, np.newaxis]], axis=1)
    train_pd = pd.DataFrame(train_pd, columns=columns)

    val_pd = np.concatenate([x_val[:, np.newaxis], y_val[:, np.newaxis]], axis=1)
    val_pd = np.concatenate([val_pd, a_val[:, np.newaxis]], axis=1)
    val_pd = pd.DataFrame(val_pd, columns=columns)

    test_pd = np.concatenate([x_test[:, np.newaxis], y_test[:, np.newaxis]], axis=1)
    test_pd = np.concatenate([test_pd, a_test[:, np.newaxis]], axis=1)
    test_pd = pd.DataFrame(test_pd, columns=columns)

    return train_pd,val_pd,test_pd

def get_dataloaders_pdtable(train_pd,val_pd,test_pd, sampler=True,cov_tags = ['x'], sensitive_tag='s',
                            utility_tag='y', balanced_tag='s',batch_size = 32,
                            n_dataloader = 32,shuffle_train=True,shuffle_val=False, regression = False):
    from .misc import get_weight_dict

    n_utility = train_pd[utility_tag].nunique()
    n_sensitive = train_pd[sensitive_tag].nunique()
    print('HERE',train_pd[sensitive_tag].nunique() )
    if train_pd[sensitive_tag].nunique()>1:
        train_pd['sensitive_cat'] = train_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
        test_pd['sensitive_cat'] = test_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
        val_pd['sensitive_cat'] = val_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
    else:
        train_pd.loc[:,'sensitive_cat'] = 1
        test_pd.loc[:,'sensitive_cat'] = 1
        val_pd.loc[:,'sensitive_cat'] = 1

    if not regression:
        train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
        test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
        val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    else:
        train_pd.loc[:,'utility_cat'] = train_pd[utility_tag]
        test_pd.loc[:,'utility_cat'] = test_pd[utility_tag]
        val_pd.loc[:,'utility_cat'] = val_pd[utility_tag]

    # get prior of subgroups
    config.p_sensitive = train_pd['sensitive_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = None  # Tabular data

    if sampler:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=cov_tags,
                                                         utility_tag='utility_cat', sensitive_tag='sensitive_cat',
                                                         transform=composed),
                                      batch_size=batch_size,
                                      sampler=train_sampler, num_workers=n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd,cov_list = cov_tags,
                                                         utility_tag = 'utility_cat', sensitive_tag = 'sensitive_cat',
                                                         transform=composed),
                                    batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=n_dataloader, pin_memory = True)

    val_dataloader = DataLoader(TablePandasDataset(pd=val_pd, cov_list=cov_tags,
                                                   utility_tag='utility_cat', sensitive_tag='sensitive_cat',
                                                   transform=composed),
                                batch_size=batch_size,
                                shuffle=shuffle_val, num_workers=n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(TablePandasDataset(pd=test_pd, cov_list=cov_tags,
                                                    utility_tag='utility_cat', sensitive_tag='sensitive_cat',
                                                    transform=composed),
                                 batch_size=batch_size,
                                 shuffle=False, num_workers=n_dataloader, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

# def get_dataloaders_pdtable(train_pd,val_pd,test_pd, sampler=True,cov_tags = ['x'], sensitive_tag='s',
#                             utility_tag='y', balanced_tag='s',batch_size = 32, n_dataloader = 32,shuffle_train=True,shuffle_val=False):
#     from .misc import get_weight_dict
#
#     n_utility = train_pd[utility_tag].nunique()
#     n_sensitive = train_pd[sensitive_tag].nunique()
#
#     train_pd['sensitive_cat'] = train_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
#     test_pd['sensitive_cat'] = test_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
#     val_pd['sensitive_cat'] = val_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
#
#     train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
#     test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
#     val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
#
#     # get prior of subgroups
#     config.p_sensitive = train_pd['sensitive_cat'].mean()
#     config.p_utility = train_pd['utility_cat'].mean()
#
#     weight_dic = get_weight_dict(train_pd, balanced_tag)
#     train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
#     train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
#
#     composed = None  # Tabular data
#
#     if sampler:
#         train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=cov_tags,
#                                                          utility_tag='utility_cat', sensitive_tag='sensitive_cat',
#                                                          transform=composed),
#                                       batch_size=batch_size,
#                                       sampler=train_sampler, num_workers=n_dataloader, pin_memory=True)
#     else:
#         train_dataloader = DataLoader(TablePandasDataset(pd=train_pd,cov_list = cov_tags,
#                                                          utility_tag = 'utility_cat', sensitive_tag = 'sensitive_cat',
#                                                          transform=composed),
#                                     batch_size=batch_size,
#                                     shuffle=shuffle_train, num_workers=n_dataloader, pin_memory = True)
#
#     val_dataloader = DataLoader(TablePandasDataset(pd=val_pd, cov_list=cov_tags,
#                                                    utility_tag='utility_cat', sensitive_tag='sensitive_cat',
#                                                    transform=composed),
#                                 batch_size=batch_size,
#                                 shuffle=shuffle_val, num_workers=n_dataloader, pin_memory=True)
#
#     test_dataloader = DataLoader(TablePandasDataset(pd=test_pd, cov_list=cov_tags,
#                                                     utility_tag='utility_cat', sensitive_tag='sensitive_cat',
#                                                     transform=composed),
#                                  batch_size=batch_size,
#                                  shuffle=False, num_workers=n_dataloader, pin_memory=True)
#
#     return train_dataloader, val_dataloader, test_dataloader
#

def synthetic_samples_ybin_xgmm(param_pa, param_pxa, param_pyxa, seed=42, n_samples=10000, verbose = False, ratios = [0.6, 0.2, 0.2]):
    print(param_pa, param_pyxa, param_pxa)
    # x_gen, a_gen, y_gen, y_gen_prob = sample_ybin_xgmm(param_pa, param_pxa, param_pyxa, seed=seed, n_samples=n_samples)
    x_gen, a_gen, y_gen, y_gen_prob = sample_ybin_xgmm_extended(param_pa, param_pxa, param_pyxa, seed=seed, n_samples=n_samples)
    # ix = (x_gen>-0.5)&(x_gen<0.5)
    # x_gen = x_gen[ix>0]
    # y_gen = y_gen[ix>0]
    # y_gen_prob = y_gen_prob[ix>0]
    # a_gen = a_gen[ix>0]
    if verbose:
        plt.figure(figsize=(10, 2))
        for a_ix in np.unique(a_gen):
            plt.hist(x_gen[a_gen == a_ix], 50, density=True)
        plt.show()

        plt.figure(figsize=(10, 2))
        for a_ix in np.unique(a_gen):
            x_plot = x_gen[a_gen == a_ix]
            y_plot = y_gen_prob[a_gen == a_ix]
            y_aux = y_gen[a_gen == a_ix]

            ix_sort = np.argsort(x_plot)
            plt.plot(x_plot[ix_sort], y_plot[ix_sort])
        plt.show()
    train_pd, val_pd, test_pd = get_pandas_data(x_gen, a_gen, y_gen, ratios=ratios)

    return train_pd, val_pd, test_pd

# def make_classifier(config,resnet = True,residual_depth= 2, gate = F.elu):
#
#     ## NETWORK ##
#     # if seed is None:
#     #     torch.manual_seed(config.seed)
#     # else:
#     #     torch.manual_seed(seed)
#
#     if config.shidden == '' :
#         hidden_units = ()
#     else:
#         hidden_units = make_tuple(config.shidden)
#
#     # classifier_network = VanillaNet(config.n_utility,
#     #                                 FCBody(state_dim=len(config.cov_tags), hidden_units=hidden_units,
#     #                                            gate=F.relu)).to(config.DEVICE)
#     if resnet:
#         classifier_network = VanillaNet(config.n_utility,
#                                         FCResnetBody(state_dim=len(config.cov_tags), hidden_units=hidden_units,
#                                                      residual_depth=residual_depth,
#                                                gate=gate,use_batchnorm=config.batchnorm))
#     else:
#         classifier_network = VanillaNet(config.n_utility,
#                                         FCBody(state_dim=len(config.cov_tags), hidden_units=hidden_units,
#                                                    gate=gate,use_batchnorm=config.batchnorm))
#
#     print(classifier_network)
#
#     return classifier_network


