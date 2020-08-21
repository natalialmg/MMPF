import pandas as pd
import numpy as np
# from . import *
import sys, os
sys.path.append(".")
sys.path.append("..")
# from MinimaxParetoFair.network import *

from MinimaxParetoFair import *
from .dataloader_utils import *
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize, ToPILImage, ToTensor,RandomHorizontalFlip,RandomVerticalFlip, RandomRotation, RandomAffine
from torchvision import models
from .misc import *
from .network import *

# from .synthetic_data_loaders import *
from ast import literal_eval as make_tuple


## MIMIC ##
def load_mimic(split = None):
    dirname = os.path.dirname(__file__)
    if split is None:
        train_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/train_mimic_database.pkl'))
        test_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/test_mimic_database.pkl'))
        val_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/val_mimic_database.pkl'))
    else:
        train_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/train_mimic_database_{:d}.pkl'.format(split)))
        test_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/test_mimic_database_{:d}.pkl'.format(split)))
        val_pd = pd.read_pickle(os.path.join(dirname,'/MIMIC/val_mimic_database_{:d}.pkl'.format(split)))


    tag = 'age'
    values = train_pd[tag].values
    values = np.floor(values / 10)
    values = np.minimum(values, 9)
    train_pd['age_binned'] = values

    values = val_pd[tag].values
    values = np.floor(values / 10)
    values = np.minimum(values, 9)
    val_pd['age_binned'] = values

    values = test_pd[tag].values
    values = np.floor(values / 10)
    values = np.minimum(values, 9)
    test_pd['age_binned'] = values

    secret_tag2 = 'secret'  # sensitive attribute 2 is ethnicity
    secret_tag = 'age_binned' # sensitive attribute 1 is age bin
    utility_tag = 'utility'

    ## Bin into two categories age and then secret
    th_val = 5  # age >= 50 or <50
    secret_values = train_pd[secret_tag].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    train_pd[secret_tag] = secret_values

    secret_values = val_pd[secret_tag].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    val_pd[secret_tag] = secret_values

    secret_values = test_pd[secret_tag].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    test_pd[secret_tag] = secret_values

    th_val = 4  # majority ethnicity (white)
    secret_values = train_pd[secret_tag2].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    train_pd[secret_tag2] = secret_values

    secret_values = val_pd[secret_tag2].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    val_pd[secret_tag2] = secret_values

    secret_values = test_pd[secret_tag2].values
    secret_values[secret_values < th_val] = 0
    secret_values[secret_values >= th_val] = 1
    test_pd[secret_tag2] = secret_values

    # combine binary ethnicity with binary age
    train_pd['combined'] = list(zip(train_pd[utility_tag], train_pd[secret_tag], train_pd[secret_tag2]))
    train_pd['combined'] = pd.Categorical(train_pd['combined']).codes

    val_pd['combined'] = list(zip(val_pd[utility_tag], val_pd[secret_tag], val_pd[secret_tag2]))
    val_pd['combined'] = pd.Categorical(val_pd['combined']).codes

    test_pd['combined'] = list(zip(test_pd[utility_tag], test_pd[secret_tag], test_pd[secret_tag2]))
    test_pd['combined'] = pd.Categorical(test_pd['combined']).codes

    return train_pd, val_pd, test_pd

def get_dataloaders_mimic(config, sampler=True, secret_tag='combined', utility_tag='utility', balanced_tag='combined',
                          shuffle_train=True,shuffle_val = True):
    from .misc import get_weight_dict

    train_pd, val_pd, test_pd = load_mimic(split = config.split)

    n_utility = train_pd[utility_tag].nunique()
    n_secret = train_pd[secret_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_sensitive = n_secret  # depends on dataset
    config.size = 10000 # depends on dataset
    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 512
    if config.LEARNING_RATE == 0: # default learning rate
        if config.type_loss == 0:
            config.LEARNING_RATE = 1e-6
        else:
            config.LEARNING_RATE = 5e-6

    if config.patience == 0:
        config.patience = 10

    if (config.resnet != 0) & (config.resnet != 1):
        resnet = False
    else:
        resnet = bool(config.resnet)

    train_pd['secret_cat'] = train_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    test_pd['secret_cat'] = test_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    val_pd['secret_cat'] = val_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))

    train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))

    # get prior of subgroups
    config.p_sensitive = train_pd['secret_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = None  # Tabular data

    if sampler:
        train_dataloader = DataLoader(PandasDatasetNPKW(pd=train_pd,
                                                        transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(PandasDatasetNPKW(pd=train_pd,
                                                        transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=shuffle_train, num_workers=config.n_dataloader, pin_memory=True)

    val_dataloader = DataLoader(PandasDatasetNPKW(pd=val_pd,
                                                  transform=composed),
                                batch_size=config.BATCH_SIZE,
                                shuffle=shuffle_val, num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(PandasDatasetNPKW(pd=test_pd,
                                                   transform=composed),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.n_dataloader, pin_memory=True)

    # torch.manual_seed(config.seed)
    if config.shidden == '' :
        hidden_units = (2048, 2048)
        config.shidden = '(2048, 2048)'
    else:
        hidden_units = make_tuple(config.shidden)

    if resnet:
        classifier_network = VanillaNet(config.n_utility,
                                        FCResnetBody(state_dim=config.size, hidden_units=hidden_units,
                                                     residual_depth=2,
                                                     gate=F.elu))
    else:
        classifier_network = VanillaNet(config.n_utility,
                                        FCBody(state_dim=config.size, hidden_units=hidden_units,
                                               gate=F.elu,use_batchnorm = config.batchnorm))
    print(classifier_network)

    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config

## HAM ##
def load_HAM():
    dirname = os.path.dirname(__file__)
    test_pd = pd.read_pickle(os.path.join(dirname,'/HAM100000/prelim_kaggle_test_data.pkl'))
    train_pd = pd.read_pickle(os.path.join(dirname,'/HAM100000/prelim_kaggle_train_data.pkl'))
    val_pd = pd.read_pickle(os.path.join(dirname,'/HAM100000/prelim_kaggle_val_data.pkl'))

    return train_pd, val_pd, test_pd

def get_dataloaders_HAM(config,train_pd, val_pd, test_pd,sampler=True, sensitive_tag='cell_type_idx',utility_tag='cell_type_idx',
                        balanced_tag='cell_type_idx',shuffle_train=True,shuffle_val = False,composed_val = True,resize = 224):

    # train_pd, val_pd, test_pd = load_HAM()

    n_utility = train_pd[utility_tag].nunique()
    n_sensitive = train_pd[sensitive_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_sensitive = n_sensitive  # depends on dataset
    config.size = 224  # depends on dataset
    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 32
    if config.LEARNING_RATE == 0:
        if config.type_loss == 0:
            config.LEARNING_RATE = 1e-6
        else:
            config.LEARNING_RATE = 1e-6

    if config.patience == 0:
        config.patience = 10

    train_pd.loc[:,'sensitive_cat'] = train_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
    test_pd.loc[:,'sensitive_cat'] = test_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))
    val_pd.loc[:,'sensitive_cat'] = val_pd[sensitive_tag].apply(lambda x: to_categorical(x, num_classes=n_sensitive))

    train_pd.loc[:,'utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    test_pd.loc[:,'utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    val_pd.loc[:,'utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))

    # get prior of subgroups
    config.p_sensitive = train_pd['sensitive_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = torchvision.transforms.Compose([ColorizeToPIL(),
                                               Resize((config.size, config.size), interpolation=2), ToTensor(), ])

    # composed = torchvision.transforms.Compose([ColorizeToPIL(),
    #                                            Resize((resize, resize), interpolation=2),
    #                                            RandomHorizontalFlip(),RandomVerticalFlip(),
    #                                            RandomRotation(180),
    #                                            ToTensor(), ])

    # composed = torchvision.transforms.Compose([ColorizeToPIL(),
    #                                            Resize((resize, resize), interpolation=2),
    #                                            RandomAffine(degrees = 0,translate=(0.12, 0.12),scale=(0.8,1.2)),
    #                                            RandomHorizontalFlip(),RandomVerticalFlip(),
    #                                            ToTensor(), ])

    composed = torchvision.transforms.Compose([ColorizeToPIL(),
                                               Resize((resize, resize), interpolation=2),
                                               RandomAffine(degrees = 0,translate=(0, 0),scale=(0.9,1.2)),
                                               RandomHorizontalFlip(),RandomVerticalFlip(),
                                               ToTensor(), ])

    composed_test = torchvision.transforms.Compose([ColorizeToPIL(),
                                               Resize((resize, resize), interpolation=2), ToTensor(), ])


    if sampler:
        train_dataloader = DataLoader(ImageDataset(pd=train_pd, utility_tag='utility_cat',
                                                   secret_tag='sensitive_cat',
                                                   transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(ImageDataset(pd=train_pd, utility_tag='utility_cat',
                                                   secret_tag='sensitive_cat',
                                                   transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=shuffle_train, num_workers=config.n_dataloader, pin_memory=True)

    if composed_val:
        val_dataloader = DataLoader(ImageDataset(pd=val_pd, utility_tag='utility_cat',
                                                 secret_tag='sensitive_cat',
                                                 transform=composed),
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=shuffle_val, num_workers=config.n_dataloader, pin_memory=True)
    else:
        val_dataloader = DataLoader(ImageDataset(pd=val_pd, utility_tag='utility_cat',
                                                 secret_tag='sensitive_cat',
                                                 transform=composed_test),
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=shuffle_val, num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(ImageDataset(pd=test_pd, utility_tag='utility_cat',
                                              secret_tag='sensitive_cat',
                                              transform=composed_test),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.n_dataloader, pin_memory=True)

    ### NETWORK BIG #### config.size = 224
    classifier_network = VanillaNet(config.n_utility, body=models.densenet121(pretrained=True),
                                    feature_dim=1000).to(config.DEVICE)

    # print(classifier_network)

    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config

## GERMAN ##
def load_german(split=None):
    dirname = os.path.dirname(__file__)
    if split is None:
        train_pd = pd.read_csv(os.path.join(dirname,'./datasets/german/train.csv')).drop(columns='Unnamed: 0', axis=1)
        test_pd = pd.read_csv(os.path.join(dirname,'./datasets/german/test.csv')).drop(columns='Unnamed: 0', axis=1)
        val_pd = pd.read_csv(os.path.join(dirname,'./datasets/german/val.csv')).drop(columns='Unnamed: 0', axis=1)
    else:
        train_pd = pd.read_csv(
          os.path.join(dirname,'./datasets/german/train_{:d}.csv'.format(split))).drop(columns='Unnamed: 0', axis=1)
        test_pd = pd.read_csv(
          os.path.join(dirname,'./datasets/german/test_{:d}.csv'.format(split))).drop(columns='Unnamed: 0', axis=1)
        val_pd = pd.read_csv(
          os.path.join(dirname,'./datasets/german/val_{:d}_{:d}.csv'.format(split))).drop(columns='Unnamed: 0', axis=1)

    tags = ['sex-age', 'sex', 'age', 'month', 'credit']
    for t in tags:
        train_pd[t + '_cat'] = pd.Categorical(train_pd[t]).codes
        val_pd[t + '_cat'] = pd.Categorical(val_pd[t]).codes
        test_pd[t + '_cat'] = pd.Categorical(test_pd[t]).codes

    col_tags = list(train_pd.columns)
    remove_tags = ['sex-age', 'sex', 'age', 'sex-age_cat', 'sex_cat', 'age_cat', 'credit', 'credit_cat']
    for rm in remove_tags:
        col_tags.remove(rm)

    return train_pd, val_pd, test_pd, col_tags

def get_dataloaders_german(config, sampler=True, secret_tag='sex_cat', utility_tag='credit_cat',
                           balanced_tag='sex_cat',shuffle_train=True,shuffle_val = True):
    from .misc import get_weight_dict
    train_pd, val_pd, test_pd, col_tags = load_german(split=config.split)

    n_utility = train_pd[utility_tag].nunique()
    n_secret = train_pd[secret_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_sensitive = n_secret  # depends on dataset
    config.cov_tags = col_tags

    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 32
    if config.LEARNING_RATE == 0:
        if config.type_loss == 0:
            config.LEARNING_RATE = 5e-4
        else:
            config.LEARNING_RATE = 1e-3
    config.size = len(config.cov_tags)
    if (config.resnet != 0) & (config.resnet != 1):
        resnet = True
    else:
        resnet = bool(config.resnet)

    ## Dataframe rename
    train_pd['secret_cat'] = train_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))
    test_pd['secret_cat'] = test_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))
    val_pd['secret_cat'] = val_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))

    train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))
    test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))
    val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))

    # get prior of subgroups
    config.p_sensitive = train_pd['secret_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = None  # Tabular data
    if sampler:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=config.cov_tags,
                                                         utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                         transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=config.cov_tags,
                                                         utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                         transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=shuffle_train, num_workers=config.n_dataloader, pin_memory=True)

    val_dataloader = DataLoader(TablePandasDataset(pd=val_pd, cov_list=config.cov_tags,
                                                   utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                   transform=composed),
                                batch_size=config.BATCH_SIZE,
                                shuffle=shuffle_val , num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(TablePandasDataset(pd=test_pd, cov_list=config.cov_tags,
                                                    utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                    transform=composed),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.n_dataloader, pin_memory=True)

    ## NETWORK ##
    # torch.manual_seed(config.seed)

    if config.shidden == '' :
        hidden_units = ()
        config.shidden = '()'
    else:
        hidden_units = make_tuple(config.shidden)

    if resnet:
        classifier_network = VanillaNet(config.n_utility,
                                        FCResnetBody(state_dim=config.size, hidden_units=hidden_units,
                                                     residual_depth=2,
                                                     gate=F.elu))
    else:
        classifier_network = VanillaNet(config.n_utility,
                                        FCBody(state_dim=config.size, hidden_units=hidden_units,
                                               gate=F.elu,use_batchnorm = config.batchnorm))

    print(classifier_network)


    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config

## ADULT ##
def load_adult(split=None):
    dirname = os.path.dirname(__file__)
    if split is None:
        train_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/train.csv')).drop(columns='Unnamed: 0', axis=1)
        test_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/test.csv')).drop(columns='Unnamed: 0', axis=1)
        val_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/val.csv')).drop(columns='Unnamed: 0', axis=1)
    else:
        train_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/train_' + str(split) + '.csv')).drop(columns='Unnamed: 0',
                                                                                                   axis=1)
        test_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/test_' + str(split) + '.csv')).drop(columns='Unnamed: 0',
                                                                                                 axis=1)
        val_pd = pd.read_csv(os.path.join(dirname,'./datasets/adult/val_' + str(split) + '.csv')).drop(columns='Unnamed: 0',
                                                                                               axis=1)

    tags = ['race', 'sex', 'race-sex', 'income-per-year']
    for t in tags:
        train_pd[t + '_cat'] = pd.Categorical(train_pd[t]).codes
        val_pd[t + '_cat'] = pd.Categorical(val_pd[t]).codes
        test_pd[t + '_cat'] = pd.Categorical(test_pd[t]).codes

    col_tags = list(train_pd.columns)
    remove_tags = ['race', 'sex', 'race-sex', 'race_cat', 'sex_cat',
                   'race-sex_cat', 'income-per-year', 'income-per-year_cat']
    for rm in remove_tags:
        col_tags.remove(rm)

    # utility_tag = 'income-per-year_cat'
    secret_tag2 = 'race_cat'
    secret_tag = 'sex_cat'

    ## Majority race ##
    secret_values = train_pd[secret_tag2].values
    secret_values[secret_values < 4] = 0
    secret_values[secret_values == 4] = 1
    train_pd[secret_tag2] = secret_values

    secret_values = val_pd[secret_tag2].values
    secret_values[secret_values < 4] = 0
    secret_values[secret_values == 4] = 1
    val_pd[secret_tag2] = secret_values

    secret_values = test_pd[secret_tag2].values
    secret_values[secret_values < 4] = 0
    secret_values[secret_values == 4] = 1
    test_pd[secret_tag2] = secret_values

    train_pd['combined'] = list(zip(train_pd[secret_tag2], train_pd[secret_tag]))
    val_pd['combined'] = list(zip(val_pd[secret_tag2], val_pd[secret_tag]))
    test_pd['combined'] = list(zip(test_pd[secret_tag2], test_pd[secret_tag]))

    train_pd['combined'] = pd.Categorical(train_pd['combined']).codes
    val_pd['combined'] = pd.Categorical(val_pd['combined']).codes
    test_pd['combined'] = pd.Categorical(test_pd['combined']).codes


    train_pd['race_sex_cat'] = list(zip(train_pd[secret_tag2], train_pd[secret_tag]))
    val_pd['race_sex_cat'] = list(zip(val_pd[secret_tag2], val_pd[secret_tag]))
    test_pd['race_sex_cat'] = list(zip(test_pd[secret_tag2], test_pd[secret_tag]))

    train_pd['race_sex_cat'] = pd.Categorical(train_pd['race_sex_cat']).codes
    val_pd['race_sex_cat'] = pd.Categorical(val_pd['race_sex_cat']).codes
    test_pd['race_sex_cat'] = pd.Categorical(test_pd['race_sex_cat']).codes

    # white man vs world (Uncomment)
    c_th = 3
    tag_th = 'combined'
    secret_values = train_pd[tag_th].values
    secret_values[secret_values != c_th] = 0
    secret_values[secret_values == c_th] = 1
    train_pd[tag_th] = secret_values

    secret_values = val_pd[tag_th].values
    secret_values[secret_values != c_th] = 0
    secret_values[secret_values == c_th] = 1
    val_pd[tag_th] = secret_values

    secret_values = test_pd[tag_th].values
    secret_values[secret_values != c_th] = 0
    secret_values[secret_values == c_th] = 1
    test_pd[tag_th] = secret_values

    return train_pd, val_pd, test_pd, col_tags

def get_dataloaders_adult(config, sampler=True, secret_tag='sex_cat', utility_tag='income-per-year_cat',
                          balanced_tag='sex_cat',shuffle_train=True,shuffle_val = True):
    from .misc import get_weight_dict
    train_pd, val_pd, test_pd, col_tags = load_adult(split=config.split)

    n_utility = train_pd[utility_tag].nunique()
    n_secret = train_pd[secret_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_sensitive = n_secret  # depends on dataset
    config.cov_tags = col_tags
    config.size = len(config.cov_tags)

    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 32
    if config.LEARNING_RATE == 0:
        if config.type_loss == 0:
            config.LEARNING_RATE = 5e-4
        else:
            config.LEARNING_RATE = 5e-4
    if config.patience == 0:
        config.patience = 15
    if (config.resnet != 0) & (config.resnet != 1):
        resnet = True
    else:
        resnet = bool(config.resnet)
    ## Dataframe rename
    train_pd['secret_cat'] = train_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))
    test_pd['secret_cat'] = test_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))
    val_pd['secret_cat'] = val_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=config.n_sensitive))

    train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))
    test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))
    val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=config.n_utility))

    # get prior of subgroups
    config.p_sensitive = train_pd['secret_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()



    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[balanced_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = None  # Tabular data
    if sampler:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=config.cov_tags,
                                                         utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                         transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(TablePandasDataset(pd=train_pd, cov_list=config.cov_tags,
                                                         utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                         transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=shuffle_train, num_workers=config.n_dataloader, pin_memory=True)
        print('Not using balanced sampling!')

    val_dataloader = DataLoader(TablePandasDataset(pd=val_pd, cov_list=config.cov_tags,
                                                   utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                   transform=composed),
                                batch_size=config.BATCH_SIZE,
                                shuffle=shuffle_val, num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(TablePandasDataset(pd=test_pd, cov_list=config.cov_tags,
                                                    utility_tag='utility_cat', sensitive_tag='secret_cat',
                                                    transform=composed),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.n_dataloader, pin_memory=True)

    ## NETWORK ##
    # torch.manual_seed(config.seed)
    if config.shidden == '' :
        hidden_units = (512,512)
        config.shidden = '(512,512)'
    else:
        hidden_units = make_tuple(config.shidden)

    if resnet:
        classifier_network = VanillaNet(config.n_utility,
                                        FCResnetBody(state_dim=config.size, hidden_units=hidden_units,
                                                     residual_depth=2,
                                                     gate=F.elu))
    else:
        classifier_network = VanillaNet(config.n_utility,
                                        FCBody(state_dim=config.size, hidden_units=hidden_units,
                                               gate=F.elu,use_batchnorm = config.batchnorm))

    print(classifier_network)
    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config


def get_dataloaders_image(config,train_pd, val_pd, test_pd,
                          sampler=True, secret_tag='sensitive',
                        utility_tag='utility', balanced_tag='sensitive',
                          shuffle_train=True,shuffle_val = True,resize = 224):

    n_utility = train_pd[utility_tag].nunique()
    n_secret = train_pd[secret_tag].nunique()

    config.n_utility = n_utility  # depends on dataset
    config.n_sensitive = n_secret  # depends on dataset
    # config.size = 224  # depends on dataset
    if config.BATCH_SIZE == 0:
        config.BATCH_SIZE = 32
    if config.LEARNING_RATE == 0:
        if config.type_loss == 0:
            config.LEARNING_RATE = 2e-6
        else:
            config.LEARNING_RATE = 5e-6

    if config.patience == 0:
        config.patience = 10


    train_pd['secret_cat'] = train_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    test_pd['secret_cat'] = test_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))
    val_pd['secret_cat'] = val_pd[secret_tag].apply(lambda x: to_categorical(x, num_classes=n_secret))

    train_pd['utility_cat'] = train_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    test_pd['utility_cat'] = test_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))
    val_pd['utility_cat'] = val_pd[utility_tag].apply(lambda x: to_categorical(x, num_classes=n_utility))

    # get prior of subgroups
    config.p_sensitive = train_pd['secret_cat'].mean()
    config.p_utility = train_pd['utility_cat'].mean()

    weight_dic = get_weight_dict(train_pd, balanced_tag)
    train_weights = torch.DoubleTensor(train_pd[secret_tag].apply(lambda x: weight_dic[x]).values)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

    composed = torchvision.transforms.Compose([ColorizeToPIL(),
                                               Resize((resize, resize), interpolation=2),
                                               RandomHorizontalFlip(),RandomVerticalFlip(),
                                               RandomRotation(180),
                                               ToTensor(), ])

    composed_test = torchvision.transforms.Compose([ColorizeToPIL(),
                                               Resize((resize, resize), interpolation=2), ToTensor(), ])

    if sampler:
        train_dataloader = DataLoader(ImageDataset(pd=train_pd, utility_tag='utility_cat',
                                                   secret_tag='secret_cat',
                                                   transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      sampler=train_sampler, num_workers=config.n_dataloader, pin_memory=True)
    else:
        train_dataloader = DataLoader(ImageDataset(pd=train_pd, utility_tag='utility_cat',
                                                   secret_tag='secret_cat',
                                                   transform=composed),
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=shuffle_train, num_workers=config.n_dataloader, pin_memory=True)

    val_dataloader = DataLoader(ImageDataset(pd=val_pd, utility_tag='utility_cat',
                                             secret_tag='secret_cat',
                                             transform=composed),
                                batch_size=config.BATCH_SIZE,
                                shuffle=shuffle_val, num_workers=config.n_dataloader, pin_memory=True)

    test_dataloader = DataLoader(ImageDataset(pd=test_pd, utility_tag='utility_cat',
                                              secret_tag='secret_cat',transform=composed_test),
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.n_dataloader, pin_memory=True)

    ### NETWORK BIG #### config.size = 224
    # classifier_network = VanillaNet(config.n_utility, body=models.densenet121(pretrained=True),
                                    # feature_dim=1000).to(config.DEVICE)

    classifier_network = VanillaNet(config.n_utility, body=MyDenseBody()).to(config.DEVICE)

    # print(classifier_network)

    return train_dataloader, val_dataloader, test_dataloader, classifier_network,config



## GET DATALOADERS ##
def get_dataloaders(config, *args,**kwargs):
    if config.dataset=='mimic':
        dl = get_dataloaders_mimic
    elif config.dataset=='adult_gender':
        dl = get_dataloaders_adult
    elif config.dataset=='adult_malewhite':
        dl = get_dataloaders_adult
        kwargs['secret_tag'] = 'combined'
        kwargs['balanced_tag'] = 'combined'
    elif config.dataset=='adult_race_gender':
        dl = get_dataloaders_adult
        kwargs['secret_tag'] = 'race_sex_cat'
        kwargs['balanced_tag'] = 'race_sex_cat'
    elif config.dataset=='german':
        dl = get_dataloaders_german
    elif config.dataset=='HAM':
        dl = get_dataloaders_HAM
    # elif config.dataset =='synthetic':
    #     dl = get_dataloaders_gauss3groups
    # elif config.dataset =='synthetic2d':
    #     dl = get_dataloaders_gauss3groups
    #     kwargs['mean_array'] = [-0.5, 0.5]
    #     kwargs['pa_array'] = [1/2, 1/2]
    #     kwargs['low_rho_array'] = [0.4, 0.4]
    #     kwargs['high_rho_array'] = [0.9, 0.9]
    #     kwargs['transitions_array'] = [-0.15, 0.15]
    #     print('SYNTHETIC 2D!!!')
    else:
        raise NotImplementedError

    return dl(config, *args,**kwargs)


def get_datasets(config):
  secret_tag = 'combined'
  utility_tag = 'utility'
  balanced_tag = 'combined'
  if config.dataset == 'mimic':
    dl = load_mimic
  elif config.dataset == 'adult_gender':
    dl = load_adult
    secret_tag = 'sex_cat'
    utility_tag = 'income-per-year_cat'
  elif config.dataset == 'adult_malewhite':
    dl = load_adult
    secret_tag = 'combined'
    utility_tag='income-per-year_cat'
  elif config.dataset == 'adult_race_gender':
    dl = load_adult
    secret_tag = 'race_sex_cat'
    utility_tag = 'income-per-year_cat'
  elif config.dataset == 'german':
    dl = load_german
  elif config.dataset == 'HAM':
    dl = load_HAM
  else:
    raise NotImplementedError

  return dl(config.split) + (secret_tag, utility_tag)