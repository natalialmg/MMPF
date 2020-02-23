import pandas as pd
import numpy as np
import sys, pickle, os
from distutils.util import strtobool

import sys
sys.path.append(".")
sys.path.append("..")
from MinimaxParetoFair import *
import postprocessing
from ast import literal_eval as make_tuple

def main():
  # Make config
  config = Config()
  # Basic configuration
  config.add_argument('--dataset', action='store', default='adult_race_gender', type=str, dest='dataset', help='dataset name')
  config.add_argument('--split', action='store', default=0, type=int, dest='split', help='dataset split number')
  config.add_argument('--base_save_dir', action='store', default='save_fairness/', type=str, dest='save_dir',
                      help='base directory to save final results')
  config.add_argument('--type', type=str, default='minimax', help='type of fit (minimax, balanced , naive)')
  argv = sys.argv
  config.merge(argv)


  #------------ save dirs ------------#
  config.save_dir = os.path.join(config.save_dir,config.dataset)
  os.makedirs(config.save_dir, exist_ok=True)
  #-----------------------------------#
  if config.type=='minimax':
    savepath = config.save_dir + '/apstar_llr_{:d}.pkl'.format(config.split)
    model_tag = 'apstar_llr'
  elif config.type=='naive':
    savepath = config.save_dir + '/naive_llr_{:d}.pkl'.format(config.split)
    model_tag = 'naive_llr'
  elif config.type=='balanced':
    savepath = config.save_dir + '/balanced_llr_{:d}.pkl'.format(config.split)
    model_tag = 'balanced_llr'

  print('------------- Model file -------------')
  print(savepath)

  print('------------- INITIALIZING Classifier &  Loading datasets -------------')

  train_pd, val_pd, test_pd, col_tags, secret_tag, utility_tag = get_datasets(config)
  x_train = train_pd[col_tags].values
  y_train = train_pd[utility_tag].values
  a_train = train_pd[secret_tag].values

  x_val = val_pd[col_tags].values
  y_val = val_pd[utility_tag].values
  a_val = val_pd[secret_tag].values

  x_test = test_pd[col_tags].values
  y_test = test_pd[utility_tag].values
  a_test = test_pd[secret_tag].values

  # Instantiate logistic regression model
  model = SKLearn_Weighted_LLR(x_train, y_train, a_train, x_val, y_val, a_val)

  # get priors
  sensitive_prior = np.zeros([a_train.max() + 1])
  for a in range(a_train.max() + 1):
    sensitive_prior[a] = np.mean(a_train == a)
  utility_prior = np.zeros([y_train.max() + 1])
  for y in range(y_train.max() + 1):
    utility_prior[y] = np.mean(y_train == y)

  print('------------- Training -------------')
  # Run algorithm
  if config.type=='minimax':
    mua_ini = np.ones(a_val.max() + 1)
    mua_ini /= mua_ini.sum()
    results =APSTAR(model, mua_ini, niter=200, max_patience=200, Kini=1,
                          Kmin=20, alpha=0.5, verbose=False)
    mu_best_list = results['mu_best_list']

    mu_best = mu_best_list[-1]
    model.weighted_fit(x_train, y_train, a_train, mu_best)
  elif config.type =='naive':
    mua_ini = np.ones(a_val.max() + 1)
    mua_ini /= mua_ini.sum()
    model.weighted_fit(x_train, y_train, a_train, mua_ini)
  elif config.type =='balanced':
    mua_ini = 1/(sensitive_prior+1e-3)
    mua_ini /= mua_ini.sum()
    model.weighted_fit(x_train, y_train, a_train, mua_ini)

  # get classifier result tables
  # test tables
  test_pd['secret_gt'] = test_pd[secret_tag]
  test_pd['utility_gt'] = test_pd[utility_tag]
  pu = model.predict_proba(x_test)
  test_pd['utility_pest_0'] = pu[:, 0]
  test_pd['utility_pest_1'] = pu[:, 1]
  df_test_result = test_pd.drop(val_pd.columns.difference(
    ['secret_gt', 'utility_gt', 'utility_pest_0', 'utility_pest_1']), 1)

  pd_data_stats = postprocessing.results_dataframe(df_test_result,config.dataset,
                              model_tag=model_tag,split=config.split,settype = 'test')

  print(pd_data_stats)



  print('------------- Saving final results------------- ')
  priors = {'sensitive': sensitive_prior, 'utility': utility_prior}

  save_data = {}
  if config.type=='minimax':
    save_data['MMPF_saves'] = results
  save_data['group_test_stats'] = pd_data_stats
  save_data['test_prediction'] = df_test_result
  save_data['priors'] = priors
  # save_data['save_file'] = MMPF.config.save_file_model

  with open(savepath, 'wb') as f:
      pickle.dump(save_data, f)

if __name__ == '__main__':
    main()