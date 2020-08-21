import pandas as pd
import numpy as np
import torch, sys, pickle, os
from distutils.util import strtobool

import sys
sys.path.append(".")
sys.path.append("..")
from MinimaxParetoFair import *
# from dataset_loaders import *
# from MMPF_trainer import *
# from postprocessing import *
import postprocessing
from ast import literal_eval as make_tuple

def main():
  # Make config
  config = Config()
  # Basic configuration
  config.add_argument('--bs', action='store', default=0, type=int, dest='BATCH_SIZE', help='batch size')
  config.add_argument('--epochs', action='store', default=1000, type=int, dest='EPOCHS', help='maximum number of epochs')
  config.add_argument('--lr', action='store', default=0, type=float, dest='LEARNING_RATE', help='learning rate')
  config.add_argument('--gpu_id', action='store', default=0, type=int, dest='GPU_ID', help='gpu id, use -1 to run on cpu')
  config.add_argument('--n_workers', action='store', default=32, type=int, dest='n_dataloader',
                      help='number of dataloader parallel workers')
  config.add_argument('--seed', action='store', default=42, type=int, dest='seed', help='randomizer seed')
  config.add_argument('--dataset', action='store', default='adult_race_gender', type=str, dest='dataset', help='dataset name')
  config.add_argument('--split', action='store', default=0, type=int, dest='split', help='dataset split number')
  config.add_argument('--lrdecay', action='store', default=0.25, type=float, dest='lrdecay', help='learning rate decay')
  config.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (sgd, adam)')
  # config.add_argument('--prefix', action='store', default='', type=str, dest='prefix', help='save file prefix')
  config.add_argument('--shl', action='store', default='', type=str, dest='shidden', help='size of hidden layers')
  config.add_argument('--batchnorm', action='store', default=False, type=lambda x: bool(strtobool(x)),
                      dest='batchnorm',
                      help='boolean: batchnorm')
  config.add_argument('--resnet', action='store', default=-1, type=int, dest='resnet',
                      help='resnet -1 is dataset default, 0 is noresnet, 1 is resnet')
  config.add_argument('--regression', action='store', default=False, type=lambda x: bool(strtobool(x)),
                      dest='regression',
                      help='boolean: create tensorboard log?')
  config.add_argument('--resetopt', action='store', default=False, type=lambda x: bool(strtobool(x)),
                      dest='resetopt',
                      help='boolean: reset optimizer after each new apstar iteration?')

  # Print, log and store options
  config.add_argument('--verbose', action='store', default=True, type=lambda x: bool(strtobool(x)),
                      help='boolean: print updates?')
  config.add_argument('--n_print', action='store', default=5, type=int, dest='n_print', help='print frequency')
  config.add_argument('--tb_log', action='store', default='tb_log/', type=str, dest='loggdir',
                      help='tensorboard log path')
  config.add_argument('--log', action='store', default=False, type=lambda x: bool(strtobool(x)), dest='logger_active',
                      help='boolean: create tensorboard log?')
  config.add_argument('--base_save_dir', action='store', default='save_fairness/', type=str, dest='save_dir',
                      help='base directory to save final results')
  config.add_argument('--base_model_save_dir', action='store', default='models/', type=str, dest='base_model_path',
                      help='base directory to store classifier')


  # Pareto-optimality options
  config.add_argument('--patience', action='store', default=20, type=int, dest='patience',
                      help='epoch patience parameter inside adaptive optimization')
  config.add_argument('--mu_init', action='store', default='', type=str, dest='mu_init', help='initial mu penalty')
  config.add_argument('--loss_type', action='store', default=0, type=int, dest='type_loss',
                      help='0: CrossEntropy, 1: L1 2:Brier score (Categorical MSE)')
  config.add_argument('--sampler', action='store', default=True, type=lambda x: bool(strtobool(x)), dest='sampler',
                      help='boolean: activate discrimination penalty')
  config.add_argument('--type', type=str, default='minimax', help='type of fit (minimax, balanced , naive)')
  config.add_argument('--niter', action='store', default=0, type=int, dest='niter_apstar',
                      help='niterations for APSTAR')

  argv = sys.argv
  config.merge(argv)

  #GPU
  if torch.cuda.is_available() and config.GPU_ID>=0:
      DEVICE = torch.device('cuda:%d' % (config.GPU_ID))
  else:
      DEVICE = torch.device('cpu')
  config.DEVICE = DEVICE
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)

  #------------ save dirs ------------#
  config.save_dir = os.path.join(config.save_dir,config.dataset)
  mkdir(config.save_dir)
  os.makedirs(config.save_dir, exist_ok=True)
  base_dir_nw= os.path.join(config.base_model_path, config.dataset)
  mkdir(base_dir_nw)
  os.makedirs(base_dir_nw, exist_ok=True)
  config.best_network_path = base_dir_nw+'/goat_adult_{:d}.pth'.format(config.split) # Save path for best performing network in run
  config.best_adaptive_network_path = base_dir_nw+'/best_iter_adult_{:d}.pth'.format(config.split)  # Save path for best performing network for fixed mu values
  #-----------------------------------#

  type_opt = config.type
  str_path = type_opt+''
  # config.prefix = str_path

  if (type_opt == 'balanced')|(type_opt == 'naive'):
      niter_apstar = 0
  else:
      if config.niter_apstar == 0:
          niter_apstar = 20
      else:
          niter_apstar = config.niter_apstar+0

  if config.type_loss == 2:
      loss_str = 'MSE'
  elif config.type_loss ==1:
      loss_str = 'TV'
  else:
      loss_str = 'CE'

  if config.shidden != '':
      aux = make_tuple(config.shidden)
      str_network = '_'
      for i in aux:
          str_network = str_network + str(i) + '_'
  else:
      str_network = '_'

  savepath = config.save_dir + '/MMPFNN_arq' + str_network
  savepath = savepath + 'loss' + loss_str + '_seed' + str(config.seed) + '_split' + str(config.split) + '_' + str_path + '.pkl'

  print('------------- Model file -------------')
  print(savepath)

  print('------------- INITIALIZING NETWORK &  DATALOADERS -------------')
  train_dataloader, val_dataloader, test_dataloader, classifier_network, config = get_dataloaders(config,
                                                                                                  sampler=config.sampler)
  print('------------- INITIALIZING MMPF trainer -------------')
  print('reset opt ', config.resetopt)
  # Initial weighting
  if (type_opt == 'naive'):
      mu_init = config.p_sensitive * 1000 + 0
      config.mu_init = str(tuple(mu_init.astype('int')))

  MMPF = MMPF_trainer(config, train_dataloader, val_dataloader, test_dataloader, classifier_network)

  print('------------- Training -------------')
  MMPF_saves = MMPF.APSTAR_torch(MMPF.config.mu_init, niter=niter_apstar, max_patience=MMPF.config.patience,
                                 reset_optimizer = MMPF.config.resetopt)
  model_save(MMPF.config.save_file_model, MMPF.classifier_network,MMPF.criteria,MMPF.optimizer)

  print('------------- Evaluating -------------')
  df_test_result = MMPF.fast_epoch_evaluation_bundle()
  pd_data_stats = postprocessing.results_dataframe(df_test_result,MMPF.config.dataset,
                              model_tag=MMPF.config.type,split=MMPF.config.split,settype = 'test')
  print(pd_data_stats)

  print('------------- Saving final results------------- ')
  priors = {}
  priors['sensitive'] = config.p_sensitive
  priors['utility'] = config.p_utility

  save_data = {}
  save_data['MMPF_saves'] = MMPF_saves
  save_data['group_test_stats'] = pd_data_stats
  save_data['test_prediction'] = df_test_result
  save_data['priors'] = priors
  save_data['save_file'] = MMPF.config.save_file_model

  with open(savepath, 'wb') as f:
      pickle.dump(save_data, f)

  print('saved')
if __name__ == '__main__':
    main()