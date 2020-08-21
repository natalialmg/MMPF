import torch
import numpy as np
from .train_utils import *
from .logger import instantiate_logger
from .losses import losses
from .misc import *
from torch import optim
from ast import literal_eval as make_tuple
import sys
# sys.path.append(".")
# sys.path.append("..")

import pandas as pd

class MMPF_trainer():
    def __init__(self,config,train_dataloader, val_dataloader, test_dataloader, classifier_network):

        #Load datasets; classifier and config file
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.classifier_network = classifier_network
        self.config = config
        self.classifier_network = self.classifier_network.to(self.config.DEVICE)

        #Get initial weight_penalty from tuple_string
        if self.config.mu_init == '':
            self.config.mu_init = np.ones(int(self.config.n_sensitive))
        else:
            self.config.mu_init = np.array(make_tuple(self.config.mu_init))
        self.config.mu_penalty = torch.from_numpy(self.config.mu_init/self.config.mu_init.sum()).float()
        self.config.mu_penalty = self.config.mu_penalty.to(self.config.DEVICE)

        ## Loss type ##
        if self.config.type_loss == 2:
            loss_str = 'MSE'
            reduction = 'sum'
        elif self.config.type_loss == 1:
            loss_str = 'TV'
            reduction = 'sum'
        else:
            loss_str = 'CE'
            reduction = 'sum'
            if self.config.regression:
                print('CAREFUL CE LOSS WITH REGRESSION OBJECTIVE!')

        self.criteria = losses(type_loss=config.type_loss,reduction=reduction,regression=config.regression)

        ## Optimizer ##
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE)
        else:
            self.optimizer = optim.SGD(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE)

        ## Evaluation  ##
        self.df_train_result = []
        self.df_val_result = []
        self.df_test_result = []

        ### Save File Name Generation ###
        #NW tag
        if self.config.shidden != '':
            aux = make_tuple(self.config.shidden)
            str_network = '_'
            for i in aux:
                str_network = str_network + str(i) + '_'
        else:
            str_network = ''

        #tag for the mu_init spec
        mu_init_str = ''
        flag = self.config.mu_init/np.sum(self.config.mu_init) - np.ones(self.config.mu_init.shape)/self.config.mu_init.shape[0]
        flag = np.sum(np.abs(flag))
        if (self.config.type != 'naive') & (self.config.type != 'balanced') & (flag != 0):
            for i in self.config.mu_init:
                mu_init_str = mu_init_str+str(int(i))

        #save file name
        self.save_file = '{:s}_split{:s}_paretofair_{:s}lr{:s}dlr{:s}_hls{:s}bs{:s}_{:s}_muini{:s}_seed{:d}'.format(
            self.config.dataset, str(self.config.split), self.config.optimizer, str(self.config.LEARNING_RATE), str(self.config.lrdecay),
            str(str_network), str(self.config.BATCH_SIZE),loss_str, str(mu_init_str), self.config.seed)

        #add prefixes
        if not config.sampler:
            self.save_file = self.save_file+'_samplerfalse'
        if self.config.type != '':
            self.save_file = self.save_file +'_'+self.config.type

        self.config.save_file = self.config.save_dir + self.save_file
        self.config.save_file_logger = self.save_file
        self.config.save_file_model = self.config.save_dir + self.save_file+'.pth'

        # Instantiate logger
        self.config = instantiate_logger(config=self.config)

        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('-------------- MMPF Trainer OBJECT CREATED ----------------------------')
        print('save_file:', self.config.save_file)
        print('-------------------------------------------------------------------------')
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def APSTAR_torch(self, mua_ini, niter = 15, max_patience = 15, Kini=1, Kmin = 20, alpha = 0.5,
                     risk_round_factor=3,reset_optimizer = False):
        i = 0
        i_patience = 0

        mu_i = mua_ini+0
        mu_i = mu_i.astype('float16')
        mu_i = mu_i/np.sum(mu_i)

        ##save lists
        risk_list = []
        mu_list = []
        risk_best_list = []
        mu_best_list = []
        params_list = []

        ##save dictionaries
        pareto_saves = {}
        base_loss_saves = {}
        # accuracy_loss_saves = {}
        # opt_params_saves = {}
        # full_loss_saves = {}

        K = Kini+0
        while ((i <=niter) & (i_patience <=max_patience)):

            self.config.mu_penalty = torch.from_numpy(mu_i / mu_i.sum()).float()
            self.config.mu_penalty = self.config.mu_penalty.to(self.config.DEVICE)
            print('#### Iteration:', i, '; current mu: ', to_np(self.config.mu_penalty))

            if reset_optimizer:
                print('Optimizer reset ; lr : ',self.config.LEARNING_RATE )
                if self.config.optimizer == 'adam':
                    self.optimizer = optim.Adam(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE)
                else:
                    self.optimizer = optim.SGD(self.classifier_network.parameters(), lr=self.config.LEARNING_RATE)

            # get h optiman and max risks
            base_loss_dic, full_loss_dic, accuracy_s_dic, opt_params_dic, best_base_loss_val = adaptive_optimizer(
                self.train_dataloader,
                self.val_dataloader,
                self.optimizer,
                self.classifier_network,
                self.criteria, self.config)

            # h,risk,_ = bs_optimal(mu_i)
            risk = np.round(best_base_loss_val + 0,risk_round_factor)
            risk_max = np.max(risk)

            # argmax_risks
            argrisk_max = np.arange(risk.shape[0])
            # argrisk_max = argrisk_max[((risk_max - risk ) /risk_max) < 1e-8] # consider an argmax ball
            argrisk_max = argrisk_max[risk == risk_max] #consider argmax exactly

            if i == 0:
                # Initialization#
                risk_max_best = risk_max + 1 #to enter condition of improved risk in i=0 and save initial best risk

            # improved risk
            if risk_max_best > risk_max:

                # update best risk
                risk_max_best = risk_max + 0
                argrisk_max_best = argrisk_max + 0
                risk_best = risk + 0
                mu_best = mu_i + 0

                ## resets
                K = np.minimum(K, Kmin)
                i_patience = 0
                type_step = 0 #improvement

                model_save(self.config.best_network_path, self.classifier_network, self.criteria,
                           self.optimizer)  #save model to best network Messi GOAT
                print('Iteration:', i,' k:',K,'Improved minimax risk (arg/max): ', argrisk_max_best, risk_max_best)

            else:  # no risk improvement

                K += 1
                i_patience += 1
                type_step = 1 #no improvement
                print('Iteration: ', i,' k:',K, 'No minimax risk improvement, current best (arg/max): ', argrisk_max_best, risk_max_best)

            self.classifier_network, self.criteria, self.optimizer = model_load(self.config.best_network_path)  # check this.. loading best last

            # step update
            mask_aux = np.zeros(mu_i.shape)
            mask_aux[risk >= risk_max_best] = 1
            step_mu_i = mask_aux / np.sum(mask_aux)
            #weight vector updated after save lists

            ##############    Save lists  ##############
            params_list.append([type_step, K, alpha])
            risk_list.append(risk)
            mu_list.append(mu_i)
            risk_best_list.append(risk_best)
            mu_best_list.append(mu_best)
            #dir_step_list.append(step_mu_i)
            base_loss_saves[i] = base_loss_dic
            #accuracy_loss_saves[i] = accuracy_s_dic
            #opt_params_saves[i] = opt_params_dic
            #full_loss_saves[i] = full_loss_dic

            print('Iteration:', i, '; step reduced minimax?:', type_step, ' risk (arg/ max)', argrisk_max,
                  risk[argrisk_max], '; (arg/ max best): ', argrisk_max_best, risk_max_best)
            print('mu_i: ', mu_i, '; new delta_mu: ', step_mu_i, ' alpha: ', alpha, '; K: ', K)
            print('risks: ', risk, ' ; best risk: ', risk_best)
            print()
            ############################################

            ### UPDATE WEIGHTING_VECTOR ###
            mu_i = (1 - alpha) * mu_i + step_mu_i * alpha / K
            mu_i = mu_i / np.sum(mu_i)
            i += 1

            ### Empirical Pareto Check ###
            risk_list_np = np.array(risk_list)
            pareto_mask = pareto_check(risk_list_np.transpose())
            print('pareto[iteration] (1: non dominated, 0: dominated): ', pareto_mask)
            print('risks: ',risk_list_np)
            pareto_flag = (pareto_mask[pareto_mask.shape[0] - 1] > 0)
            if not pareto_flag:
                self.optimizer.param_groups[0]['lr'] *= self.config.lrdecay
                print('lr_decay!!')

        pareto_saves['params_list'] = params_list
        pareto_saves['risk_list'] = risk_list
        pareto_saves['risk_best_list'] = risk_best_list
        pareto_saves['mu_list'] = mu_list
        pareto_saves['mu_best_list'] = mu_best_list
        #pareto_saves['dir_step_list'] = dir_step_list
        pareto_saves['base_loss_saves'] = base_loss_saves
        # pareto_saves['accuracy_loss_saves'] = accuracy_loss_saves
        # pareto_saves['opt_params_saves'] = opt_params_saves
        # pareto_saves['full_loss_saves'] = full_loss_saves

        print('patience counter:', i_patience, 'total iterations:', i)
        return pareto_saves

    def fast_epoch_evaluation_bundle(self, set = 'test'):

        columns_tag = ['secret_gt', 'utility_gt']
        for _ in range(self.config.n_utility):
            columns_tag.append('utility_pest_'+str(_))

        df_result = pd.DataFrame(columns=columns_tag)

        ### train ###
        if set == 'train':
            utility_pred_l, utility_gt_l, secret_gt_l= fast_epoch_evaluation(self.train_dataloader,
                                                                                           self.classifier_network, self.config)
            data_train_results = np.concatenate([np.array(secret_gt_l)[:, np.newaxis],
                                                 np.array(utility_gt_l)[:, np.newaxis],
                                                 np.array(utility_pred_l).transpose()], axis=1)
            df_result = pd.DataFrame(data_train_results, columns=columns_tag)

        ### validation ###
        if set == 'val':
            utility_pred_l, utility_gt_l, secret_gt_l = fast_epoch_evaluation(self.val_dataloader,
                                                                                           self.classifier_network, self.config)
            data_val_results = np.concatenate([np.array(secret_gt_l)[:, np.newaxis],
                                               np.array(utility_gt_l)[:, np.newaxis],
                                               np.array(utility_pred_l).transpose()], axis=1)
            df_result = pd.DataFrame(data_val_results, columns=columns_tag)

        ### test ###
        if set == 'test':
            utility_pred_l, utility_gt_l, secret_gt_l = fast_epoch_evaluation(self.test_dataloader,
                                                                                           self.classifier_network, self.config)
            data_test_results = np.concatenate([np.array(secret_gt_l)[:, np.newaxis],
                                                 np.array(utility_gt_l)[:, np.newaxis], np.array(utility_pred_l).transpose()],axis=1)
            df_result = pd.DataFrame(data_test_results, columns=columns_tag)

        return df_result

def APSTAR(bs_optimal, mua_ini, niter = 100, max_patience = 20, Kini=1,
           Kmin = 20, alpha = 0.5, verbose = False):

    ####
    # bs_optimal: model function; returns the risk for a given weight_vector
    #####
    i = 0
    i_patience = 0

    mu_i = mua_ini+0

    ##outputs
    risk_list = []
    mu_list = []
    risk_best_list = []
    mu_best_list = []
    params_list = []
    pareto_saves = {}

    K = Kini
    while ((i <=niter) & ( i_patience <=max_patience)):

        # get h optiman and max risks
        risk = bs_optimal(mu_i)
        # print(risk)
        risk_max = np.max(risk)

        # argmax_risks
        argrisk_max = np.arange(risk.shape[0])
        # argrisk_max = argrisk_max[((risk_max - risk ) /risk_max) < 1e-8] ## argmax ball
        argrisk_max = argrisk_max[risk == risk_max]

        if i == 0:
            # Initialization#
            risk_max_best = risk_max + 1

        # improved risk
        if risk_max_best > risk_max:

            # update best risk
            risk_max_best = risk_max + 0
            argrisk_max_best = argrisk_max + 0
            risk_best = risk + 0
            mu_best = mu_i + 0

            ## resets
            K = np.minimum(K, Kmin)
            i_patience = 0
            type_step = 0

        else:  # no risk improvement

            # update decay series
            K += 1
            i_patience += 1
            type_step = 1

        #step update
        mask_aux = np.zeros(mu_i.shape)
        mask_aux[risk >= risk_max_best] = 1
        step_mu_i = mask_aux / np.sum(mask_aux)

        if verbose:
            print('Iteration:', i, ' ; step reduced minimax?: ', type_step,' risk (arg/ max)', argrisk_max,
                  risk[argrisk_max], ';best max risk (arg/ max): ',argrisk_max_best, risk_max_best)

            #improve others
            print('mu_i:', mu_i, np.sum(mu_i),'; new delta_mu:', step_mu_i,' alpha:', alpha, '; K: ', K)
            print('risks:',risk, ' ; best risk:',risk_best)
            print()

        #save lists
        params_list.append([type_step, K, alpha])
        risk_list.append(risk)
        mu_list.append(mu_i)
        risk_best_list.append(risk_best)
        mu_best_list.append(mu_best)

        mu_i = (1 - alpha) * mu_i + step_mu_i * alpha / K
        mu_i = mu_i / np.sum(mu_i)
        i += 1

    pareto_saves['params_list'] = params_list
    pareto_saves['risk_list'] = risk_list
    pareto_saves['risk_best_list'] = risk_best_list
    pareto_saves['mu_list'] = mu_list
    pareto_saves['mu_best_list'] = mu_best_list

    print('patience counter:', i_patience, 'total iterations:', i)
    print('-----------------------------------------')
    return pareto_saves

class SKLearn_Weighted_LLR():

    def __init__(self, x_train, y_train, a_train, x_val, y_val, a_val, C_reg=1e7):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(solver='lbfgs', max_iter=10000, warm_start=True, C=C_reg)
        self.x_train = x_train
        self.y_train = y_train
        self.a_train = a_train
        self.x_val = x_val
        self.y_val = y_val
        self.a_val = a_val

    def weighted_fit(self, x, y, a, mu):
        sample_weights = np.take(mu, a)
        self.model.fit(X=x, y=y, sample_weight=sample_weights)


    def eval(self,x_val, y_val, a_val):
        risks = []
        for a in np.unique(a_val):
            mask = a_val == a
            y_mask = y_val[mask]
            x_mask = x_val[mask]
            log_proba = self.model.predict_log_proba(x_mask)
            running_risk = []
            for idx in range(len(y_mask)):
                running_risk.append(-log_proba[idx, y_mask[idx]])
            risks.append(np.mean(running_risk))
        return np.array(risks)

    def __call__(self, mu):
        self.weighted_fit(self.x_train, self.y_train, self.a_train, mu)
        return self.eval(self.x_val, self.y_val, self.a_val)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    def predict_log_proba(self, x):
        return self.model.predict_log_proba(x)