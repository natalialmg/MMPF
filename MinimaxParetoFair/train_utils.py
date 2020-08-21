import numpy as np
import torch
from torch import optim
from .misc import *
from .losses import *
from .torch_utils import *
import pandas as pd


#basic utilities
class early_stopping(object):
    def __init__(self, patience, counter, best_loss):
        self.patience = patience #max number of nonimprovements until stop
        self.counter = counter #number of consecutive nonimprovements
        self.best_loss = best_loss

    def evaluate(self, loss):
        save = False #save nw
        stop = False #stop training
        if loss < 0.999*self.best_loss:
            self.counter = 0
            self.best_loss = loss
            save = True
            stop = False
        else:
            self.counter += 1
            if self.counter > self.patience:
                stop = True

        return save, stop

def pareto_check(table_results, eps_pareto=0):

    #table_results dimension: objectives x iterations
    pareto = np.zeros([table_results.shape[1]])
    if table_results.shape[1] == 1:
        pareto[0] = True
        # print('one')
    else:
        for i in np.arange(table_results.shape[1]):
            dif_risks = table_results[:, i][:, np.newaxis] - table_results  #(dim: objectives x iterations)
            dif_risks = dif_risks < eps_pareto  #evaluates if item iteration i dominates some coordinate, 1 indicates dominance
            dif_risks[:,i] = 1 #we would like to have a row full of ones
            dif_risks = np.sum(dif_risks,axis = 0) #sum on objectives
            pareto[i] = (np.prod(dif_risks) > 0) #prod of iterations (I cannot have an iteration that = 0 when sum all objectives differences

    if np.sum(pareto) == 0: #if all vectors are the same basically...
        ix_p = np.argmin(np.sum(table_results,axis=0))
        pareto[ix_p] = 1
    return pareto

#Functions#
def epoch_training_linearweight(
        dataloader, optimizer, classifier_network, criterions,
        config, logger, train_type='Train'):
    '''
    This function train or evaluates an epoch
    #inputs:
    dataloader, optimizer, classifier_network
    criterions: function that provides a base_loss
    config: must have .DEVICE , .n_sensitive & .mu_penalty
    logger: (tensorboard)
    train_type: if train performs backprop y otherwise only evaluates

    #Outputs:
    base_loss_all_out: output base loss per sensitive
    accuracy_out: output accuracy per sensitive
    full_loss_out: output full loss
    '''

    ###    INITIALIZE MEAN OBJECTS  #######
    #loss summary lists
    base_loss_l = [TravellingMean() for _ in range(config.n_sensitive)]
    accuracy_l = [TravellingMean() for _ in range(config.n_sensitive)]
    sensitive_values = []

    if train_type == 'Train':
        ## list of travelling mean objects -> in each batch updates the current epoch mean, at the end we have the mean of the epoch
        full_loss = TravellingMean()
        classifier_network = classifier_network.train()
    else:
        classifier_network = classifier_network.eval()

    # Loop through samples
    for i_batch, sample_batch in enumerate(dataloader):
        x, utility, sensitive = sample_batch
        x = x.to(config.DEVICE)
        utility = utility.to(config.DEVICE) #batch x nutility
        sensitive = sensitive.to(config.DEVICE) #batch x nsensitive

        # zero the parameter gradients
        optimizer.zero_grad()

        # get output and losses
        logits = classifier_network(x)
        targets = utility
        base_loss = criterions(logits, targets)

        # base loss size:  batch ;
        softmax = nn.Softmax(dim=-1)(logits)
        accuracy_np = to_np(softmax).argmax(-1) == to_np(utility.argmax(-1))

        ##### TRAIN OPTION ##########################################

        if train_type == 'Train':

            base_loss = base_loss.to(config.DEVICE)
            loss = 0
            #get the group base losses
            base_term_e = (base_loss.unsqueeze(-1) * sensitive).sum(0)
            base_term_e = (base_term_e/(torch.max(sensitive.sum(0), torch.ones_like(sensitive.sum(0))))).to(config.DEVICE)
            loss += (base_term_e*config.mu_penalty).sum(0)

            #backpropagation
            loss.backward()
            optimizer.step()

            ######### SAVES FOR VISUALIZATION ###################

            sensitive_np = to_np(sensitive.argmax(-1))
            base_loss_np = to_np(base_loss)
            sensitive_values.extend(list(sensitive_np))
            loss_np = to_np(loss)

            for s in np.unique(sensitive_np):  # store sensitive-segregated values (update travelling means)
                base_loss_l[s].update(base_loss_np[sensitive_np == s])  #base loss per sensitive & ensemble
                accuracy_l[s].update((accuracy_np[sensitive_np == s]))
            full_loss.update(np.array([loss_np])) #full loss all together

        ##### EVALUATION (VALIDATION OR TEST) ################
        else:
            sensitive_np = to_np(sensitive.argmax(-1))
            base_loss_np = to_np(base_loss)
            sensitive_values.extend(list(sensitive_np))

            for s in np.unique(sensitive_np):  # store sensitive-segregated values (update travelling means)
                base_loss_l[s].update(base_loss_np[sensitive_np == s])  #base loss per sensitive & ensemble
                accuracy_l[s].update((accuracy_np[sensitive_np == s]))

    ################ PROCESSING EPOCH PERFORMANCE (GET OUTPUTS) ######################################
    sensitive_values = np.unique(np.array(sensitive_values))
    mu_penalty = to_np(config.mu_penalty)

    base_loss_all_out = np.zeros([config.n_sensitive]) #output base loss per sensitive and ensemble
    accuracy_out = np.zeros([config.n_sensitive])  #output accuracy
    full_loss_out = 0

    n_samples = 0
    for ix_s in sensitive_values:
        # Base loss saves
        base_loss_all_out[ix_s] = base_loss_l[ix_s].mean  # get base loss epoch for each sensitive and ensemble
        n_samples += base_loss_l[ix_s].count  # total number of samples
        accuracy_out[ix_s] = accuracy_l[ix_s].mean

        if logger is not None:
            logger.add_scalar('accuracy_sec/{:s}/{:d}'.format(train_type, ix_s),
                              accuracy_out[ix_s])
            logger.add_scalar('base_loss_sec/{:s}/{:d}'.format(train_type, ix_s),
                              np.mean(base_loss_all_out[ix_s]))
            logger.add_scalar('mu_penalty/{:s}/{:d}'.format(train_type, ix_s),
                              mu_penalty[ix_s])

    if train_type == 'Train':
        full_loss_out += np.sum(mu_penalty * base_loss_all_out)
    else:
        full_loss_out += np.sum(mu_penalty * base_loss_all_out)

    if logger is not None:
        logger.add_scalar('full_loss/{:s}'.format(train_type), full_loss_out)

    return base_loss_all_out, accuracy_out, full_loss_out

def adaptive_optimizer(train_dataloader, val_dataloader,
                       optimizer, classifier_network, criterio, config):

    '''for each epoch:
        penalty_tolerance = constant + min_risk_group*no_harm
        train_network
        update min_risk group

        if improvement in val loss:
            save_nw
        else:
            patience (done by early_stopping class)
            decrease learning rate (optional)

        if patience reach limit (stop):
            break
    '''

    #################### OUTPUTS ##################
    base_loss_all_train = np.zeros([config.EPOCHS, config.n_sensitive]) #loss per group per ensemble
    full_loss_train = np.zeros([config.EPOCHS])
    accuracy_s_train = np.zeros([config.EPOCHS, config.n_sensitive])

    base_loss_all_val = np.zeros([config.EPOCHS, config.n_sensitive]) #loss per group per ensemble
    full_loss_val = np.zeros([config.EPOCHS])
    accuracy_s_val = np.zeros([config.EPOCHS, config.n_sensitive])

    mu_penalty_all = np.zeros([config.EPOCHS, config.mu_penalty.shape[0]])
    learning_rate_all = np.zeros([config.EPOCHS])
    ################################################

    stop = False
    epoch = 0
    epoch_best = 0
    lrdecay = config.lrdecay + 0
    #----------------------------------------

    while (not stop) & (epoch < config.EPOCHS):

        mu_penalty_all[epoch,:] = to_np(config.mu_penalty)
        learning_rate_all[epoch] = optimizer.param_groups[0]['lr'] + 0

        ### TRAIN ###
        base_loss_all_train[epoch, ...], accuracy_s_train[epoch, ...], full_loss_train[epoch] = epoch_training_linearweight(
            train_dataloader, optimizer, classifier_network, criterio,
            config, config.logger, train_type='Train')

        ### VALIDATION ###
        base_loss_all_val[epoch, ...], accuracy_s_val[epoch, ...], full_loss_val[epoch] = epoch_training_linearweight(
            val_dataloader, optimizer, classifier_network, criterio,
            config, config.logger, train_type='Val')

        ######### Check Stopping Criteria #################################################

        if epoch == 0:
            stopper = early_stopping(config.patience, 0, full_loss_val[0])
            model_params_save(config.best_adaptive_network_path, classifier_network, optimizer)
            best_base_loss_val = base_loss_all_val[epoch, ...] + 0
            best_accuracy_val = accuracy_s_val[epoch, ...] + 0
            epoch_best = epoch+1
        else:
            save, stop = stopper.evaluate(full_loss_val[epoch])

            if save:
                model_params_save(config.best_adaptive_network_path, classifier_network, optimizer)
                best_base_loss_val = base_loss_all_val[epoch, ...]+0
                best_accuracy_val = accuracy_s_val[epoch, ...]+0
                lrdecay = config.lrdecay + 0 #reset lrdecay
                epoch_best = epoch + 1

            if (stopper.best_loss < full_loss_val[epoch]):
                model_params_load(config.best_adaptive_network_path, classifier_network, optimizer, config.DEVICE) #loading best last
                optimizer.param_groups[0]['lr'] *= lrdecay #apply lrdecay
                lrdecay *= config.lrdecay #update lrdecay


        ## PRINT ##
        if ((epoch % config.n_print == config.n_print - 1) & (epoch >= 1))| (epoch == 0):
            print('Epoch: ' + str(epoch) +'; lr: '+ str(optimizer.param_groups[0]['lr'])+
                  ';loss tr: ' + str(np.round(full_loss_train[epoch], 3)) +
                  ',val: ' + str(np.round(full_loss_val[epoch], 3)) +
                  '|base_val : ' + str(np.round(base_loss_all_val[epoch, ...], 3)) +
                  '|acc_val : ' + str(np.round(accuracy_s_val[epoch, ...], 3)) +
                  '|stop_c : ' + str(stopper.counter))

        if epoch == (config.EPOCHS - 1):
            stop = True
            print('stop')

        if stop:
            print('________End adaptiveOptimizer for penalty ::: '+str(config.mu_penalty))
            print('Best base loss val ', str((best_base_loss_val)))
            print('Best accuracy val ', str(best_accuracy_val))
            print('________________________________________________________')

        epoch += 1

    #################################################
    #load best network
    model_params_load(config.best_adaptive_network_path, classifier_network, optimizer, config.DEVICE)

    # Base loss dic
    base_loss_dic = {}
    base_loss_dic['train'] = base_loss_all_train[0:epoch_best, ...]
    base_loss_dic['val'] = base_loss_all_val[0:epoch_best, ...]

    # full_loss dic
    full_loss_dic = {}
    full_loss_dic['train'] = full_loss_train[0:epoch_best, ...]
    full_loss_dic['val'] = full_loss_val[0:epoch_best, ...]

    # accuracy_dic
    accuracy_s_dic = {}
    accuracy_s_dic['train'] = accuracy_s_train[0:epoch_best, ...]
    accuracy_s_dic['val'] = accuracy_s_val[0:epoch_best, ...]

    opt_params_dic = {}
    opt_params_dic['mu_penalty'] = mu_penalty_all[0:epoch_best,...]
    opt_params_dic['learning_rate_all'] = learning_rate_all[0:epoch_best]

    return base_loss_dic, full_loss_dic, accuracy_s_dic, opt_params_dic, best_base_loss_val

def fast_epoch_evaluation(dataloader, classifier_network,config):

    # loss summary lists
    utility_pred_l = [[] for _ in range(config.n_utility)]
    utility_gt_l = []
    secret_gt_l = []
    classifier_network = classifier_network.eval()

    # Loop through samples and evaluate
    for i_batch, sample_batch in enumerate(dataloader):
        x, utility, secret = sample_batch
        x = x.to(config.DEVICE)
        utility = utility.to(config.DEVICE)
        secret = secret.to(config.DEVICE)

        # forward pass
        logits = classifier_network(x)
        softmax = nn.Softmax(dim=-1)(logits)

        ### SAVES FOR VISUALIZATION ###
        softmax_np = to_np(softmax)
        # print(softmax)
        for _ in range(config.n_utility):
            utility_pred_l[_].extend(list(softmax_np[:,_]))

        utility_gt_l.extend(list(to_np(utility).argmax(-1)))
        secret_gt_l.extend(list(to_np(secret).argmax(-1)))

    return utility_pred_l, utility_gt_l, secret_gt_l

## Saves ##
def model_save(filename,classifier_network,criteria,optimizer):
    with open(filename, 'wb') as f:
        torch.save([classifier_network,criteria,optimizer], f)

def model_load(filename):
    with open(filename, 'rb') as f:
        classifier_network,criteria, optimizer = torch.load(f)
    return classifier_network,criteria,optimizer

def model_params_save(filename,classifier_network, optimizer):
    torch.save([classifier_network.state_dict(),optimizer.state_dict()], filename)

def model_params_load(filename,classifier_network, optimizer,DEVICE):
    classifier_dic, optimizer_dic = torch.load(filename, map_location=DEVICE)
    classifier_network.load_state_dict(classifier_dic)
    optimizer.load_state_dict(optimizer_dic)
