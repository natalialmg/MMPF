import pandas as pd
import numpy as np

def get_brier_score(y_gt,y_est):
    return np.mean(np.sum((y_gt - y_est)**2,axis = 1))

def get_cross_entropy(y_gt,y_est):
    return np.mean(-1*np.sum(y_gt*np.log(np.maximum(y_est,1e-20)),axis = 1))

def get_accuracy(y_gt,y_est):
    ixmax_gt = np.argmax(y_gt,axis=1)
    ixmax_est = np.argmax(y_est,axis=1)
    return np.mean(ixmax_gt == ixmax_est)

def get_confidence(y_est):
    ixmax_gt = np.argmax(y_est,axis=1)
    return np.mean(y_est[np.arange(y_est.shape[0]),ixmax_gt])

def results_dataframe(pd_eval, dataset_tag, model_tag='MMPF', split=0, settype='test'):

    # data frame full
    tags_params = ['dataset', 'model', 'split', 'set', 'group', 'ratio']
    tags_metrics = ['ACC', 'BS', 'CE', 'Entropy', 'ECE', 'MCE', 'ACC_prior']
    tags_params.extend(tags_metrics)
    pd_data = pd.DataFrame(columns=tags_params)

    util_values = np.unique(pd_eval['utility_gt'].values)
    nutil = util_values.shape[0]

    group_values = np.unique(pd_eval['secret_gt'].values)
    ngroup = group_values.shape[0]
    ntotal = 0
    cont = 0

    for group in group_values:

        # filter by group
        pd_group = pd_eval.loc[pd_eval['secret_gt'] == group]
        table_data = pd_group.values + 0
        nsamples = np.sum(pd_eval['secret_gt'] == group)
        ntotal += nsamples

        # get gt and estimated y
        y_est = np.zeros([len(pd_group), nutil])
        for util in util_values.astype('int'):
            y_est[:, util] = pd_group['utility_pest_' + str(np.round(util, 0))].values
        y_gt = np.zeros(y_est.shape)
        y_gt[np.arange(y_gt.shape[0]), pd_group['utility_gt'].values.astype(int)] = 1

        # scores
        briers = get_brier_score(y_gt, y_est)
        cross_entropy = get_cross_entropy(y_gt, y_est)
        accuracy = get_accuracy(y_gt, y_est)
        ent = np.mean(entropy(y_est))
        accuracy_prior = np.max(np.sum(y_gt, axis=0) / np.sum(y_gt))

        y_est = np.maximum(y_est, 0)
        y_est /= np.sum(y_est, axis=1)[:, np.newaxis]
        y_gt = np.maximum(y_gt, 0)
        y_gt /= np.sum(y_gt, axis=1)[:, np.newaxis]

        try:
            cal = calibration(y_gt, y_est, num_bins=10)
        except ZeroDivisionError:
            print('GROUP :: ' + str(group))
            print('Division zero in calibration problem!')
            break

        ece = cal['ece']
        mce = cal['mce']

        row_i = [dataset_tag, model_tag, split, settype, int(group), nsamples]
        row_i.extend([accuracy, briers, cross_entropy, ent, ece, mce, accuracy_prior])
        pd_data.loc[cont] = row_i
        cont += 1

    pd_data['ratio'][:] = pd_data['ratio'][:]/ntotal

    return pd_data

import scipy.stats as spstats
#The Following Code is from Google-Research github :
# https://github.com/google-research/google-research/blob/master/uncertainties/sources/postprocessing/metrics.py

def entropy(p_mean):
  """Compute the entropy.
  Args:
    p_mean: numpy array, size (?, num_classes, ?)
           containing the (possibly mean) output predicted probabilities
  Returns:
    ent: entropy along the iterations, numpy vector of size (?, ?)
  """
  ent = np.apply_along_axis(spstats.entropy, axis=1, arr=p_mean)
  return ent


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    cal: a dictionary
      {reliability_diag: realibility diagram
       ece: Expected Calibration Error
       mce: Maximum Calibration Error
      }
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  # print(conf)
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    if (i == num_bins-1):
        # print(tau_tab[i], tau_tab[i + 1])
        sec = (tau_tab[i + 1] >= conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]
  # print(nb_items_bin)

  # Reliability diagram
  reliability_diag = (mean_conf, acc_tab)
  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  # Saving
  cal = {'reliability_diag': reliability_diag,
         'ece': ece,
         'mce': mce}
  return cal