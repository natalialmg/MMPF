#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *

class Logger(object):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        self.log_level = log_level
        self.writer = None
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}
        self.log_dir = log_dir

    def lazy_init_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        self.lazy_init_writer()
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

def get_logger(tag='default',dir = None, log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    mkdir('./log')
    mkdir('./tf_log')
    if dir is None:
        dir = './tf_log/'
    if tag is not None:
        fh = logging.FileHandler('./log/%s-%s.txt' % (tag, get_time_str()))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    print('logger',dir+'%s-%s' % (tag, get_time_str()))

    return Logger(logger, dir+'%s-%s' % (tag, get_time_str()), log_level)

def instantiate_logger(config):
    if config.logger_active:
        logger = get_logger(tag=config.save_file_logger,dir=config.loggdir, log_level=0)
        layout = {}

        for measure in ['base_loss_sec', 'accuracy_sec']:
            auxiliary_layout = {}
            for train_type in ['Train', 'Val']:
                aux_l = []
                for s in np.arange(config.n_sensitive):
                    aux_l.append('{:s}/{:s}/{:d}'.format(measure, train_type, s))
                auxiliary_layout['{:s}/{:d}'.format(train_type, s)] = ['Multiline', aux_l]
            layout['ML_{:s}'.format(measure)] = auxiliary_layout

        for measure in ['mu_penalty']:
            auxiliary_layout = {}
            aux_l = []
            train_type = 'Train'
            for s in np.arange(config.mu_penalty.shape[0]):
                aux_l.append('{:s}/{:s}/{:d}'.format(measure, train_type, s))
            auxiliary_layout['mu_penalty'] = ['Multiline', aux_l]
            layout['ML_{:s}'.format(measure)] = auxiliary_layout

        logger.lazy_init_writer()
        logger.writer.add_custom_scalars(layout)

        config.logger = logger
    else:
        config.logger = None
    return config



