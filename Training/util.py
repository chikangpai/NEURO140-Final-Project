import pandas as pd
from sklearn import metrics
import numpy as np
import random
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import loralib as lora
import torch.nn.functional as F
import math
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
# argument library
import argparse
from copy import deepcopy
from typing import Literal
import glob
import os
import re
from fnmatch import translate as regex_glob

def list_files(directories, pattern='*.pt'):
    '''
    List all files in the directories that match the pattern
    Input:
        directories: list of directories, or a single directory
        pattern: pattern to match
    '''
    if isinstance(directories, str):
        directories = [directories]
    files = []
    for directory in directories:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return files
# ### ================================================== ####

def load_weights(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

# ### ================================================== ####

def replace_linear(model, rank=4, dropout=0.3):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, lora.Linear(module.in_features,
                    module.out_features, r=rank, lora_dropout=dropout))
        else:
            replace_linear(module)

def loss_function(loss_func, preds, targets, group):
    ##
    loss = 0.
    group_of_loss = {}
    group_length = {}
    ##
    loss_func_noavg = deepcopy(loss_func)
    loss_func_noavg.reduction = 'none'
    loss = loss_func_noavg(preds, targets)
    # get unique groups
    unique_groups = list(set(group))
    ##
    for g in unique_groups:
        idx = [i for i, x in enumerate(group) if x == g]
        group_of_loss[g] = loss[idx].mean()
    loss = loss.mean()

    return loss, group_of_loss

# ### =================================================== ####


def batch_resample(batch_size, group_loss):
    group_weight = {}
    group_sampling_probability = {}
    group_samples = {}
    for group in group_loss:
        group_weight[group] = 1 / group_loss[group]
    for group in group_weight:
        group_sampling_probability[group] = group_weight[group] / \
            sum(group_weight.values())
        if math.isnan(group_sampling_probability[group]):
            group_samples[group] = 0
        else:
            group_samples[group] = int(
                group_sampling_probability[group] * batch_size)
        if (group_samples[group] == 0):
            group_samples[group] = 1

    total = sum(group_samples.values())
    if (total < batch_size):
        group_samples[max(group_samples, key=group_samples.get)
                      ] += batch_size - total
    elif (total > batch_size):
        group_samples[max(group_samples, key=group_samples.get)
                      ] -= total - batch_size
    # print('group_samples_after: ', group_samples)
    return group_samples