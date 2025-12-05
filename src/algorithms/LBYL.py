from __future__ import print_function
import argparse
import pickle
import numpy as np
# NOTE: Heavy sklearn imports are moved into functions to reduce import-time overhead
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
import sys
import os
import scipy
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
# from sklearn import linear_model  # moved to function scope
# import statsmodels.api as sm
# import cvxpy as cp
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
from scipy.optimize import least_squares
from typing import Callable
from scipy.optimize import lsq_linear
from . import Ridge

cwd = os.getcwd()
sys.path.append(cwd + '/../')




class Coreset:
    def __init__(self, points, weights, activation_function: Callable, upper_bound: int = 1):
        assert points.shape[0] == weights.shape[1]

        self.__points = points.cpu()
        self.__weights = weights.cpu()
        self.__activation = activation_function
        self.__beta = upper_bound
        self.__sensitivity = None
        self.indices = None

    # @property
    def sensitivity(self):
        if self.__sensitivity is None:
            points_norm = self.__points.reshape(self.__points.shape[0],-1).norm(dim=1)
            assert points_norm.shape[0] == self.__points.shape[0]
            weights = torch.abs(self.__weights.reshape(self.__weights.shape[1],-1)).max(dim=1)[0]  # max returns (values, indices)
            assert weights.shape[0] == self.__points.shape[0]
            self.__sensitivity = weights * torch.abs(self.__activation(self.__beta * points_norm))
            self.__sensitivity /= self.__sensitivity.sum()

        return self.__sensitivity

    def compute_coreset(self, coreset_size):
        assert coreset_size <= self.__points.shape[0]
        prob = np.array(self.sensitivity())#  .cpu().numpy()
        prob /= prob.sum()
        points = self.__points
        indices = set()
        idxs = []


        cnt = 0
        while len(indices) < coreset_size:
            i = np.random.choice(a=points.shape[0], size=1, p=prob).tolist()[0]
            idxs.append(i)
            indices.add(i)
            cnt += 1

        hist = np.histogram(idxs, bins=range(points.shape[0] + 1))[0].flatten()
        idxs = np.nonzero(hist)[0]
        self.indices = idxs
        coreset = points[idxs, :, :, :]

        for i in idxs:
            self.__weights[i, :, :, :] = (self.__weights[i, :, :, :] * torch.tensor(hist[i]).float()) / (
                        torch.tensor(prob[i]) * cnt)

        return coreset, self.__weights[:,idxs,:,:]



def create_scaling_mat_ip_thres_bias(weight, ind, threshold, model_type, lam=0, lam_2=0):
    '''
    weight - 2D matrix (n_{i+1}, n_i), np.ndarray
    ind - chosen indices to remain, np.ndarray
    threshold - cosine similarity threshold
    '''
    assert (type(weight) == np.ndarray)
    assert (type(ind) == np.ndarray)

    from sklearn.metrics import pairwise_distances
    from sklearn import linear_model
    cosine_sim = 1 - pairwise_distances(weight, metric="cosine")
    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind:  # chosen
            ind_i, = np.where(ind == i)
            assert (len(ind_i) == 1)  # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else:  # not chosen
            if model_type == 'prune':
                continue

            elif model_type == 'merge':
                max_cos_value = np.max(cosine_sim[i][ind])
                max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]

                if threshold and max_cos_value < threshold:
                    continue

                baseline_weight = weight_chosen[max_cos_value_index]
                current_weight = weight[i]
                baseline_norm = np.linalg.norm(baseline_weight)
                current_norm = np.linalg.norm(current_weight)
                scaling_factor = current_norm / baseline_norm
                scaling_mat[i, max_cos_value_index] = scaling_factor
                scaling_mat[i, max_cos_value_index] = scaling_factor

            elif model_type == 'OURS':
                preserved_weights = []
                for chosen_i in ind:
                    preserved_weights.append(weight[chosen_i])
                preserved_weights = np.array(preserved_weights).T
                alpha_scale_list = [1] * len(ind)

                linear_clf = linear_model.Ridge(alpha=lam_2, fit_intercept= True)# , positive= True)
                linear_clf.fit(np.array(preserved_weights), weight[i])
                alpha_list = linear_clf.coef_


                # linear_cif = Ridge.Ridge_Regression(np.array(preserved_weights), weight[i], alpha=[lam, lam_2],
                #                                     fit_intercept=True, alpha_scale_list=alpha_scale_list, # fit_intercept=False
                #                                     K_scale=1, sigma_1=1,
                #                                     gamma_1=1, mu_1=1, beta_1=1)
                # alpha_list = linear_cif.fit()

                for index, chosen_i in enumerate(ind):
                    scaling_mat[i, index] = alpha_list[index]

    return scaling_mat # np.where(scaling_mat>= 0,scaling_mat,0) #


def create_scaling_mat_conv_thres_bn(weight, ind, threshold,
                                     bn_weight, bn_bias,
                                     bn_mean, bn_var, lam, model_type):
    '''
    weight - 4D tensor(n, c, h, w), np.ndarray
    ind - chosen indices to remain
    threshold - cosine similarity threshold
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam - how much to consider cosine sim over bias, float value between 0 and 1
    '''
    assert (type(weight) == np.ndarray)
    assert (type(ind) == np.ndarray)
    assert (type(bn_weight) == np.ndarray)
    assert (type(bn_bias) == np.ndarray)
    assert (type(bn_mean) == np.ndarray)
    assert (type(bn_var) == np.ndarray)
    assert (bn_weight.shape[0] == weight.shape[0])
    assert (bn_bias.shape[0] == weight.shape[0])
    assert (bn_mean.shape[0] == weight.shape[0])
    assert (bn_var.shape[0] == weight.shape[0])

    weight = weight.reshape(weight.shape[0], -1)

    from sklearn.metrics import pairwise_distances
    cosine_dist = pairwise_distances(weight, metric="cosine")

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])  # 16,11

    for i in range(weight.shape[0]):  # 16
        if i in ind:  # chosen
            ind_i, = np.where(ind == i)
            assert (len(ind_i) == 1)  # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else:  # not chosen

            if model_type == 'prune':
                continue

            current_weight = weight[i]
            current_norm = np.linalg.norm(current_weight)
            current_cos = cosine_dist[i]
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]

            # choose one
            cos_list = []
            scale_list = []
            bias_list = []

            for chosen_i in ind:
                chosen_weight = weight[chosen_i]
                chosen_norm = np.linalg.norm(chosen_weight, ord=2)
                chosen_cos = current_cos[chosen_i]
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]

                # compute cosine sim
                cos_list.append(chosen_cos)

                # compute s
                s = current_norm / chosen_norm

                # compute scale term
                scale_term_inference = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
                scale_list.append(scale_term_inference)

                # compute bias term
                bias_term_inference = abs(
                    (gamma_2 / sigma_2) * (s * (-(sigma_1 * beta_1 / gamma_1) + mu_1) - mu_2) + beta_2)

                bias_term_inference = bias_term_inference / scale_term_inference

                bias_list.append(bias_term_inference)

            assert (len(cos_list) == len(ind))
            assert (len(scale_list) == len(ind))
            assert (len(bias_list) == len(ind))

            # merge cosine distance and bias distance
            bias_list = (bias_list - np.min(bias_list)) / (np.max(bias_list) -
                                                           np.min(bias_list))

            score_list = lam * np.array(cos_list) + (1 - lam) * np.array(bias_list)

            # find index and scale with minimum distance
            min_ind = np.argmin(score_list)

            min_scale = scale_list[min_ind]
            min_cosine_sim = 1 - cos_list[min_ind]

            # check threshold - second
            if threshold and min_cosine_sim < threshold:
                continue

            scaling_mat[i, min_ind] = min_scale

    return scaling_mat


def ours_create_scaling(weight, ind, bn_weight, bn_bias,
                        bn_mean, bn_var, lam, lam_2):

    weight = weight.reshape(weight.shape[0], -1)  # (16, -1)

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])  # 16,11


    for i in range(weight.shape[0]):  # 16
        if i in ind:  # chosen
            ind_i, = np.where(ind == i)
            assert (len(ind_i) == 1)  # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else:  # not chosen

            # choose one
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]
            preserved_weights = []

            for chosen_i in ind:
                preserved_weights.append(weight[chosen_i])
            preserved_weights = np.array(preserved_weights).T
            alpha_scale_list = []
            K_scale = []

            for index, chosen_i in enumerate(ind):
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]
                alpha_scale = (mu_2 - (sigma_2 * beta_2 / gamma_2))
                alpha_scale_list.append(alpha_scale)
                K_scale.append((gamma_1 / gamma_2) * (sigma_2 / sigma_1))

            linear_cif = Ridge.Ridge_Regression(np.array(preserved_weights), weight[i], alpha=[lam, lam_2],
                                                fit_intercept=False, alpha_scale_list=alpha_scale_list,K_scale = K_scale,sigma_1=sigma_1,
                                                gamma_1=gamma_1, mu_1=mu_1, beta_1=beta_1)
            alpha_list = linear_cif.fit()

            for index, chosen_i in enumerate(ind):
                # gamma_2 = bn_weight[chosen_i]
                # sigma_2 = bn_var[chosen_i]
                scaling_mat[i, index] = alpha_list[index]

    return scaling_mat


class Decompose:
    def __init__(self, arch, param_dict, criterion, threshold, lamda, lamda_2, model_type, cfg, cuda):

        self.param_dict = param_dict
        self.arch = arch
        self.criterion = criterion
        self.threshold = threshold
        self.lamda = lamda
        self.lamda_2 = lamda_2
        self.model_type = model_type
        self.cfg = cfg
        self.cuda = cuda
        self.output_channel_index = {}
        self.pruned_channel_index = {}
        self.decompose_weight = []
        self.conv1_norm_dictionary = dict()
        self.conv2_norm_dictionary = dict()


    def get_output_channel_index(self, value, layer_id):

        output_channel_index = []

        if len(value.size()):
            weight_vec = value.view(value.size()[0], -1)
            weight_vec = weight_vec.cuda()

            # l1-norm
            if self.criterion == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-norm
            elif self.criterion == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-GM
            elif self.criterion == 'l2-GM':
                weight_vec = weight_vec.cpu().detach().numpy()
                matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(np.abs(matrix), axis=0)

                output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]
                output_channel_index.sort()

            elif self.criterion == 'random_1':
                summation = torch.sum(weight_vec, dim=1).tolist()
                random_list = []
                for val in summation:
                    val_to_str = str(val)[-1]
                    random_list.append(int(val_to_str))
                sorted_random_list = np.argsort(random_list)
                select_index = sorted_random_list[:self.cfg[layer_id]]
                output_channel_index = select_index
                
            elif self.criterion == 'random_2': # first
                select_index = list(range(weight_vec.shape[0]))[::-1][:self.cfg[layer_id]]
                output_channel_index = select_index
                
            elif self.criterion == 'random_3': # last
                select_index = list(range(weight_vec.shape[0]))[:self.cfg[layer_id]]
                output_channel_index = select_index            
            
            
            
        pruned_channel_index = list(set(list(range(weight_vec.shape[0]))) - set(output_channel_index))

        return output_channel_index, np.array(pruned_channel_index)

    def compress_conv_layer(self, layer1, layer2, compressed_size, activation: Callable, upper_bound, device):

        coreset = Coreset(points=layer1, weights=layer2, activation_function=activation, upper_bound=upper_bound)
        points, weights = coreset.compute_coreset(compressed_size)
        indices = coreset.indices
        layer1 = points.to(device)
        layer2 = weights.to(device)
        return layer1, layer2, indices


    def get_decompose_weight(self):

        # scale matrix
        z = None

        # copy original weight
        self.decompose_weight = list(self.param_dict.values())

        # cfg index
        layer_id = -1
        vgg_conv_fc_list = []
        for key in  self.param_dict.keys():
            if len(self.param_dict[key].shape) == 4 or len(self.param_dict[key].shape)==2:
                vgg_conv_fc_list.append(key)

        vgg_conv_fc_list = vgg_conv_fc_list[:-1]

        for index, layer in enumerate(self.param_dict):

            original = self.param_dict[layer]

            # LeNet_300_100
            if self.arch == 'LeNet_300_100':
                pass
                # ip
                if layer in ['ip1.weight', 'ip2.weight']:

                    # Merge scale matrix
                    if z != None:
                        original = torch.mm(original, z)

                    layer_id += 1

                    # concatenate weight and bias
                    if layer in 'ip1.weight':
                        weight = self.param_dict['ip1.weight'].cpu().detach().numpy()
                        bias = self.param_dict['ip1.bias'].cpu().detach().numpy()

                    elif layer in 'ip2.weight':
                        weight = self.param_dict['ip2.weight'].cpu().detach().numpy()
                        bias = self.param_dict['ip2.bias'].cpu().detach().numpy()

                    bias_reshaped = bias.reshape(bias.shape[0], -1)
                    concat_weight = np.concatenate([weight, bias_reshaped], axis=1)

                    # get index
                    self.output_channel_index[index], _ = self.get_output_channel_index(torch.from_numpy(concat_weight),
                                                                                     layer_id)

                    # make scale matrix with bias
                    x = create_scaling_mat_ip_thres_bias(concat_weight, np.array(self.output_channel_index[index]),
                                                         self.threshold, self.model_type, self.lamda, self.lamda_2)
                    z = torch.from_numpy(x).type(dtype=torch.float)

                    if self.cuda:
                        z = z.cuda()

                    # pruned
                    pruned = original[self.output_channel_index[index], :]

                    # update next input channel
                    input_channel_index = self.output_channel_index[index]

                    # update decompose weight
                    self.decompose_weight[index] = pruned

                elif layer in 'ip3.weight':

                    # Ensure device match between operands
                    original = original.to(z.device)
                    original = torch.mm(original, z)

                    # update decompose weight
                    self.decompose_weight[index] = original

                # update bias
                elif layer in ['ip1.bias', 'ip2.bias']:
                    self.decompose_weight[index] = original[input_channel_index]

                else:
                    pass


                    # VGG
            elif self.arch == 'VGG16':

                # feature
                if 'feature' in layer:

                    # conv
                    if len(self.param_dict[layer].shape) == 4:

                        layer_id += 1

                        # get index
                        self.output_channel_index[index], self.pruned_channel_index[
                            index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # Merge scale matrix
                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            # Move original to the same device as z for in-place updates
                            original = original.to(z.device)
                            for i, f in enumerate(self.param_dict[layer]):
                                # Compute on the same device as z
                                o = f.view(f.shape[0], -1).to(z.device)
                                o = torch.mm(z, o)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o

                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        bn_weight = bn[index + 1].cpu().detach().numpy()
                        bn_bias = bn[index + 2].cpu().detach().numpy()
                        bn_mean = bn[index + 3].cpu().detach().numpy()
                        bn_var = bn[index + 4].cpu().detach().numpy()

                        if self.model_type == 'merge' or self.model_type == 'prune':
                            z = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)


                        elif self.model_type == 'OURS':
                            z = ours_create_scaling(
                                self.param_dict[layer].cpu().detach().numpy(),
                                np.array(self.output_channel_index[index]),
                                bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.lamda_2)

                        z = torch.from_numpy(z).type(dtype=torch.float)
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index], :, :, :]
                        # update next input channel
                        input_channel_index = self.output_channel_index[index]
                        # update decompose weight
                        self.decompose_weight[index] = pruned


                    # batchNorm
                    elif len(self.param_dict[layer].shape):

                        # pruned
                        pruned = self.param_dict[layer][input_channel_index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned


                # first classifier
                else:

                    # Allocate on the same device as z to avoid device mismatch
                    pruned = torch.zeros(original.shape[0], z.shape[0], device=z.device)

                    for i, f in enumerate(original):
                        o_old = f.view(z.shape[1], -1).to(z.device)
                        o = torch.mm(z, o_old).view(-1)
                        pruned[i, :] = o
                    self.decompose_weight[index] = pruned

                    break

            # ResNet50
            elif self.arch == 'ResNet50':
                # block
                if 'layer' in layer:

                    # last layer each block
                    if 'block_1.conv1.weight' in layer:
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:

                        # get index
                        self.output_channel_index[index], self.pruned_channel_index[
                            index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        if self.model_type == 'merge':
                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)
                            z = torch.from_numpy(x).type(dtype=torch.float)
                            if self.cuda:
                                z = z.cuda()

                            z = z.t()

                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]
                            # update next input channel
                            input_channel_index = self.output_channel_index[index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'prune':
                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'coreset':
                            next_layer = layer[:19] + '2' + layer[20:]
                            conv_layer1 = self.param_dict[layer]
                            conv_layer2 = self.param_dict[next_layer]

                            layer1, _layer2, indices = self.compress_conv_layer(layer1=conv_layer1,
                                                                         layer2=conv_layer2,
                                                                         compressed_size = self.cfg[layer_id],
                                                                         activation=torch.nn.ReLU(), upper_bound=1,
                                                                         device='cuda')
                            self.decompose_weight[index] = layer1
                            input_channel_index = indices


                            ############adasdasofsofsokp

                        elif self.model_type == 'OURS':

                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            scale = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                          bn_weight,bn_bias ,bn_mean,bn_var, self.lamda, self.lamda_2)
                            scale_mat = scale.T
                            scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)

                            if self.cuda:
                                scale_mat = scale_mat.cuda()

                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned

                    # batchNorm
                    elif 'bn1' in layer:

                        if len(self.param_dict[layer].shape):
                            pruned = self.param_dict[layer][input_channel_index]
                            self.decompose_weight[index] = pruned


                    # Merge scale matrix
                    elif 'conv2' in layer:
                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())
                        bn_weight = bn[index + 1].cpu().detach().numpy()
                        bn_bias = bn[index + 2].cpu().detach().numpy()
                        bn_mean = bn[index + 3].cpu().detach().numpy()
                        bn_var = bn[index + 4].cpu().detach().numpy()


                        if z != None and self.model_type == 'merge':

                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(z, o)  # (11,16) * (16, 9)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                            x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)
                            z = torch.from_numpy(x).type(dtype=torch.float)
                            if self.cuda:
                                z = z.cuda()
                            z = z.t()
                            self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]

                            input_channel_index = self.output_channel_index[index]

                        elif self.model_type == 'prune':
                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                            _original = original[:, input_channel_index, :, :]
                            self.decompose_weight[index] = _original[self.output_channel_index[index], :, :, :]
                            input_channel_index = self.output_channel_index[index]

                        elif self.model_type == 'coreset':
                            next_layer = layer[:19] + '3' + layer[20:]
                            conv_layer2 = _layer2
                            # conv_layer2 = self.param_dict[layer]
                            conv_layer3 = self.param_dict[next_layer]

                            layer2, _layer3, indices = self.compress_conv_layer(layer1=conv_layer2,
                                                                                layer2=conv_layer3,
                                                                                compressed_size=self.cfg[layer_id],
                                                                                activation=torch.nn.ReLU(),
                                                                                upper_bound=1,
                                                                                device='cuda')
                            self.decompose_weight[index] = layer2
                            input_channel_index = indices


                        elif self.model_type == 'OURS':

                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
                                o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                            scale = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                          bn_weight,bn_bias ,bn_mean,bn_var, self.lamda, self.lamda_2)
                            scale_mat = scale.T
                            scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)

                            if self.cuda:
                                scale_mat = scale_mat.cuda()

                            self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]


                            input_channel_index = self.output_channel_index[index]


                    elif 'bn2' in layer:
                        if len(self.param_dict[layer].shape):
                            pruned = self.param_dict[layer][input_channel_index]
                            self.decompose_weight[index] = pruned

                    elif 'conv3' in layer:

                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0], -1)
                                o = torch.mm(z, o)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # update decompose weight
                            self.decompose_weight[index] = scaled

                        elif self.model_type == 'prune':
                            self.decompose_weight[index] = original[:, input_channel_index, :, :]

                        elif self.model_type =='coreset':
                            self.decompose_weight[index] = _layer3


                        elif self.model_type == 'OURS':
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(
                                    self.param_dict[layer]):
                                o = f.view(f.shape[0], -1)
                                o = torch.mm(scale_mat, o)
                                o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # # update decompose weight
                            self.decompose_weight[index] = scaled



            elif self.arch == 'ResNet34':

                # block
                if 'layer' in layer:

                    # last layer each block
                    if '0.group1.conv1.weight' in layer:
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:

                        # get index
                        self.output_channel_index[index], self.pruned_channel_index[
                            index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        if self.model_type == 'merge':
                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)
                            z = torch.from_numpy(x).type(dtype=torch.float)
                            if self.cuda:
                                z = z.cuda()

                            z = z.t()

                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]
                            # update next input channel
                            input_channel_index = self.output_channel_index[index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'prune':
                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'OURS':

                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            scale = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                          bn_weight,bn_bias ,bn_mean,bn_var, self.lamda,
                                                                                    self.lamda_2)

                            scale_mat = scale.T
                            scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)

                            if self.cuda:
                                scale_mat = scale_mat.cuda()

                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned


                    # batchNorm
                    elif 'bn1' in layer:

                        if len(self.param_dict[layer].shape):
                            pruned = self.param_dict[layer][input_channel_index]
                            self.decompose_weight[index] = pruned


                    # Merge scale matrix
                    elif 'conv2' in layer:

                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(
                                    self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(z, o)  # (11,16) * (16, 9)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # update decompose weight
                            self.decompose_weight[index] = scaled

                        elif self.model_type == 'prune':
                            self.decompose_weight[index] = original[:, input_channel_index, :, :]

                        elif self.model_type == 'OURS':
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(
                                    self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
                                o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # # update decompose weight
                            self.decompose_weight[index] = scaled

            elif self.arch == 'ResNet101':

                # block
                if 'layer' in layer:

                    # last layer each block
                    if '.0.group1.conv1.weight' in layer:
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:

                        # get index
                        self.output_channel_index[index], self.pruned_channel_index[
                            index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        if self.model_type == 'merge':
                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)
                            z = torch.from_numpy(x).type(dtype=torch.float)
                            if self.cuda:
                                z = z.cuda()

                            z = z.t()

                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]
                            # update next input channel
                            input_channel_index = self.output_channel_index[index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'prune':
                            # pruned
                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned

                        elif self.model_type == 'OURS':

                            bn = list(self.param_dict.values())

                            bn_weight = bn[index + 1].cpu().detach().numpy()
                            bn_bias = bn[index + 2].cpu().detach().numpy()
                            bn_mean = bn[index + 3].cpu().detach().numpy()
                            bn_var = bn[index + 4].cpu().detach().numpy()

                            scale  = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                          bn_weight,bn_bias ,bn_mean,bn_var, self.lamda,
                                                                                    self.lamda_2)


                            scale_mat = scale.T
                            scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)

                            if self.cuda:
                                scale_mat = scale_mat.cuda()

                            pruned = original[self.output_channel_index[index], :, :, :]  # 11,16,3,3
                            input_channel_index = self.output_channel_index[index]
                            # update decompose weight
                            self.decompose_weight[index] = pruned

                    # batchNorm
                    elif 'bn1' in layer:

                        if len(self.param_dict[layer].shape):
                            pruned = self.param_dict[layer][input_channel_index]
                            self.decompose_weight[index] = pruned


                    # Merge scale matrix
                    elif 'conv2' in layer:
                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())
                        bn_weight = bn[index + 1].cpu().detach().numpy()
                        bn_bias = bn[index + 2].cpu().detach().numpy()
                        bn_mean = bn[index + 3].cpu().detach().numpy()
                        bn_var = bn[index + 4].cpu().detach().numpy()


                        if z != None and self.model_type == 'merge':

                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(z, o)  # (11,16) * (16, 9)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                            x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                 self.threshold,
                                                                 bn_weight, bn_bias, bn_mean, bn_var, self.lamda,
                                                                 self.model_type)
                            z = torch.from_numpy(x).type(dtype=torch.float)
                            if self.cuda:
                                z = z.cuda()
                            z = z.t()
                            self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]
                            input_channel_index = self.output_channel_index[index]


                        elif self.model_type == 'prune':
                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                            _original = original[:, input_channel_index, :, :]
                            self.decompose_weight[index] = _original[self.output_channel_index[index], :, :, :]
                            input_channel_index = self.output_channel_index[index]

                        elif self.model_type == 'OURS':

                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
                                o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            self.output_channel_index[index], self.pruned_channel_index[
                                index] = self.get_output_channel_index(self.param_dict[layer], layer_id)


                            scale  = ours_create_scaling(self.param_dict[layer].cpu().detach().numpy(),
                                                                 np.array(self.output_channel_index[index]),
                                                                          bn_weight,bn_bias ,bn_mean,bn_var, self.lamda,
                                                                                    self.lamda_2)

                            scale_mat = scale.T
                            scale_mat = torch.from_numpy(scale_mat).type(dtype=torch.float)

                            if self.cuda:
                                scale_mat = scale_mat.cuda()

                            self.decompose_weight[index] = scaled[self.output_channel_index[index],:,:,:]

                            input_channel_index = self.output_channel_index[index]


                    elif 'bn2' in layer:
                        if len(self.param_dict[layer].shape):
                            pruned = self.param_dict[layer][input_channel_index]
                            self.decompose_weight[index] = pruned

                    elif 'conv3' in layer:

                        if z != None:
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(z, o)  # (11,16) * (16, 9)
                                o = o.view(z.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # update decompose weight
                            self.decompose_weight[index] = scaled

                        elif self.model_type == 'prune':
                            self.decompose_weight[index] = original[:, input_channel_index, :, :]

                        elif self.model_type == 'OURS':
                            original = original[:, input_channel_index, :, :]
                            for i, f in enumerate(
                                    self.param_dict[layer]):  # self.param_dict[layer] : 16,16,3,3 / f :,16,3,3
                                o = f.view(f.shape[0], -1)  # 16,9
                                o = torch.mm(scale_mat, o)  # (11,16) * (16, 9)
                                o = o.view(scale_mat.shape[0], f.shape[1], f.shape[2])
                                original[i, :, :, :] = o
                            scaled = original

                            # # update decompose weight
                            self.decompose_weight[index] = scaled




    def main(self):

        if self.cuda == False:
            for layer in self.param_dict:
                self.param_dict[layer] = self.param_dict[layer].cpu()

        self.get_decompose_weight()

        return self.decompose_weight
