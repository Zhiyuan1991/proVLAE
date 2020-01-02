#Adapt code from: https://github.com/rtqichen/beta-tcvae
#Made modification for computing MIG-sup and for 3dshapes
import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import lib.utils as utils

metric_name = 'MIG'

def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])

def compute_metric_shapes(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0,1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2[active_units,:])
    return metric,metric_sup #first one is MIG, second one is MIG-sup

def compute_metric_3dshapes(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [10, 10, 10, 8, 4, 15]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2[active_units,:])
    return metric, metric_sup #first one is MIG, second one is MIG-sup

def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    #pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(50, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        #pbar.update(batch_size)
    #pbar.close()

    entropies /= S

    return entropies

def mutual_info_metric_shapes(sess, vae, manager):
    n_samples=manager.sample_size
    N = manager.sample_size
    K = vae.z_dim*vae.layer_num                    # number of latent variables
    nparams = vae.q_dist.nparams

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    indices = list(range(n_samples))
    batch_size=128
    total_batch = n_samples // batch_size

    # Loop over all batches
    for i in range(total_batch):
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]
        xs = manager.get_images(batch_indices)

        z_mean, z_logvar= vae.transform(sess, xs)
        z_logsigma=z_logvar*0.5
        qz_params[n:n + batch_size,:,0]=torch.from_numpy(z_mean)
        qz_params[n:n + batch_size,:,1]=torch.from_numpy(z_logsigma)
        n += batch_size

    qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    qz_means = qz_params[:, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > 1e-2].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist)

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies,active_units)
    print("metric:",metric,marginal_entropies,cond_entropies)
    return metric, marginal_entropies, cond_entropies

def mutual_info_metric_3dshapes(sess, vae, manager, zlayer=0):
    n_samples=manager.sample_size
    N = manager.sample_size
    nparams = vae.q_dist.nparams
    if zlayer==0:
        K = vae.z_dim  * vae.layer_num  # number of latent variables
    else:
        K = vae.z_dim

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    indices = list(range(n_samples))
    batch_size=100
    total_batch = n_samples // batch_size

    # Loop over all batches
    for i in range(total_batch):
        batch_indices = indices[batch_size * i: batch_size * (i + 1)]
        xs = manager.get_images(batch_indices)

        z_mean, z_logvar= vae.transform(sess, xs, zlayer=zlayer)
        z_logsigma=z_logvar*0.5
        qz_params[n:n + batch_size,:,0]=torch.from_numpy(z_mean)
        qz_params[n:n + batch_size,:,1]=torch.from_numpy(z_logsigma)
        n += batch_size


    qz_params = Variable(qz_params.view(10, 10, 10, 8, 4, 15, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    qz_means = qz_params[:, :, :, :, :, :, :, 0]
    var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
    active_units = torch.arange(0, K)[var > 1e-2].long()
    print('Active units: ' + ','.join(map(str, active_units.tolist())))
    n_active = len(active_units)
    print('Number of active units: {}/{}'.format(n_active, vae.z_dim))

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(6, K)

    print('Estimating conditional entropies for floor_hue.')
    for i in range(10):
        qz_samples_scale = qz_samples[i, :, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 10

    print('Estimating conditional entropies for wall_hue.')
    for i in range(10):
        qz_samples_scale = qz_samples[:, i, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 10

    print('Estimating conditional entropies for object_hue.')
    for i in range(10):
        qz_samples_scale = qz_samples[:, :, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 10

    print('Estimating conditional entropies for scale')
    for i in range(8):
        qz_samples_scale = qz_samples[:, :, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 8, K).transpose(0, 1),
            qz_params_scale.view(N // 8, K, nparams),
            vae.q_dist)

        cond_entropies[3] += cond_entropies_i.cpu() / 8

    print('Estimating conditional entropies for shape')
    for i in range(4):
        qz_samples_scale = qz_samples[:, :, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 4, K).transpose(0, 1),
            qz_params_scale.view(N // 4, K, nparams),
            vae.q_dist)

        cond_entropies[4] += cond_entropies_i.cpu() / 4

    print('Estimating conditional entropies for orientation')
    for i in range(15):
        qz_samples_scale = qz_samples[:, :, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 15, K).transpose(0, 1),
            qz_params_scale.view(N // 15, K, nparams),
            vae.q_dist)

        cond_entropies[5] += cond_entropies_i.cpu() / 15

    metric = compute_metric_3dshapes(marginal_entropies, cond_entropies, active_units)
    print("metric:",metric)
    return metric, marginal_entropies, cond_entropies
