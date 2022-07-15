import numpy as np
import torch
import torch.nn as nn


def Entropy_loss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


# sliced wasserstein computation use
def get_theta(embedding_dim, num_samples=50):
    theta = [w / np.sqrt((w ** 2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()


def sliced_wasserstein_distance(source_z, target_z, embed_dim, num_projections=256, p=1):
    # \theta is vector represents the projection directoin
    theta = get_theta(embed_dim, num_projections)
    proj_target = target_z.matmul(theta.transpose(0, 1))
    proj_source = source_z.matmul(theta.transpose(0, 1))
    w_distance = torch.sort(proj_target.transpose(0, 1), dim=1)[
                     0] - torch.sort(proj_source.transpose(0, 1), dim=1)[0]

    w_distance_p = torch.pow(w_distance, p)

    return w_distance_p.mean()


def sliced_WD_loss_function(source_feature, target_feature, dir_repeats=1, dirs_per_repeat=256):
    assert source_feature.dim() == 2 and source_feature.shape == target_feature.shape  # (neighborhood, descriptor_component)
    result = 0
    for repeat in range(dir_repeats):
        theta = np.random.randn(source_feature.shape[1], dirs_per_repeat)  # (descriptor_component, direction)
        theta /= np.sqrt(
            np.sum(np.square(theta), axis=0, keepdims=True))  # normalize descriptor components for each direction
        theta = theta.astype(np.float32)
        theta = torch.from_numpy(theta)
        if torch.cuda.is_available():
            theta = theta.cuda()
        proj_source = source_feature.matmul(theta)
        proj_target = target_feature.matmul(theta)
        proj_source, _ = torch.sort(proj_source, dim=0)  # sort neighborhood projections for each direction
        proj_target, _ = torch.sort(proj_target, dim=0)
        dists = torch.abs(proj_source - proj_target)  # pointwise wasserstein distances
        result += dists.mean()
    return result / dir_repeats