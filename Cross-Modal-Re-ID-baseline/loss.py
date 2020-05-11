import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

from IPython import embed


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CrossTriplet(nn.Module):
    def __init__(self, batch_size, margin=0):
        super(CrossTriplet, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

        n = batch_size * 2
        self.sub = [0 for i in range(batch_size)] + [1 for i in range(batch_size)]
        self.sub = torch.Tensor(self.sub).cuda()
        self.sub = self.sub.expand(n, n).eq(self.sub.expand(n, n).t())
        self.sub = self.sub - 1

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # sub = sub.expand(n, n).eq(sub.expand(n, n).t())
        # sub = 1 - sub

        mask1 = mask * self.sub
        mask2 = (1 - mask) * self.sub
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask1[i]].max())
            dist_an.append(dist[i][mask2[i]].min())

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # print(dist_an,dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        correct = torch.ge(dist_an, dist_ap).sum().item()

        return loss, correct

class Rank_loss(nn.Module):

    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, batch_size, margin_1=1.0, margin_2=1.5, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

        n = batch_size * 2
        self.sub = [0 for i in range(batch_size)] + [1 for i in range(batch_size)]
        self.sub = torch.Tensor(self.sub).cuda()

    def forward(self, x, targets, norm=False):
        if norm:
            x = torch.nn.functional.normalize(x, dim=1, p=2)

        dist_mat = self.euclidean_dist(x, x) # compute the distance

        loss = self.rank_loss(dist_mat, targets, self.sub)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, dist_an = [], []
        for i in range(N):
            dist_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        correct = torch.ge(dist_an, dist_ap).sum().item()

        return loss, correct

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])


            intra_modality = sub.eq(sub[i])
            cross_modality = 1- intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality


            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)

            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5
            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.mul(self.alpha_1-an_less_intra,an_weight_intra))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))

            loss_an = torch.div(an_sum_intra,an_weight_intra_sum ) + torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )
            loss += loss_ap + loss_an
            #loss += loss_an


        return loss * 1.0/ dist.size(0)

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

class TransitionLoss(nn.Module):
    def __init__(self, batch_size, margin_1=0.3, alpha_1=0.5):
        super(TransitionLoss, self).__init__()
        self.margin_1 = margin_1
        self.alpha_1 = alpha_1

        n = batch_size * 2
        self.sub = [0 for i in range(batch_size)] + [1 for i in range(batch_size)]
        self.sub = torch.Tensor(self.sub).cuda()


    def forward(self, feature, targets):

        loss = 0
        for i in range(feature.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 1
            is_neg = targets.ne(targets[i])

            intra_modality = self.sub.eq(self.sub[i])
            cross_modality = 1- intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            trans_pos, trans_neg = self.trans_dist(feature, mask_pos_intra, mask_pos_cross,
                mask_neg_intra, mask_neg_cross)


            ap_pos_intra = torch.clamp(torch.add(trans_neg, self.margin_1-self.alpha_1),0)
            # print(ap_pos_intra)
            loss_ap = ap_pos_intra.mean().mean()
            an_neg_intra = torch.clamp(self.alpha_1-trans_pos,0)
            loss_an = an_neg_intra.mean().mean()
            loss += loss_ap + loss_an

        loss = loss/feature.size(0)

        dist_mat = self.euclidean_dist(feature, feature)  # compute the distance

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        dist_ap, dist_an = [], []
        for i in range(N):
            dist_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        correct = torch.ge(dist_an, dist_ap).sum().item()

        return loss, correct

    def trans_dist(self, feature, mask_pos_intra, mask_pos_cross, mask_neg_intra, mask_neg_cross):

        match_pos_cross = torch.mm(feature[mask_pos_intra],feature[mask_pos_cross].t())
        match_neg_cross = torch.mm(feature[mask_neg_intra],feature[mask_neg_cross].t())
        ab_pos_prob = F.softmax(match_pos_cross, dim=-1)
        ba_pos_prob = F.softmax(match_pos_cross.t(), dim=-1)
        ab_neg_prob = F.softmax(match_neg_cross, dim=-1)
        ba_neg_prob = F.softmax(match_neg_cross.t(), dim=-1)
        trans_pos = torch.mm(ab_pos_prob, ba_pos_prob)
        trans_neg = torch.mm(ab_neg_prob, ba_neg_prob)
        # trans_pos = torch.mm(match_pos_cross, match_pos_cross.t())
        # trans_neg = torch.mm(match_neg_cross, match_neg_cross.t())

        return torch.clamp(trans_pos,1e-12).sqrt(), torch.clamp(trans_neg,1e-12).sqrt()

    def visit_los(self, p, weight = 1.0):
        pass

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

