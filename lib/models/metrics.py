from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)


class CosMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class SphereMarginProduct(nn.Module):
    r"""Implement of SphereFace loss :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin [1.5, 4.0]
        cos(m * theta)
    """

    def __init__(self, in_features, out_features, s=64, m=1.5):
        super(SphereMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # --------------------------- convert label to one-hot ---------------------------
        theta = self.m * torch.acos(cosine)
        cosine_new = torch.cos(theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * cosine_new) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class criterion_mask(nn.Module):
    """
    in order to make two masks be different
    """
    def __init__(self, IoU_thres):
        super(criterion_mask, self).__init__()
        self.IoU_thres = IoU_thres
        self.criterion_diff = torch.nn.L1Loss(reduction='none')

    def forward(self, mask1, mask2, IoU):
        IoU = IoU.cuda()
        loss_zeros = torch.zeros(mask1.size(0)).double().cuda()
        # loss_small = torch.sum((1-mask1) * (1-mask2), [1, 2, 3], dtype=float) / mask1[0].nelement()
        # loss_small = torch.sum((1-mask1) * (1-mask2), [1, 2, 3], dtype=float) / (torch.sum((1-mask1)+(1-mask2), dtype=float) + 1e-5)
        A = torch.sum((1-mask1)*(1-mask2), [1, 2, 3], dtype=float)
        B = torch.sum((1-mask1)+(1-mask2), [1, 2, 3], dtype=float) + 1e-5
        loss_small = A / torch.clamp(B, min=0.0, max=1.0)
        loss_small = loss_small.double().cuda()

        # loss_small = self.criterion_diff(mask1, (1-mask2))
        # loss_small = torch.sum(loss_small, [1, 2, 3]).double() / loss_small[0].nelement()

        loss_big = self.criterion_diff(mask1, mask2)
        loss_big = torch.sum(loss_big, [1, 2, 3]).double() / loss_big[0].nelement()

        loss_small = torch.where(IoU < self.IoU_thres[0], loss_small, loss_zeros)
        loss_big = torch.where(IoU > self.IoU_thres[1], loss_big, loss_zeros)

        loss = loss_small + loss_big

        return torch.mean(loss)

