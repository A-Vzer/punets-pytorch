import torch.nn as nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch
import torch.optim as optim
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = (epoch * 1.0 / self.warmup)
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        return lr_factor

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.truncated_normal_(m.bias, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def calc_confusion(labels, samples, class_ixs=[1], loss_mask=None):
    """
    Compute confusion matrix for each class across the given arrays.
    Assumes classes are given in integer-valued encoding.
    :param labels: 4/5D array
    :param samples: 4/5D array
    :param class_ixs: integer or list of integers specifying the classes to evaluate
    :param loss_mask: 4/5D array
    :return: 2D array
    """
    try:
        assert labels.shape == samples.shape
    except:
        raise AssertionError('shape mismatch {} vs. {}'.format(labels.shape, samples.shape))

    if isinstance(class_ixs, int):
        num_classes = class_ixs
        class_ixs = range(class_ixs)
    elif isinstance(class_ixs, list):
        num_classes = len(class_ixs)
    else:
        raise TypeError('arg class_ixs needs to be int or list, not {}.'.format(type(class_ixs)))

    if loss_mask is None:
        shp = labels.shape
        # print(shp)
        loss_mask = np.zeros(shape=(1, 1, shp[0], shp[1]))

        
    conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)

    for i,c in enumerate(class_ixs):
        pred_ = (samples == c).astype(np.uint8)
        labels_ = (labels == c).astype(np.uint8)

        conf_matrix[i,0] = int(((pred_ != 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # TP
        conf_matrix[i,1] = int(((pred_ != 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # FP
        conf_matrix[i,2] = int(((pred_ == 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # TN
        conf_matrix[i,3] = int(((pred_ == 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # FN
        
    return conf_matrix


def calc_iou(conf_matrix, replace_nan=None):
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    tns = conf_matrix[:,2]
    fns = conf_matrix[:,3]
    if tps + fps + fns != 0:  
        iou = tps / (tps + fps + fns)
    else:
        iou = replace_nan
    return iou

def calc_dsc(conf_matrix, replace_nan=None):
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    tns = conf_matrix[:,2]
    fns = conf_matrix[:,3]
    if tps + fps + fns != 0:  
        iou = 2 * tps / (2*tps + fps+ fns)
    else:
        iou = replace_nan
    return iou

def calc_mcc(conf_matrix, replace_nan=None):
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    tns = conf_matrix[:,2]
    fns = conf_matrix[:,3]
    if tps + fps or tps + fns!= 0:  
        iou = 1 + ((tps * tns - fps * fns) / np.sqrt((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)))
    else:
        iou = replace_nan
    return iou

def metrics_from_conf_matrix(conf_matrix):
    """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    tns = conf_matrix[:,2]
    fns = conf_matrix[:,3]
    metrics = {}
    metrics['iou'] = np.zeros_like(tps, dtype=np.float32)
    metrics['dsc'] = np.zeros_like(tps, dtype=np.float32)
    metrics['mcc'] = np.zeros_like(tps, dtype=np.float32)

    # iterate classes
    for c in range(tps.shape[0]):
        # unless both the prediction and the ground-truth is empty, calculate a IoU otherwise give full score.
        if tps[c] + fps[c] + fns[c] != 0:
            
            metrics['dsc'][c] = 2*tps[c] / (2*tps[c] + fps[c] + fns[c])
            metrics['mcc'][c] = 1 + ((tps[c] * tns[c] - fps[c] * fns[c]) / np.sqrt((tps[c] + fps[c]) * (tps[c] + fns[c]) * (tns[c] + fps[c]) * (tns[c] + fns[c])))
        else:
            metrics['iou'][c] = 1
            metrics['dsc'][c] = 1
#        print(tps[c], fps[c], fns[c], tns[c])
#        print(metrics['mcc'][c])
        if tps[c] + fps[c] == 0 and tps[c] + fns[c] != 0:
            metrics['mcc'][c] = 0
        elif tps[c] + fps[c] == 0 and tps[c] + fns[c] == 0:
            metrics['mcc'][c] = 2
        elif tps[c] + fps[c] != 0 and tps[c] + fns[c] == 0:
            metrics['mcc'][c] = 0
        elif fns[c] + tns[c] == 0 and tns[c] + fps[c] != 0:
            metrics['mcc'][c] = 0

    return metrics

def hd_distance(x, y):
    indexes = np.nonzero(x)
    d = edt(np.logical_not(y))
    if len(indexes[0]) != 0:
        h = np.array(np.max(d[indexes]))
        return h
    else:
        return 0

def calc_hausdorf_dist(x, y):

    right_hd = hd_distance(x, y)
    left_hd = hd_distance(y, x)
    # print(left_hd, right_hd)
    return np.maximum(right_hd, left_hd)

def gini(x, N):
    sorted, indices = torch.sort(x, -1)
    w = torch.tensor([(N - (k + 1) + 1/2 ) / N for k in range(N)], device=x.device).T
    norm = torch.sum(torch.abs(x), dim=-1)
    x = 1 - 2 * (torch.matmul(sorted, w) / norm)
    return x

def er(x):
    x = x / torch.sum(x, dim=-1, keepdims=True)
    return torch.exp(torch.sum(x * -torch.log(x), dim=-1))

def gini_np(x):
    N = x.shape[-1]
    sorted = np.sort(x, -1)
    w = np.array([(N - (k + 1) + 1/2 ) / N for k in range(N)]).T
    norm = np.sum(np.abs(x), axis=-1)
    x = 1 - 2 * (np.matmul(sorted, w) / norm)
    return x

def effective_rank(x):
    x = x / np.sum(x, axis=-1, keepdims=True)
    return np.exp(np.sum(x * -np.log(x), axis=-1))