import torch
import torch.nn as nn
from models.u_net import Unet, Fcomb, 
from models.encoders import AxisAlignedConvGaussian
from models.utils import gini, er
from torch.distributions import  kl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# code adapted from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, d):
        super(ProbabilisticUnet, self).__init__()
        self.num_classes = d['num_classes']
        self.unet = Unet(d['input_channels'], d['num_classes'], d['num_filters'], d['initializers'], apply_last_layer=False, padding=True)
        self.prior = AxisAlignedConvGaussian(d['input_channels'], d['num_filters'], d['no_convs_per_block'], d['latent_dim'], d['initializers'], norm=d['norm'])
        self.posterior = AxisAlignedConvGaussian(d['input_channels'], d['num_filters'], d['no_convs_per_block'], d['latent_dim'], d['initializers'], posterior=True, norm=d['norm'])
        self.fcomb = Fcomb(d['num_filters'], d['latent_dim'], d['input_channels'], d['num_classes'], d['no_convs_fcomb'], {'w':'orthogonal', 'b':'normal'}, use_tile=True)
        self.beta = d['beta']
        self.latent_dim = d['latent_dim']
        if self.num_classes > 1:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)


    def forward(self, patch, segm, training=True):
        self.batch_size = patch.shape[0]
        if training:
            _, self.posterior_latent_space = self.posterior.forward(patch, segm)
            self.z_q = self.posterior_latent_space.rsample()
        _, self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)



    def sample(self, testing=False, mean=False, freeze=0, manual=None):
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
        
        elif manual is not None:
            return self.fcomb.forward(self.unet_features, manual)
        else:
            if freeze > 0:
                z_prior = self.prior_latent_space.sample()
                mean = self.prior_latent_space.mean
                var = self.prior_latent_space.variance
                a = torch.topk(var, freeze, largest=False, dim=-1).indices
                
                z_prior[:, a] = mean[:, a]
            else:  
                if mean:
                    z_prior = self.prior_latent_space.base_dist.mean
                else:
                    z_prior = self.prior_latent_space.sample()
        
        self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features,self.z_prior_sample)


    def reconstruct(self, z_posterior):
        return self.fcomb.forward(self.unet_features, z_posterior)


    def kl_divergence(self):
        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space).sum()     
        
        return kl_div

    def elbo(self, segm):
        batch_size = segm.shape[0]
        self.div = self.kl_divergence()
        self.reconstruction = self.reconstruct(z_posterior=self.z_q) 
        if self.num_classes > 1:
            segm = segm.squeeze()
        reconstruction_loss = self.criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        return -(self.reconstruction_loss + self.beta * self.div) / batch_size

