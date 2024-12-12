import torch
import torch.nn as nn
from models.punet import ProbabilisticUnet
from models.flowDensities import PlanarFlowDensity
from torch.distributions import kl, Independent, Normal
from geomloss import SamplesLoss
from models.utils import gini, er

class SPUNet(ProbabilisticUnet):
    def __init__(self, d):
        super().__init__(d=d)
        self.gamma = d['gamma']
        self.samples = d['samples']
        self.latent_dim = d['latent_dim']
        self.s_dist = SamplesLoss(loss="sinkhorn", p=2, diameter=d['diameter'])

    def forward(self, patch, segm, training=True):
        self.batch_size = patch.shape[0]        

        if training:
            self.num_raters = segm.shape[1]
            self.posteriors = torch.zeros((self.num_raters, self.batch_size, self.latent_dim, 2)).to(patch.device)
            self.raters = torch.randint(low=0, high=self.num_raters, size=(self.batch_size,))
            self.z_qs = torch.zeros((self.num_raters, self.batch_size, self.latent_dim)).to(patch.device)
            self.w_q = torch.zeros((self.num_raters, self.batch_size)).to(patch.device)
            self.w_p = torch.zeros((self.num_raters, self.batch_size)).to(patch.device)
            for i in range(self.num_raters):
                mu, log_sigma = self.posterior.forward(patch, segm[:, [i], ...], return_params=True)
                dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
                sample = dist.rsample()
                self.z_qs[i, ...] = sample
                self.w_q[i] = torch.exp(dist.log_prob(sample))
                self.posteriors[i, ..., 0] = mu
                self.posteriors[i, ..., 1] = log_sigma
            
            mu = self.posteriors[self.raters, torch.arange(self.batch_size), :, 0]
            log_sigma = self.posteriors[self.raters, torch.arange(self.batch_size), :, 1]
            self.posterior_latent_space = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
            self.z_q = self.z_qs[self.raters, torch.arange(self.batch_size)]
        
        _, self.prior_latent_space = self.prior.forward(patch, return_params=False)
        self.unet_features = self.unet.forward(patch, False)

    def kl_divergence(self):
        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space).sum()     
        return kl_div

    def calc_sinkhorn(self):
        z_p = self.prior_latent_space.rsample((self.num_raters,)).reshape(self.num_raters, self.batch_size, self.latent_dim)        
        weight_p = torch.exp(self.prior_latent_space.log_prob(z_p))
        weight_q = torch.transpose(self.w_q, 0, 1)
        weight_p = torch.transpose(weight_p, 0, 1)
        z_q = self.z_qs.reshape(self.batch_size, self.num_raters, self.latent_dim)
        z_p = z_p.reshape(self.batch_size, self.num_raters, self.latent_dim)
        loss = self.s_dist(weight_q, z_q, weight_p, z_p).sum()
        return loss

    def elbo(self, segm):
        self.div = self.kl_divergence()
        self.sink = self.calc_sinkhorn()
        self.reconstruction = self.reconstruct(z_posterior=self.z_q)
        segm = segm[torch.arange(self.batch_size), self.raters, ...].unsqueeze(1)
        reconstruction_loss = self.criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        return -(self.reconstruction_loss + self.beta * self.div + self.gamma * self.sink) / self.batch_size

    