import torch.nn as nn
from models.punet import ProbabilisticUnet
from models.utils import gini
from models.flowDensities import PlanarFlowDensity

            
class PUNet_NF(ProbabilisticUnet):
    def __init__(self, d):
        super().__init__(d=d)

        self.posterior = PlanarFlowDensity(d['num_flows'], d['input_channels'], d['num_filters'], d['no_convs_per_block'],
                        d['latent_dim'], d['initializers'], flow=d['flow'], posterior=True, norm=d['norm'])  
    
    def forward(self, patch, segm, training=True, **kwargs):
        if training:
            self.log_det_j_q, self.z0_q, self.z_q, self.posterior_latent_space = self.posterior.forward(patch, segm)
        _, self.prior_latent_space = self.prior.forward(patch, None)

        self.batch_size = patch.shape[0]
        self.unet_features = self.unet.forward(patch, False)

    def kl_divergence(self, **kwargs):
        q = self.posterior_latent_space.log_prob(self.z0_q)
        p = self.prior_latent_space.log_prob(self.z_q)
        kl_div = (q - p - self.log_det_j_q).sum()

        return kl_div