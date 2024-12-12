import torch.nn as nn
from models.utils import init_weights
from models.punet import AxisAlignedConvGaussian
from models.flows import Planar

class PlanarFlowDensity(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix as
    the base distribution for a sequence of flow based transformations.
    """

    def __init__(self, num_flows, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, flow,
                 posterior=False, norm=False, samples=1):
        super().__init__()

        self.num_flows = num_flows
        self.latent_dim = latent_dim
        self.samples = samples
        self.base_density = AxisAlignedConvGaussian(input_channels, num_filters, no_convs_per_block, latent_dim,
                                                    initializers, posterior=posterior, norm=norm)
        if flow =='planar':
            flow = Planar

        nF_oP = num_flows * latent_dim

        self.amor_u = nn.Sequential(nn.Linear(num_filters[-1], nF_oP),nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),nn.BatchNorm1d(nF_oP))
        self.amor_w = nn.Sequential(nn.Linear(num_filters[-1], nF_oP),nn.ReLU(),
                nn.Linear(nF_oP, nF_oP),nn.BatchNorm1d(nF_oP))
        self.amor_b = nn.Sequential(nn.Linear(num_filters[-1], num_flows), nn.ReLU(),
            nn.Linear(num_flows, num_flows),nn.BatchNorm1d(num_flows))
        
        self.amor_u.apply(init_weights)
        self.amor_w.apply(init_weights)
        self.amor_b.apply(init_weights)

        for k in range(num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)
    
    def forward(self, input, segm=None):

        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        batch_size = input.shape[0]
        
        h, z0_density = self.base_density(input, segm)
        z0 = z0_density.rsample((self.samples,)).reshape(self.samples * batch_size, self.latent_dim)
        z = [z0]
        self.u = self.amor_u(h).view(batch_size, self.num_flows, self.latent_dim, 1)
        self.w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.latent_dim)
        self.b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        log_det_j, z = self.flow(z)
        
        return log_det_j, z[0], z[-1], z0_density
    
    def flow(self, z):
    
        log_det_j = 0.

        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            log_det_jacobian, z_k = flow_k(z[k], self.u[:, k, :, :], self.w[:, k, :, :], self.b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian
        return log_det_j, z