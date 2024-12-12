from __future__ import print_function
import torch
import torch.nn as nn


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2
    
        
    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        zk = zk.unsqueeze(-1)
        bs = u.shape[0]
        total = zk.shape[0]
        latent_dim = zk.shape[1]
        sample_size = total // bs
        if total != bs:
            u = u.unsqueeze(1).repeat(1, sample_size, 1, 1)
            w = w.unsqueeze(1).repeat(1, sample_size, 1, 1)
            b = b.unsqueeze(1).repeat(1, sample_size, 1, 1)
            u = u.reshape(bs*sample_size, latent_dim, 1)
            w = w.reshape(bs*sample_size, 1, latent_dim)
            b = b.reshape(bs*sample_size, 1, 1)
        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)

        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
            
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(-1)
            
        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return log_det_jacobian, z
        