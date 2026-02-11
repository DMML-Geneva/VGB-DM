import torch
from torch import nn, Tensor
import numpy as np

import matplotlib.pyplot as plt
   
class LorenzAttractorDynamics(nn.Module):
    def __init__(self, params: Tensor=None, full: bool = True):
        super(LorenzAttractorDynamics, self).__init__()
        self.params = params
        self.full = full
        

    def forward(self, t : Tensor, x : Tensor, args=None) -> Tensor:
        X, Y, Z = x[:, [0]], x[:, [1]], x[:, [2]]
        if self.params.shape[1] == 3:
            sigma, rho, beta = self.params[:, [0]], self.params[:, [1]], self.params[:, [2]]
        else:
            sigma, rho, beta = self.params[:, [0]], torch.tensor(0.0).to(x.device), self.params[:, [1]]
        
        if self.full:
            # Full Lorenz system
            dX_dt = sigma * (Y - X)                   # Convection
            dY_dt = X * (rho - Z) - Y                 # Temperature difference
            dZ_dt = X * Y - beta * Z                  # Energy transfer & dissipation
        else:
            # Known components
            dX_dt = sigma * (Y - X)         
            dY_dt = -Y                     
            dZ_dt = X * Y - beta * Z      
            
        return torch.cat([dX_dt, dY_dt, dZ_dt], dim=1)

    def init_phys_params(self, params: Tensor):
        if params is not None:
            self.params = torch.atleast_2d(params)
    
    def sample_parameters(self,):
        """Sample parameters from narrow, physically meaningful ranges"""
        sigma = np.random.uniform(9.5, 10.5)    
        rho = np.random.uniform(27.0, 29.0)   
        beta = np.random.uniform(2.6, 2.8)     
        return sigma, rho, beta

    def sample_higher_parameters(self,):
        sigma = 10 + np.random.uniform(-2, 2)    # σ ∈ [8, 12]
        rho = 28 + np.random.uniform(-5, 5)    # ρ ∈ [23, 33]  
        beta = 8/3 + np.random.uniform(-0.5, 0.5) # β ∈ [~2.17, ~3.17]
        return sigma, rho, beta

    def sample_initial_conditions(self,):
        """Sample initial conditions with controlled variance"""
        return np.random.normal(0, 0.5, 3)
        