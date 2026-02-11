"""Generate data of forced damped pendulum."""

from networkx import omega
import torch
from torch import nn
from torchdyn.core import NeuralODE
import time


class PendulumSolver(nn.Module):
    def __init__(
        self,
        len_episode,
        dt=0.01,
        noise_loc=None,
        noise_std=None,
        with_grad=False,
        store_time_eval=True,
        method="rk4",
    ):
        super().__init__()
        self.model_type = "pendulum"
        self.len_episode = len_episode
        self.noise_loc = noise_loc
        self.noise_std = noise_std
        self.with_grad = with_grad
        self.store_time_eval = store_time_eval
        self.method = method
        if isinstance(self.method, list):
            self.method = self.method[0]
        self.dt = dt
        self.dt_friction = []
        self.dt_states = []
        self.dyn_model = PendulumDynamics(device=None)

    def forward(self, init_conds, params: torch.Tensor, t_grad=False):
        """
        Forward computation of the forced damped pendulum.
        ----------
        Args:
            params: torch.tensor
                Parameters of the model. The first element is the natural frequency omega, the second element is the damping gamma, the third element is the amplitude A, and the fourth element is the phase phi.
            init_conds: torch.tensor
                Initial condition of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
        Returns:
            x: torch.tensor
                The observation of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
            sol: torch.tensor
                The solution of the pendulum. The first element is the angle theta, the second element is the angular velocity theta_dot.
            t: torch.tensor
                The time steps of the solution.
            time_eval: float
                The time it took to evaluate the solution.
        """
        params = torch.atleast_2d(params)
        init_conds = torch.atleast_2d(init_conds)

        self.dyn_model.init_phys_params(params)
        self.dyn_model.device = params.device

        t = torch.linspace(
            0.0,
            self.dt * (self.len_episode - 1),
            self.len_episode,
            device=params.device,
        )

        start_t = time.perf_counter() if self.store_time_eval else 0.0
        node = NeuralODE(
            self.dyn_model,
            sensitivity="adjoint",
            solver=self.method,
            atol=1e-10,
            rtol=1e-10,
        )
        if not self.with_grad:
            with torch.no_grad():
                t_eval, x = node(init_conds, t)
        else:
            t_eval, x = node(init_conds, t)
        end_t = time.perf_counter() if self.store_time_eval else 0.0

        # permute x
        x = x[:, :, 0].permute(1, 0)

        # observation noise
        if self.noise_loc is None and self.noise_std is not None:
            self.noise_loc = 0.0
        if (
            self.noise_loc is not None
            and self.noise_std is not None
            and self.noise_std > 0.0
        ):
            x = x + torch.randn_like(x) * self.noise_std + self.noise_loc

        return {
            "x": x,
            "t": t,
            "time_eval": (
                (end_t - start_t) + 1e-9 if self.store_time_eval else 0.0
            ),
        }


class PendulumDynamics(nn.Module):
    """
    Simple pendulum model with no damping and no forcing, used for the OdeNet model.
    This class does not perform the odeint integration, it only returns the derivative of the state.
    """

    def __init__(self, params=None, device=None):
        super(PendulumDynamics, self).__init__()
        self.device = device
        if params is not None:
            self.init_phys_params(params)

    def init_phys_params(self, params):
        params = torch.atleast_2d(params)
        self.omega = params[:, 0].view(-1, 1)
        self.gamma = params[:, 1].view(-1, 1) if params.shape[1] > 1 else None
        self.A = params[:, 2].view(-1, 1) if params.shape[1] > 2 else None
        self.phi = (
            params[:, 3].view(-1, 1)
            if params.shape[1] > 3
            else torch.ones_like(self.omega)
        )

    def forward(self, t, x, args=None):
        """
        Arguments:
            t: torch.tensor
                Current time step.
            x: torch.tensor
                Current state of the pendulum.
                The first element is the angle theta, the second element is the angular velocity theta_dot.
        Returns:
            dy/dt: torch.tensor
                The derivative of the state.
            d^2y/dt^2: torch.tensor
                The second derivative of the state.
        """
        th = x[:, 0].view(-1, 1)
        thdot = (
            x[:, 1].view(-1, 1) if x.shape[1] == 2 else torch.zeros_like(th)
        )

        if self.gamma is not None and (self.gamma > 0.0).any():
            dt_friction = self.gamma * thdot
        else:
            dt_friction = torch.zeros_like(thdot)
        if self.A is not None and (self.A != 0.0).any():
            force = (
                self.A
                * self.omega
                * self.omega
                * torch.cos(2.0 * torch.pi * self.phi * t)
            )
        else:
            force = torch.zeros_like(th)
        sin_x = self.omega * self.omega * torch.sin(th)
        total_acc = force - dt_friction - sin_x

        return torch.cat(
            [
                thdot,
                total_acc,
            ],
            dim=-1,
        )
