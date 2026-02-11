import torch
from torch import nn

from torch.nn import functional
from torchdiffeq import odeint

import time

HASHED_KEYS_NAME = "hashed_keys"

HASHED_KEYS = [
    "a_range",
    "b_range",
    "k",
    "t_span",
    "grid_size",
    "mesh_step",
    "dt",
    "n_samples",
    "n_stoch_samples",
    "sigma",
    "seed",
    "ode_solver",
    "ode_stepsize",
]


class ReactDiffDynamics(nn.Module):
    def __init__(
        self,
        params: torch.Tensor = None,
        dx: float = 0.06451612903,
        device="cuda",
    ):
        super(ReactDiffDynamics, self).__init__()
        self.params = params
        self.dx = dx  # mesh size
        self.device = device

        self.register_buffer(
            "_laplacian",
            torch.tensor(
                [
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0],
                ],
                device=device,
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            / (self.dx * self.dx),
        )

    def init_phys_params(self, params):
        self.params = params

    def laplacian(self, field):
        # field: (batch_size, grid_size, grid_size)
        _field = field.clone()

        # expand the field following Neumann boundary condition
        pad_top = _field[:, 0, :].unsqueeze(1)
        pad_bottom = _field[:, -1, :].unsqueeze(1)
        _field = torch.cat([pad_top, _field, pad_bottom], dim=1)
        pad_left = _field[:, :, 0].unsqueeze(2)
        pad_right = _field[:, :, -1].unsqueeze(2)
        _field = torch.cat([pad_left, _field, pad_right], dim=2)

        # compute Laplacian by five-point stencil
        top = _field[:, :-2, 1:-1]
        left = _field[:, 1:-1, :-2]
        bottom = _field[:, 2:, 1:-1]
        right = _field[:, 1:-1, 2:]
        center = _field[:, 1:-1, 1:-1]
        return (top + left + bottom + right - 4.0 * center) / self.dx**2

    def forward(self, t, x, args=None):
        """
        Step function of the advection-diffusion equation, used in the ODE solver.
        ----------
        Args:
            t: Time points.
            x: state.

        Returns:
            res: Solution of the advection-diffusion equation.
        """
        # state: (batch_size, 2, grid_size, grid_size)
        # params: (batch_size, 2)

        # params
        k = (
            None
            if self.params.shape[1] < 3 or (self.params[:, [2]] == 0.0).all()
            else self.params[:, [2]].unsqueeze(2).unsqueeze(3)
        )

        a, b = self.params[:, [0]].unsqueeze(2).unsqueeze(3), self.params[
            :, [1]
        ].unsqueeze(2).unsqueeze(3)

        # initial conditions
        U, V = x[:, [0]], x[:, [1]]

        U_ = functional.pad(U, pad=(1, 1, 1, 1), mode="circular")
        Delta_u = functional.conv2d(U_, self._laplacian)

        V_ = functional.pad(V, pad=(1, 1, 1, 1), mode="circular")
        Delta_v = functional.conv2d(V_, self._laplacian)

        dUdt = a * Delta_u
        if k is not None:
            dUdt += U - U**3 - k - V

        dVdt = b * Delta_v + U - V
        return torch.cat([dUdt, dVdt], dim=1)


class ReactDiffSolver(nn.Module):

    def __init__(
        self,
        t0: float,
        t1: float,
        dx: float,
        dt: float,
        ode_stepsize=1e-3,
        noise_loc=None,
        noise_std=None,
        store_time_eval=True,
        method="euler",
        device="cuda",
    ):
        super(ReactDiffSolver, self).__init__()
        self.t0 = t0
        self.t1 = t1
        self.dx = dx
        self.dt = dt
        self.ode_stepsize = ode_stepsize

        # Time and space grids
        self.t = torch.arange(
            self.t0,
            self.t1 + self.dt,  # include t1
            self.dt,
            device=device,
        )
        self.len_episode = len(self.t)
        self.device = device

        # noise
        self.noise_loc = noise_loc
        self.noise_std = noise_std
        self.store_time_eval = store_time_eval
        self.method = method

    def forward(
        self,
        init_conds: torch.Tensor,
        params: torch.tensor,
    ):
        """
        Forward model of advection-diffusion equation.
        ----------
        Args:

            params: Parameters of the model. Diffusion coefficient (dcoeff) and convection coefficient (ccoeff).
            y: Initial condition.
        Returns:
            t: Time points.
            x: Noisy solution.
                Shape: (n_samples, 2, grid_size, grid_size, len_episode)
            ode_sol: Clean solution.
                Shape: (n_samples, 2, grid_size, grid_size, len_episode)
            time: Time taken to solve the ODE.
        """
        params = torch.atleast_2d(params)
        init_conds = torch.atleast_2d(init_conds)

        start_t = time.perf_counter() if self.store_time_eval else 0.0
        phys_step_function = ReactDiffDynamics(
            params,
            self.dx,
            device=self.device,
        )

        # ODE solver
        ode_sol = odeint(
            phys_step_function,
            init_conds,
            self.t,
            method=self.method,
            options={"step_size": self.ode_stepsize},
        )

        x = ode_sol.permute(
            1,
            0,
            2,
            3,
            4,
        )  # n_samples, len_episode, 2, grid_size, grid_size
        end_t = time.perf_counter() if self.store_time_eval else 0.0

        # observation noise
        if self.noise_loc is None and self.noise_std is not None:
            self.noise_loc = 0.0
        if (
            self.noise_loc is not None
            and self.noise_std is not None
            and self.noise_std > 0
        ):
            x = x + torch.randn_like(x) * self.noise_std + self.noise_loc

        return {
            "x": x,  # shape: (n_samples, len_episode, 2, grid_size, grid_size)
            "sol": ode_sol,
            "init_conds": init_conds,
            "params": params,
            "t": self.t,
            "time_eval": (
                (end_t - start_t) + 1e-9 if self.store_time_eval else 0.0
            ),
        }
