#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn

from torchdyn.core import NeuralODE
from torchdyn import *
import math


import torch
import torch.nn as nn


class PhysicalModel(nn.Module):
    def __init__(self, param_values, trainable_param):
        super(PhysicalModel, self).__init__()
        self._nb_parameters = len(trainable_param)
        self._X_dim = -1
        self.incomplete_param_dim_textual = []
        self.full_param_dim_textual = []
        self.missing_param_dim_textual = []
        self.trainable_param = trainable_param
        for p in param_values.keys():
            if p in trainable_param:
                self.incomplete_param_dim_textual.append(p)
            else:
                self.missing_param_dim_textual.append(p)
            self.full_param_dim_textual.append(p)

    def _nb_parameters(self):
        return self._nb_parameters

    def _X_dim(self):
        return self._nb_parameters

    def forward(self, t, x, args=None):
        pass

    def parameterized_forward(self, t, x, **parameters):
        if len(set(parameters.keys()) - set(self.trainable_param)) != 0:
            raise Exception(
                "Parameterized forward physical arguments does not match the simulator specification. "
                "Simulator: {} - kwargs: {}".format(
                    self.trainable_param, parameters.keys()
                )
            )
        pass

    def get_x_labels(self):
        return ["$x_%d$" for i in range(self._X_dim)]

    def get_name(self):
        return "Generic Simulator"


class RLCODEDynamic(nn.Module):
    """
    RLCODEDynamic is a class that implements the RLC circuit model as a physical dynamic model for simulation using torchdyn.
    """

    def __init__(self, params=None, device=None):
        super(RLCODEDynamic, self).__init__()
        self.device = device

        if params is not None:
            self.init_phys_params(params)

    def init_phys_params(self, params):
        """
        Initialize the physical parameters of the RLC circuit.
        :param params: A tensor containing the physical parameters of the RLC circuit.
        """
        self.L = params[:, 0:1]  # Inductance
        self.C = params[:, 1:2]  # Capacitance
        self.R = (
            params[:, 2:3]
            if params.shape[1] > 2
            else torch.zeros_like(self.L, device=self.device)
        )  # Resistance
        self.V_a = (
            params[:, 3:4]
            if params.shape[1] > 3
            else torch.ones_like(self.L, device=self.device) * 2.5
        )  # Voltage across the capacitor
        self.V_c = (
            params[:, 4:5]
            if params.shape[1] > 4
            else torch.ones_like(self.L, device=self.device) * 1.0
        )  # Amplitude of the AC voltage
        self.omega = (
            params[:, 5:6]
            if params.shape[1] > 5
            else torch.ones_like(self.L, device=self.device) * 2.0
        )

    def forward(self, t, x, args=None):
        return torch.cat(
            (
                x[:, [1]] / self.C,
                (self.V(t) - x[:, [0]] - self.R * x[:, [1]] / self.C) / self.L,
            ),
            1,
        )

    def V(self, t, V_c=None, V_a=None, omega=None):
        if V_a is None:
            return self.V_c + self.V_a * torch.sin(t * self.omega)
        return V_c + V_a * torch.sin(t * omega)


class RLCODE(PhysicalModel):
    """
    RLCODE is a class that implements the RLC circuit model as a physical model for simulation, thus the DynamicODE
    can be used to simulate the circuit dynamics.
    """

    def __init__(self, param_values, trainable_param):
        super(RLCODE, self).__init__(param_values, trainable_param)
        self._X_dim = 2
        self.R = (
            nn.Parameter(torch.tensor(param_values["R"]))
            if "R" in trainable_param
            else param_values["R"]
        )
        self.L = (
            nn.Parameter(torch.tensor(param_values["L"]))
            if "L" in trainable_param
            else param_values["L"]
        )
        self.C = (
            nn.Parameter(torch.tensor(param_values["C"]))
            if "C" in trainable_param
            else param_values["C"]
        )
        self.V_c = (
            nn.Parameter(torch.tensor(param_values["V_c"]))
            if "V_c" in trainable_param
            else param_values["V_c"]
        )
        self.V_a = (
            nn.Parameter(torch.tensor(param_values["V_a"]))
            if "V_a" in trainable_param
            else param_values["V_a"]
        )
        self.omega = (
            nn.Parameter(torch.tensor(param_values["omega"]))
            if "omega" in trainable_param
            else param_values["omega"]
        )

    def forward(self, t, x, args=None):
        return torch.cat(
            (
                x[:, [1]] / self.C,
                (self.V(t) - x[:, [0]] - self.R * x[:, [1]] / self.C) / self.L,
            ),
            1,
        )

    def V(self, t, V_c=None, V_a=None, omega=None):
        if V_a is None:
            return self.V_c + self.V_a * torch.sin(t * self.omega)
        return V_c + V_a * torch.sin(t * omega)

    def parameterized_forward(self, t, x, **parameters):
        super(RLCODE, self).parameterized_forward(t, x, **parameters)
        C = self.C if "C" not in parameters else parameters["C"]
        R = self.R if "R" not in parameters else parameters["R"]
        L = self.L if "L" not in parameters else parameters["L"]
        V_a = self.V_a if "V_a" not in parameters else parameters["V_a"]
        V_c = self.V_c if "V_c" not in parameters else parameters["V_c"]
        omega = (
            self.omega if "omega" not in parameters else parameters["omega"]
        )
        return torch.cat(
            (
                x[:, [1]] / C,
                (self.V(t, V_c, V_a, omega) - x[:, [0]] - R * x[:, [1]] / C)
                / L,
            ),
            1,
        )

    def get_x_labels(self):
        return ["$U_C$", "$I_C$"]

    def get_name(self):
        return "RLC Circuit" + str(self.trainable_param)


class RLCCircuit:
    """
    RLCCircuit is a class that implements the RLC circuit model for generating datasets by sampling parameters and
    simulating the circuit dynamics defined in the RLCODE class using the NeuralODE.
    """

    def __init__(
        self,
        init_param=None,
        true_param=None,
        T0=0.0,
        T1=5,
        n_timesteps=40,
        partial_model_param=None,
        name="RLCCircuit",
        **kwargs,
    ):
        if partial_model_param is None:
            partial_model_param = ["R", "L", "C"]
        self.full_param_dim_textual = ["R", "L", "C", "V_a", "V_c", "omega"]
        self.incomplete_param_dim_textual = partial_model_param
        self.init_param = (
            {"R": 5, "L": 5.0, "C": 1.0, "V_a": 2.5, "V_c": 1.0, "omega": 2.0}
            if init_param is None
            else init_param
        )
        self.T0 = float(T0)
        self.T1 = float(T1)
        self.n_timesteps = int(n_timesteps)
        self.name = "RLC"
        self.true_param = (
            {"R": 5, "L": 5.0, "C": 1.0, "V_a": 2.5, "V_c": 1.0, "omega": 2.0}
            if true_param is None
            else true_param
        )

    def sample_init_state(self, n=1):
        return torch.cat((torch.randn(n, 1), torch.zeros([n, 1])), 1)

    def sample_sequences(self, parameters=None, n=1, x0=None):
        if parameters is None:
            parameters = self.true_param
        x0 = self.sample_init_state(n) if x0 is None else x0
        t_span = torch.linspace(self.T0, self.T1, self.n_timesteps)
        f = self.get_full_physical_model(parameters)
        model = NeuralODE(f, sensitivity="adjoint", solver="dopri5")
        with torch.no_grad():
            t_eval, y_hat = model(x0, t_span)
            return t_eval, y_hat  # + torch.randn_like(y_hat) * .01

    def get_incomplete_physical_model(
        self, parameters, trainable=True
    ) -> nn.Module:
        if trainable:
            for p in self.full_param_dim_textual:
                if (
                    p not in self.incomplete_param_dim_textual
                    and p not in parameters.keys()
                ):
                    parameters[p] = 0.0
            return RLCODE(
                param_values=parameters,
                trainable_param=self.incomplete_param_dim_textual,
            )
        else:
            return RLCODE(param_values=parameters, trainable_param=[])

    def get_full_physical_model(
        self, parameters, trainable=False
    ) -> nn.Module:
        if trainable:
            return RLCODE(
                param_values=parameters,
                trainable_param=self.full_param_dim_textual,
            )
        else:
            return RLCODE(param_values=parameters, trainable_param=[])
