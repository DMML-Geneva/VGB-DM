from src.grey_box_clim.model_utils import *
from src.grey_box_clim.utils import *
from torchdiffeq import odeint as odeint


class Climate_ResNet_2D(nn.Module):

    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(
                    self.make_layer(
                        self.block, num_channels, hidden_size[idx], layers[idx]
                    )
                )
            else:
                layers_cnn.append(
                    self.make_layer(
                        self.block,
                        hidden_size[idx - 1],
                        hidden_size[idx],
                        layers[idx],
                    )
                )

        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self, block, in_channels, out_channels, reps):
        layers = []
        layers.append(block(in_channels, out_channels))
        self.inplanes = out_channels
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for l, layer in enumerate(self.layer_cnn):
            dx_final = layer(dx_final)

        return dx_final


class VAEEncoder(nn.Module):
    """VAE Encoder"""

    def __init__(self, in_channels=5, latent_channels=5):
        super().__init__()
        # Encoder network that preserves spatial dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Separate heads for mean and log variance
        self.fc_mu = nn.Conv2d(32, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(32, latent_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, channels, height, width]
        Returns:
            mu: Mean of shape [batch, latent_channels, height, width]
            log_var: Log variance of shape [batch, latent_channels, height, width]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class HistoryAttentionEncoder(nn.Module):
    """Temporal attention over history states for weather forecasting"""

    def __init__(self, channels, num_heads=5):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert (
            channels % num_heads == 0
        ), f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = channels // num_heads

        # Learnable projection layers (convolutions for efficiency)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

        # Scale factor for attention
        self.scale = self.head_dim**-0.5

    def forward(self, current_state, history):
        """
        Temporal attention: current state attends to history at each spatial location
        Args:
            current_state: [B, C, H, W] - current state x_t
            history: [B, T, C, H, W] - past states (T=2-5 timesteps)
        Returns:
            attended_features: [B, C, H, W] - history-aware features
        """
        B, T, C, H, W = history.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"

        # Query from current state: [B, C, H, W] -> [B, heads, H*W, head_dim]
        q = self.q_proj(current_state)  # [B, C, H, W]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(
            2, 3
        )  # [B, heads, H*W, head_dim]

        # Keys and Values from history: [B, T, C, H, W] -> [B, heads, H*W, T, head_dim]
        history_flat = history.reshape(B * T, C, H, W)
        k = self.k_proj(history_flat).reshape(
            B, T, self.num_heads, self.head_dim, H * W
        )
        v = self.v_proj(history_flat).reshape(
            B, T, self.num_heads, self.head_dim, H * W
        )

        # Rearrange for attention computation
        k = k.permute(0, 2, 4, 1, 3)  # [B, heads, H*W, T, head_dim]
        v = v.permute(0, 2, 4, 1, 3)  # [B, heads, H*W, T, head_dim]

        # Attention: For each spatial position, attend over T history steps
        # [B, heads, H*W, 1, head_dim] @ [B, heads, H*W, head_dim, T] -> [B, heads, H*W, 1, T]
        attn = (q.unsqueeze(-2) @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # Softmax over temporal dimension

        # Apply attention to values: [B, heads, H*W, 1, T] @ [B, heads, H*W, T, head_dim] -> [B, heads, H*W, 1, head_dim]
        out = (attn @ v).squeeze(-2)  # [B, heads, H*W, head_dim]

        # Reshape back to spatial: [B, heads, H*W, head_dim] -> [B, C, H, W]
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.out_proj(out)

        return out


# Explicit time-scaling constants (make nondimensionalization explicit)
HOURS_PER_UNIT = 6.0  # physical hours represented by one unit in t (dataset specific)
TIME_SCALE = 100.0  # factor to divide hours by to make ODE time O(1)
EPS_DT = 1e-6  # small epsilon to avoid division by zero in dt
HOURS_in_DAY = 24.0


class Climate_VFM_ENC(nn.Module):

    def __init__(
        self,
        num_channels,
        history_size,
        const_channels,
        out_types,
        method,
        use_att,
        use_err,
        use_pos,
        use_batch_norm=False,
        var_noise=0.1,
        use_history_attention=False,
    ):
        super().__init__()

        self.var_noise = var_noise

        self.layers = [5, 3, 2]
        # self.layers = [1, 1]
        output_dim = 10
        # Time embedding (1) + day (2) + seasonal (2) + state (out_types)
        # + physics (2*out_types) + gradients (2*out_types) + z (out_types) = 5 + 5*out_types
        input_dim = 4 * out_types  # Base channels = (4*5 (no v)) = 20 when out_types=5
        input_channels = input_dim + out_types * int(use_pos) + 34 * (1 - int(use_pos))
        # self.hidden = [8, output_dim]
        self.hidden = [
            128,
            64,
            output_dim,
        ]  # [128, 64, output_dim] #[8, 64, output_dim]
        self.history_size = history_size
        self.use_history_attention = use_history_attention

        # If using attention, history contributes out_types channels; otherwise history_size * out_types
        if use_history_attention and history_size > 0:
            input_channels += (
                out_types  # Attention output is compressed to out_types channels
            )
        else:
            input_channels += history_size * out_types  # Concatenated history

        self.vel_f = Climate_ResNet_2D(input_channels, self.layers, self.hidden)
        # head for source term (unmodeled physics)
        self.source_net = Climate_ResNet_2D(input_channels, [2, 1], [64, out_types])

        phys_channels = 2 * out_types * (1)  # Assuming scale = 1, from fm_phys_train.py
        self.phys_conv = nn.Conv2d(
            in_channels=phys_channels,
            out_channels=phys_channels,
            kernel_size=1,
        )

        self.vae_encoder = VAEEncoder(in_channels=out_types, latent_channels=out_types)

        # History Attention module (if enabled)
        if use_history_attention and history_size > 0:
            self.history_attn = HistoryAttentionEncoder(
                channels=out_types, num_heads=out_types
            )
            # Learnable positional embeddings for each history timestep
            # Helps model distinguish between t-1, t-2, etc.
            self.history_pos_emb = nn.Parameter(
                torch.randn(1, history_size, out_types, 1, 1) * 0.02
            )

        if use_att:
            self.vel_att = Self_attn_conv(input_channels, output_dim)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels

        self.out_ch = out_types
        self.vel_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.method = method

        # Feature normalization flags and modules
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            # Use InstanceNorm (GroupNorm with num_groups=channels) instead of BatchNorm/LayerNorm
            # InstanceNorm is stable at ANY batch size (including batch_size=1 during inference)
            # Each channel of each sample is normalized independently - no batch dependency
            self.normalize_state = nn.GroupNorm(
                num_groups=out_types, num_channels=out_types, affine=True
            )
            self.normalize_gradients = nn.GroupNorm(
                num_groups=2 * out_types,
                num_channels=2 * out_types,
                affine=True,
            )

        self.att = use_att
        self.err = use_err
        self.pos = use_pos
        self.pos_feat = 0
        self.lsm = 0
        self.oro = 0

    def update_param_phys(self, params):
        # self.past_samples = params[0]  # vel samples
        self.vel_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE
        Args:
            mu: Mean of shape [batch, channels, height, width]
            log_var: Log variance of shape [batch, channels, height, width]
        Returns:
            z: Sampled latent variable of same shape as mu
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def get_time_pos_embedding(self, time_feats, pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:, idx].unsqueeze(dim=1) * pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out, tf], dim=1)

        return final_out

    def predict_trajectory(
        self,
        x0,
        t_timesteps,
        x_history=None,
        solver_method="rk4",
        atol=0.01,
        rtol=0.01,
    ):
        """
        Predict trajectory using coupled velocity-state ODE solver with relaxation.

        Args:
            x0: Initial state [B, C, H, W]
            t_timesteps: 1D tensor of time "units" consistent with training (e.g., [0, 1, 2, ...]).
                        This function will convert to physical hours and then to the solver nondimensional time
                        using HOURS_PER_UNIT and TIME_SCALE defined above.
            x_history: Optional history [B, history_size, C, H, W]
            solver_method: ODE solver method
            v_initial: Optional initial velocity [B, 2*C, H, W]. If None, uses zero velocity.

        Returns:
            trajectory: Stacked predictions [time_steps, batch, channels, height, width]
        """
        x_initial = x0.to(x0.device)  # shape [B, C, H, W]

        # Convert provided time units to physical hours, then to solver (scaled) time
        t_hours = t_timesteps.float().to(x0.device) * HOURS_PER_UNIT
        t_solver = (t_hours / TIME_SCALE).to(
            x0.device
        )  # scaled nondimensional times used by solver

        # ODE integration
        trajectory_states = [x_initial]  # Track state trajectory only
        history_buffer = x_history  # shape [B, history_size, C, H, W] or None

        # Helper ode function: when called by odeint, `t` will be scalar (0-d tensor). We need to expand it to batch.

        def ode_func(t, vs):
            B = vs.shape[0]
            # create a batch-shaped time tensor [B,1] with the same scalar t
            if t.dim() == 0:
                t_batch = t.unsqueeze(0).repeat(B, 1)  # [B,1]
            elif t.dim() == 1 and t.shape[0] == B:
                t_batch = t.unsqueeze(1)
            else:
                t_batch = t
            return self.forward(
                vs,
                t_batch,
                x_history=history_buffer,
                z=None,
                return_velocity=False,
                return_velocity_components=False,
            )

        if history_buffer is None:
            # integrate in one shot
            solution = odeint(
                ode_func,
                x_initial,
                t_solver,
                method=solver_method,
            )
            # Extract state from coupled solution [T, B, 2*C + C, H, W] -> [T, B, C, H, W]
            state_trajectory = solution
            return state_trajectory
        else:
            # integrate step-by-step to allow updating history between intervals

            t_start = t_solver[0].item()
            x_current = x_initial

            for idx in range(1, len(t_solver)):
                t_end = t_solver[idx].item()
                times = torch.tensor(
                    [t_start, t_end], device=x0.device, dtype=t_solver.dtype
                )
                solution = odeint(
                    ode_func,
                    x_current,
                    times,
                    method=solver_method,
                    rtol=1e-2,
                    atol=1e-2,
                )
                t_start = t_end
                x_current = solution[-1]

                # Extract state for trajectory
                trajectory_states.append(x_current)

                # Update history buffer with previously predicted state
                if self.history_size > 0:
                    history_buffer = torch.cat(
                        [
                            history_buffer[:, 1:, :, :, :],
                            trajectory_states[-2].unsqueeze(1),
                        ],
                        dim=1,
                    )

        res = torch.stack(trajectory_states)
        return res

    def flow_matching_loss(
        self,
        x0,
        x1,
        t0,
        t1,
        x_history=None,
        lambda_output=0.0,
        lambda_grad=0.0,
        lambda_selective=0.0,
    ):
        """
        Combined Flow Matching + VAE loss function (sigma removed as it's not used)
        Args:
            x0: Starting state at time t0, shape [B, C, H, W]
            x1: Target state at time t1, shape [B, C, H, W]
            t0, t1: tensors of time units (one per batch element) consistent with HOURS_PER_UNIT
        Returns:
            combined_loss, fm_loss, kl_loss, x_t, v_target, t_scaled, output_reg, grad_reg, l2_norm
        """

        t_noise = torch.rand(x0.shape[0], 1).to(x0.device)

        t0 = (t0.float() * HOURS_PER_UNIT) / TIME_SCALE
        t1 = (t1.float() * HOURS_PER_UNIT) / TIME_SCALE

        # Linear interpolation between x0 and x1
        t_noise = t_noise.reshape(*t_noise.shape, *([1] * (x0.ndim - t_noise.ndim)))

        # Ensure dt has right shape for broadcasting to [B, C, H, W]
        dt_scaled = (t1 - t0).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        t_fm = t_noise * dt_scaled + t0.reshape_as(dt_scaled)  # [B, 1]

        if self.var_noise > 0:
            # Stochastic Flow Matching
            sigma_t = self.var_noise * torch.sqrt(t_noise * (1 - t_noise) * dt_scaled)
            x_t = (1 - t_noise) * x0 + t_noise * x1 + sigma_t * torch.randn_like(x0)

            # Conditional velocity
            v_target = (x1 - x0) + (1 - 2 * t_noise) * (self.var_noise**2) * (
                x_t - (1 - t_noise) * x0 - t_noise * x1
            ) / (sigma_t**2 + EPS_DT)
            v_target = v_target / dt_scaled
        else:
            x_t = (1 - t_noise) * x0 + t_noise * x1
            v_target = (x1 - x0) / dt_scaled

        # Network prediction expects t_scaled
        ds_pred, v_basic, source_pred = self.forward(
            x_t, t_fm, x_history=x_history, return_velocity_components=True
        )

        # Advection part of the loss
        advection_target = v_target
        v_loss = nn.MSELoss()(ds_pred, advection_target)

        fm_loss = v_loss

        # Selective Regularization Terms
        l2_norm = 0.0
        if lambda_selective > 0:
            for name, param in self.named_parameters():
                # Only regularize final velocity output layers
                if "vel_f" in name or "source_net" in name:
                    l2_norm += param.pow(2).sum()

        # Velocity Regularization: penalize large velocity magnitudes
        # Use v_basic (internal velocity) for regularization instead of v_pred (state derivative)
        output_reg = torch.tensor(0.0, device=ds_pred.device)
        if lambda_output > 0:
            output_reg = torch.mean(v_basic**2)

        # Gradient penalty: encourage smooth velocity fields (suppress high-frequency noise)
        grad_reg = torch.tensor(0.0, device=ds_pred.device)
        if lambda_grad > 0:
            v_grad_x = torch.gradient(v_basic, dim=3)[0]
            v_grad_y = torch.gradient(v_basic, dim=2)[0]
            grad_reg = torch.mean(v_grad_x**2 + v_grad_y**2)

        combined_loss = (
            fm_loss
            + lambda_selective * l2_norm
            + lambda_output * output_reg
            + lambda_grad * grad_reg
        )

        return (
            combined_loss,
            fm_loss,
            v_loss,
            x_t,
            v_target,
            t_fm,
            output_reg,
            grad_reg,
            l2_norm,
        )

    def forward(
        self,
        x_t,
        flow_time,
        x_history=None,
        z=None,
        return_velocity=False,
        return_velocity_components=False,
    ):
        """
        Forward pass with optional latent variable z
        Args:
            x_t: Current state [B, C, H, W]
            flow_time: time input in the same units the rest of the class expects: SCALED solver time (i.e. hours/TIME_SCALE).
                       Expected shape: [B, 1] (but function tolerates scalar/1D times passed by odeint).
            z: Latent variable from VAE encoder (optional, for inference)
            return_velocity: If True, returns (ds, v_basic). If False, returns ds.
            return_velocity_components: If True, returns (ds, v_basic, source).
        """
        # flow_time is expected to be scaled time (t_hours / TIME_SCALE)
        t_scaled = flow_time.float().to(x_t.device)

        # Recover physical hours for periodic embeddings
        t_hours = t_scaled * TIME_SCALE
        t_emb = t_hours % HOURS_in_DAY  # hours in day

        # t_hours can be [B,1] from training or a scalar expanded in predict_trajectory
        t_emb = t_emb.view(-1, 1, 1, 1).expand(
            x_t.shape[0], 1, x_t.shape[2], x_t.shape[3]
        )

        # Diurnal (12-hour) and seasonal (yearly) periodicity computed using physical hours
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)

        day_emb = torch.cat([sin_t_emb, cos_t_emb], dim=1)
        seas_emb = torch.cat([sin_seas_emb, cos_seas_emb], dim=1)

        # Get physics features from vel_samples
        # vel_samples is set by update_param_phys() which is called before every forward()
        # phys = (
        #    self.vel_samples[:6]
        #    if not torch.cuda.is_available()
        #    else self.vel_samples
        # )
        # phys = self.phys_conv(phys)

        # Compute Spatial gradients
        ds_grad_x = torch.gradient(x_t, dim=3)[0]
        ds_grad_y = torch.gradient(x_t, dim=2)[0]
        nabla_u = torch.cat([ds_grad_x, ds_grad_y], dim=1)

        input_feats = [
            t_emb / HOURS_in_DAY,
            day_emb,
            seas_emb,
            nabla_u,
            x_t,
        ]  # ,phys

        # Basic feature combination with z
        # If z is not provided (e.g., during initial training), use zeros
        if z is not None:
            pass

        # Normalize raw time feature (not the sin/cos, which are already bounded)
        # Raw time needs normalization to help network learning
        if self.use_batch_norm:
            x_t_norm = self.normalize_state(x_t)
            nabla_u_norm = self.normalize_gradients(nabla_u)

        if self.history_size > 0 and x_history is not None:
            if self.use_history_attention:
                # Use attention over history: more parameter efficient and learns temporal dependencies
                # Add positional embeddings to distinguish different timesteps
                x_hist_with_pos = x_history + self.history_pos_emb
                # Attention output: [B, C, H, W] - compressed representation of history
                hist_features_encoded = self.history_attn(x_t, x_hist_with_pos)
                # add to input features
                input_feats = input_feats + [hist_features_encoded]
            else:
                # Original approach: flatten and concatenate history
                x_hist_concat = x_history.reshape(
                    x_t.shape[0],
                    self.history_size * self.out_ch,
                    *x_t.shape[2:],
                )
                input_feats = input_feats + [x_hist_concat]

        if self.pos:
            lat_map = self.lat_map.unsqueeze(dim=0) * torch.pi / 180
            lon_map = self.lon_map.unsqueeze(dim=0) * torch.pi / 180
            pos_rep = torch.cat(
                [
                    lat_map.unsqueeze(dim=0),
                    lon_map.unsqueeze(dim=0),
                    self.const_info,
                ],
                dim=1,
            )
            self.pos_feat = self.pos_enc(pos_rep).expand(
                x_t.shape[0], -1, x_t.shape[2], x_t.shape[3]
            )

            input_feats = input_feats + [self.pos_feat]
        else:
            self.oro, self.lsm = self.const_info[0, 0], self.const_info[0, 1]
            self.lsm = self.lsm.unsqueeze(dim=0).expand(
                x_t.shape[0], -1, x_t.shape[2], x_t.shape[3]
            )
            self.oro = (
                F.normalize(self.const_info[0, 0])
                .unsqueeze(dim=0)
                .expand(x_t.shape[0], -1, x_t.shape[2], x_t.shape[3])
            )
            self.new_lat_map = (
                self.lat_map.expand(x_t.shape[0], 1, x_t.shape[2], x_t.shape[3])
                * torch.pi
                / 180
            )  # Converting to radians
            self.new_lon_map = (
                self.lon_map.expand(x_t.shape[0], 1, x_t.shape[2], x_t.shape[3])
                * torch.pi
                / 180
            )
            cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(
                self.new_lat_map
            )
            cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(
                self.new_lon_map
            )
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )

            t_cyc_emb = torch.cat([day_emb, seas_emb], dim=1)
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )
            pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_feats)
            input_feats = input_feats + [
                self.new_lat_map,
                self.new_lon_map,
                self.lsm,
                self.oro,
                pos_feats,
                pos_time_ft,
            ]

        comb_rep_basic = torch.cat(input_feats, dim=1)

        # Velocity field
        v_basic = self.vel_f(comb_rep_basic)
        if self.att:  # Self-attention mechanism
            v_basic = v_basic + self.gamma * self.vel_att(comb_rep_basic)

        # Separate velocity components
        v_x = v_basic[:, : self.out_ch, :, :]
        v_y = v_basic[:, self.out_ch : 2 * self.out_ch, :, :]

        # First Term: Simple advection
        adv1 = v_x * ds_grad_x + v_y * ds_grad_y

        # Second Term: Include compression term with proper gradient computation
        div_v = torch.gradient(v_x, dim=3)[0] + torch.gradient(v_y, dim=2)[0]
        adv2 = x_t * div_v

        advection = adv1 + adv2
        source = self.source_net(comb_rep_basic)
        ds = advection + source

        if return_velocity_components:
            return ds, v_basic, source
        if return_velocity:
            return ds, v_basic
        return ds


class Climate_VFM_Monthly(nn.Module):

    def __init__(
        self,
        num_channels,
        history_size,
        const_channels,
        out_types,
        use_att,
        use_err,
        use_pos,
        use_batch_norm=False,
        var_noise=0.1,
        use_history_attention=True,
    ):
        super().__init__()

        self.var_noise = var_noise

        self.layers = [4, 3, 2]
        output_dim = 10
        input_channels = 40  # input_dim + out_types*int(use_pos) + 34*(1-int(use_pos))
        # self.hidden = [8, output_dim]
        self.hidden = [
            128,
            64,
            output_dim,
        ]  # [128, 64, output_dim] #[8, 64, output_dim]
        self.history_size = history_size
        self.use_history_attention = use_history_attention

        # If using attention, history contributes out_types channels; otherwise history_size * out_types
        if use_history_attention and history_size > 0:
            input_channels += (
                out_types  # Attention output is compressed to out_types channels
            )
        else:
            input_channels += history_size * out_types  # Concatenated history

        self.vel_f = Climate_ResNet_2D(input_channels, self.layers, self.hidden)
        # head for source term (unmodeled physics)
        self.source_net = Climate_ResNet_2D(input_channels, [2, 1], [64, out_types])

        # History Attention module (if enabled)
        if use_history_attention and history_size > 0:
            self.history_attn = HistoryAttentionEncoder(
                channels=out_types, num_heads=out_types
            )
            # Learnable positional embeddings for each history timestep
            # Helps model distinguish between t-1, t-2, etc.
            self.history_pos_emb = nn.Parameter(
                torch.randn(1, history_size, out_types, 1, 1) * 0.02
            )

        if use_att:
            self.vel_att = Self_attn_conv(input_channels, output_dim)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

        self.scales = num_channels
        self.const_channel = const_channels

        self.out_ch = out_types
        self.vel_samples = 0
        self.const_info = 0
        self.lat_map = 0
        self.lon_map = 0
        self.elev = 0
        self.pos_emb = 0
        self.elev_info_grad_x = 0
        self.elev_info_grad_y = 0
        self.pos_feat = 0
        self.lsm = 0
        self.oro = 0

        # Feature normalization flags and modules
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            # Use InstanceNorm (GroupNorm with num_groups=channels) instead of BatchNorm/LayerNorm
            # InstanceNorm is stable at ANY batch size (including batch_size=1 during inference)
            # Each channel of each sample is normalized independently - no batch dependency
            self.normalize_state = nn.GroupNorm(
                num_groups=out_types, num_channels=out_types, affine=True
            )
            self.normalize_gradients = nn.GroupNorm(
                num_groups=2 * out_types,
                num_channels=2 * out_types,
                affine=True,
            )

        self.att = use_att
        self.err = use_err
        self.pos = use_pos

    def update_param_phys(self, params):
        # self.past_samples = params[0]  # vel samples
        self.vel_samples = params[0]
        self.const_info = params[1]
        self.lat_map = params[2]
        self.lon_map = params[3]

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE
        Args:
            mu: Mean of shape [batch, channels, height, width]
            log_var: Log variance of shape [batch, channels, height, width]
        Returns:
            z: Sampled latent variable of same shape as mu
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def get_time_pos_embedding(self, time_feats, pos_feats):
        for idx in range(time_feats.shape[1]):
            tf = time_feats[:, idx].unsqueeze(dim=1) * pos_feats
            if idx == 0:
                final_out = tf
            else:
                final_out = torch.cat([final_out, tf], dim=1)

        return final_out

    def predict_trajectory(
        self,
        x0,
        t_timesteps,
        x_history=None,
        solver_method="rk4",
        atol=0.01,
        rtol=0.01,
    ):
        """
        Predict trajectory using coupled velocity-state ODE solver with relaxation.

        Args:
            x0: Initial state [B, C, H, W]
            t_timesteps: 1D tensor of time "units" consistent with training (e.g., [0, 1, 2, ...]).
                        This function will convert to physical hours and then to the solver nondimensional time
                        using HOURS_PER_UNIT and TIME_SCALE defined above.
            x_history: Optional history [B, history_size, C, H, W]
            solver_method: ODE solver method
            v_initial: Optional initial velocity [B, 2*C, H, W]. If None, uses zero velocity.

        Returns:
            trajectory: Stacked predictions [time_steps, batch, channels, height, width]
        """
        x_initial = x0.to(x0.device)  # shape [B, C, H, W]

        # Convert provided time units to physical hours, then to solver (scaled) time
        t_hours = t_timesteps.float().to(x0.device) * HOURS_PER_UNIT
        t_solver = (t_hours / TIME_SCALE).to(
            x0.device
        )  # scaled nondimensional times used by solver

        # ODE integration
        trajectory_states = [x_initial]  # Track state trajectory only
        history_buffer = x_history  # shape [B, history_size, C, H, W] or None

        # Helper ode function: when called by odeint, `t` will be scalar (0-d tensor). We need to expand it to batch.

        def ode_func(t, vs):
            B = vs.shape[0]
            # create a batch-shaped time tensor [B,1] with the same scalar t
            if t.dim() == 0:
                t_batch = t.unsqueeze(0).repeat(B, 1)  # [B,1]
            elif t.dim() == 1 and t.shape[0] == B:
                t_batch = t.unsqueeze(1)
            else:
                t_batch = t
            return self.forward(
                vs,
                t_batch,
                x_history=history_buffer,
                z=None,
                return_velocity=False,
                return_velocity_components=False,
            )

        if history_buffer is None:
            # integrate in one shot
            solution = odeint(
                ode_func,
                x_initial,
                t_solver,
                method=solver_method,
            )
            # Extract state from coupled solution [T, B, 2*C + C, H, W] -> [T, B, C, H, W]
            state_trajectory = solution
            return state_trajectory
        else:
            # integrate step-by-step to allow updating history between intervals

            t_start = t_solver[0].item()
            x_current = x_initial

            for idx in range(1, len(t_solver)):
                t_end = t_solver[idx].item()
                times = torch.tensor(
                    [t_start, t_end], device=x0.device, dtype=t_solver.dtype
                )
                solution = odeint(
                    ode_func,
                    x_current,
                    times,
                    method=solver_method,
                    rtol=1e-2,
                    atol=1e-2,
                )
                t_start = t_end
                x_current = solution[-1]

                # Extract state for trajectory
                trajectory_states.append(x_current)

                # Update history buffer with previously predicted state
                if self.history_size > 0:
                    history_buffer = torch.cat(
                        [
                            history_buffer[:, 1:, :, :, :],
                            trajectory_states[-2].unsqueeze(1),
                        ],
                        dim=1,
                    )

        res = torch.stack(trajectory_states)
        return res

    def flow_matching_loss(
        self,
        x0,
        x1,
        t0,
        t1,
        x_history=None,
        lambda_output=0.0,
        lambda_grad=0.0,
        lambda_selective=0.0,
    ):
        """
        Combined Flow Matching + VAE loss function (sigma removed as it's not used)
        Args:
            x0: Starting state at time t0, shape [B, C, H, W]
            x1: Target state at time t1, shape [B, C, H, W]
            t0, t1: tensors of time units (one per batch element) consistent with HOURS_PER_UNIT
        Returns:
            combined_loss, fm_loss, kl_loss, x_t, v_target, t_scaled, output_reg, grad_reg, l2_norm
        """

        t_noise = torch.rand(x0.shape[0], 1).to(x0.device)

        t0 = (t0.float() * HOURS_PER_UNIT) / TIME_SCALE
        t1 = (t1.float() * HOURS_PER_UNIT) / TIME_SCALE

        # Linear interpolation between x0 and x1
        t_noise = t_noise.reshape(*t_noise.shape, *([1] * (x0.ndim - t_noise.ndim)))

        # Ensure dt has right shape for broadcasting to [B, C, H, W]
        dt_scaled = (t1 - t0).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        t_fm = t_noise * dt_scaled + t0.reshape_as(dt_scaled)  # [B, 1]

        if self.var_noise > 0:
            # Stochastic Flow Matching
            sigma_t = self.var_noise * torch.sqrt(t_noise * (1 - t_noise) * dt_scaled)
            x_t = (1 - t_noise) * x0 + t_noise * x1 + sigma_t * torch.randn_like(x0)

            # Conditional velocity
            v_target = (x1 - x0) + (1 - 2 * t_noise) * (self.var_noise**2) * (
                x_t - (1 - t_noise) * x0 - t_noise * x1
            ) / (sigma_t**2 + EPS_DT)
            v_target = v_target / dt_scaled
        else:
            x_t = (1 - t_noise) * x0 + t_noise * x1
            v_target = (x1 - x0) / dt_scaled

        # Network prediction expects t_scaled
        ds_pred, v_basic, source_pred = self.forward(
            x_t, t_fm, x_history=x_history, return_velocity_components=True
        )

        # Advection part of the loss
        advection_target = v_target
        v_loss = nn.MSELoss()(ds_pred, advection_target)

        fm_loss = v_loss

        # Selective Regularization Terms
        l2_norm = 0.0
        if lambda_selective > 0:
            for name, param in self.named_parameters():
                # Only regularize final velocity output layers
                if "vel_f" in name or "source_net" in name:
                    l2_norm += param.pow(2).sum()

        # Velocity Regularization: penalize large velocity magnitudes
        # Use v_basic (internal velocity) for regularization instead of v_pred (state derivative)
        output_reg = torch.tensor(0.0, device=ds_pred.device)
        if lambda_output > 0:
            output_reg = torch.mean(v_basic**2)

        # Gradient penalty: encourage smooth velocity fields (suppress high-frequency noise)
        grad_reg = torch.tensor(0.0, device=ds_pred.device)
        if lambda_grad > 0:
            v_grad_x = torch.gradient(v_basic, dim=3)[0]
            v_grad_y = torch.gradient(v_basic, dim=2)[0]
            grad_reg = torch.mean(v_grad_x**2 + v_grad_y**2)

        combined_loss = (
            fm_loss
            + lambda_selective * l2_norm
            + lambda_output * output_reg
            + lambda_grad * grad_reg
        )

        return (
            combined_loss,
            fm_loss,
            v_loss,
            x_t,
            v_target,
            t_fm,
            output_reg,
            grad_reg,
            l2_norm,
        )

    def forward(
        self,
        x_t,
        flow_time,
        x_history=None,
        z=None,
        return_velocity=False,
        return_velocity_components=False,
    ):
        """
        Forward pass with optional latent variable z
        Args:
            x_t: Current state [B, C, H, W]
            flow_time: time input in the same units the rest of the class expects: SCALED solver time (i.e. hours/TIME_SCALE).
                       Expected shape: [B, 1] (but function tolerates scalar/1D times passed by odeint).
            z: Latent variable from VAE encoder (optional, for inference)
            return_velocity: If True, returns (ds, v_basic). If False, returns ds.
            return_velocity_components: If True, returns (ds, v_basic, source).
        """
        # flow_time is expected to be scaled time (t_hours / TIME_SCALE)
        t_scaled = flow_time.float().to(x_t.device)

        # Recover physical hours for periodic embeddings
        t_hours = t_scaled * TIME_SCALE
        t_emb = t_hours % HOURS_PER_UNIT  # per unit

        # t_hours can be [B,1] from training or a scalar expanded in predict_trajectory
        t_emb = t_emb.view(-1, 1, 1, 1).expand(
            x_t.shape[0], 1, x_t.shape[2], x_t.shape[3]
        )

        # 12 hour periodicity computed using physical hours
        sin_t_emb = torch.sin(torch.pi * t_emb / (12 * HOURS_PER_UNIT) - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / (12 * HOURS_PER_UNIT) - torch.pi / 2)
        day_emb = torch.cat([sin_t_emb, cos_t_emb], dim=1)

        # Compute Spatial gradients
        ds_grad_x = torch.gradient(x_t, dim=3)[0]
        ds_grad_y = torch.gradient(x_t, dim=2)[0]
        nabla_u = torch.cat([ds_grad_x, ds_grad_y], dim=1)

        input_feats = [
            t_emb / HOURS_PER_UNIT,
            day_emb,
            nabla_u,
            x_t,
        ]  # MONTHLY embeddings

        # Basic feature combination with z
        # If z is not provided (e.g., during initial training), use zeros
        if z is not None:
            pass

        # Normalize raw time feature (not the sin/cos, which are already bounded)
        # Raw time needs normalization to help network learning
        if self.use_batch_norm:
            x_t_norm = self.normalize_state(x_t)
            nabla_u_norm = self.normalize_gradients(nabla_u)

        if self.history_size > 0 and x_history is not None:
            if self.use_history_attention:
                # Use attention over history: more parameter efficient and learns temporal dependencies
                # Add positional embeddings to distinguish different timesteps
                x_hist_with_pos = x_history + self.history_pos_emb
                # Attention output: [B, C, H, W] - compressed representation of history
                hist_features_encoded = self.history_attn(x_t, x_hist_with_pos)
                # add to input features
                input_feats = input_feats + [hist_features_encoded]
            else:
                # Original approach: flatten and concatenate history
                x_hist_concat = x_history.reshape(
                    x_t.shape[0],
                    self.history_size * self.out_ch,
                    *x_t.shape[2:],
                )
                input_feats = input_feats + [x_hist_concat]

        if self.pos:
            lat_map = self.lat_map.unsqueeze(dim=0) * torch.pi / 180
            lon_map = self.lon_map.unsqueeze(dim=0) * torch.pi / 180
            pos_rep = torch.cat(
                [
                    lat_map.unsqueeze(dim=0),
                    lon_map.unsqueeze(dim=0),
                    self.const_info,
                ],
                dim=1,
            )
            self.pos_feat = self.pos_enc(pos_rep).expand(
                x_t.shape[0], -1, x_t.shape[2], x_t.shape[3]
            )
            input_feats = input_feats + [self.pos_feat]
        else:
            self.oro, self.lsm = self.const_info[0, 0], self.const_info[0, 1]
            self.lsm = self.lsm.unsqueeze(dim=0).expand(
                x_t.shape[0], -1, x_t.shape[2], x_t.shape[3]
            )
            self.oro = (
                F.normalize(self.const_info[0, 0])
                .unsqueeze(dim=0)
                .expand(x_t.shape[0], -1, x_t.shape[2], x_t.shape[3])
            )
            self.new_lat_map = (
                self.lat_map.expand(x_t.shape[0], 1, x_t.shape[2], x_t.shape[3])
                * torch.pi
                / 180
            )  # Converting to radians
            self.new_lon_map = (
                self.lon_map.expand(x_t.shape[0], 1, x_t.shape[2], x_t.shape[3])
                * torch.pi
                / 180
            )
            cos_lat_map, sin_lat_map = torch.cos(self.new_lat_map), torch.sin(
                self.new_lat_map
            )
            cos_lon_map, sin_lon_map = torch.cos(self.new_lon_map), torch.sin(
                self.new_lon_map
            )
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )

            t_cyc_emb = day_emb  # MONTHLY embeddings
            pos_feats = torch.cat(
                [
                    cos_lat_map,
                    cos_lon_map,
                    sin_lat_map,
                    sin_lon_map,
                    sin_lat_map * cos_lon_map,
                    sin_lat_map * sin_lon_map,
                ],
                dim=1,
            )
            pos_time_ft = self.get_time_pos_embedding(t_cyc_emb, pos_feats)
            input_feats = input_feats + [
                self.new_lat_map,
                self.new_lon_map,
                self.lsm,
                self.oro,
                pos_feats,
                pos_time_ft,
            ]

        comb_rep_basic = torch.cat(input_feats, dim=1)

        # Velocity field
        v_basic = self.vel_f(comb_rep_basic)
        if self.att:  # Self-attention mechanism
            v_basic = v_basic + self.gamma * self.vel_att(comb_rep_basic)

        # Separate velocity components
        v_x = v_basic[:, : self.out_ch, :, :]
        v_y = v_basic[:, self.out_ch : 2 * self.out_ch, :, :]

        # First Term: Simple advection
        adv1 = v_x * ds_grad_x + v_y * ds_grad_y

        # Second Term: Include compression term with proper gradient computation
        div_v = torch.gradient(v_x, dim=3)[0] + torch.gradient(v_y, dim=2)[0]
        adv2 = x_t * div_v

        advection = adv1 + adv2
        source = self.source_net(comb_rep_basic)
        ds = advection + source

        if return_velocity_components:
            return ds, v_basic, source
        if return_velocity:
            return ds, v_basic
        return ds
