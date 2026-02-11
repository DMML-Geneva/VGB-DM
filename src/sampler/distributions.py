import torch
from torch.distributions import Uniform
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets


class LoaderSampler:
    def __init__(self, loader, device="cuda"):
        self.device = device
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=-1):
        assert size <= self.loader.batch_size
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)

        if self.loader.batch_size < size:
            return self.sample(size)

        if size > 0 and size < self.loader.batch_size:
            if isinstance(batch, dict):
                for k in batch:
                    batch[k] = batch[k][:size]
            else:
                if size > 0:
                    batch = batch[:size]

        return batch

    def reset(self):
        self.it = iter(self.loader)


class Sampler:
    def __init__(
        self,
        device="cuda",
        seed=None,
    ):
        self.device = torch.device(device)
        self.seed = seed
        self.rnd_generator = torch.Generator(device=device)
        if seed is not None:
            self.rnd_generator.manual_seed(seed)

    def sample(self, size=5):
        pass


class BoxUniformSampler(Sampler):
    def __init__(self, lower_bounds, upper_bounds, seed=None, device="cuda"):
        super().__init__(device=device, seed=seed)
        if isinstance(lower_bounds, float):
            lower_bounds = [lower_bounds]
        if isinstance(upper_bounds, float):
            upper_bounds = [upper_bounds]
        if not isinstance(lower_bounds, torch.Tensor):
            lower_bounds = torch.tensor(lower_bounds, device=device)
        if not isinstance(upper_bounds, torch.Tensor):
            upper_bounds = torch.tensor(upper_bounds, device=device)
        assert lower_bounds.shape == upper_bounds.shape
        assert lower_bounds.ndim == 1
        assert torch.all(lower_bounds <= upper_bounds)

        self.ranges = torch.stack((lower_bounds, upper_bounds), dim=1)
        # Remove constant dimensions
        self.init_ranges = self.ranges[self.ranges[:, 0] != self.ranges[:, 1]]
        self.uniform_sampler = Uniform(
            self.init_ranges[:, 0], self.init_ranges[:, 1]
        )
        self.constant_positions = torch.where(
            self.ranges[:, 0] == self.ranges[:, 1]
        )[0]
        self.non_constant_positions = torch.where(
            self.ranges[:, 0] != self.ranges[:, 1]
        )[0]
        self.device = device

    def sample(self, size):
        samples = self.uniform_sampler.sample(size)

        if samples.ndim > 2:
            samples = samples.reshape(-1, samples.shape[-1])

        if len(self.constant_positions) > 0:
            res = torch.empty(
                samples.shape[0], self.ranges.shape[0], device=self.device
            )
            res[:, self.non_constant_positions] = samples
        else:
            return samples

        res[:, self.constant_positions] = self.ranges[
            self.constant_positions, 0
        ]
        return res.reshape(size + (-1,))


class Bernoulli(Sampler):
    def __init__(
        self,
        p=0.5,
        head=1,
        tail=0,
        seed=None,
        device="cpu",
    ):
        super().__init__(seed=seed, device=device)
        self.head = head
        self.head = torch.atleast_2d(head)
        self.tail = torch.atleast_2d(tail)
        self.p = p
        if isinstance(self.p, list):
            self.p = self.p[0]
        if torch.is_tensor(self.p):
            self.p = self.p.item()

    def sample(self, size=(1,)):
        if isinstance(size, int):
            size = (size, 1)

        tosses = torch.bernoulli(
            self.p * torch.ones(size, device=self.device),
            generator=self.rnd_generator,
        )
        tosses = tosses.view(-1, 1).to(bool)
        samples = self.head * tosses
        samples += self.tail * ~tosses
        return samples.reshape(size + (self.head.shape[-1],))


class Categorical(Sampler):
    def __init__(
        self,
        probs=[0.5],
        classes=[0, 1],
        seed=None,
        device="cpu",
    ):
        super().__init__(seed=seed, device=device)
        self.probs = torch.tensor(probs, device=device)
        assert self.probs.sum() == 1
        self.classes = torch.tensor(classes, device=device)
        assert self.classes.shape[0] == self.probs.shape[0]
        self.categorical = torch.distributions.Categorical(
            probs=torch.tensor(probs, device=device)
        )

    def sample(self, size=(1,)):
        samples = self.categorical.sample(size)
        return self.classes[samples]


class LoaderSampler(Sampler):
    def __init__(self, loader, device="cuda", seed=None):
        super(LoaderSampler, self).__init__(device=device, seed=seed)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)


class SwissRollSampler(Sampler):
    def __init__(self, dim=2, device="cuda", seed=None):
        super(SwissRollSampler, self).__init__(device=device, seed=seed)
        assert dim == 2
        self.dim = 2

    def sample(self, batch_size=10):
        batch = (
            datasets.make_swiss_roll(n_samples=batch_size, noise=0.8)[
                0
            ].astype("float32")[:, [0, 2]]
            / 7.5
        )
        return torch.tensor(batch, device=self.device)


class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device="cuda", seed=None):
        super(StandardNormalSampler, self).__init__(device=device, seed=seed)
        self.dim = dim

    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)


class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, seed=None, device="cuda"
    ):
        super(CubeUniformSampler, self).__init__(device=device, seed=seed)
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = (
            np.eye(self.dim, dtype=np.float32)
            if self.normalized
            else np.eye(self.dim, dtype=np.float32) / 12
        )
        self.mean = (
            np.zeros(self.dim, dtype=np.float32)
            if self.centered
            else 0.5 * np.ones(self.dim, dtype=np.float32)
        )

        self.bias = torch.tensor(self.mean, device=self.device)

    def sample(self, size=10):
        with torch.no_grad():
            sample = (
                np.sqrt(self.var)
                * (torch.rand(size, self.dim, device=self.device) - 0.5)
                / np.sqrt(self.dim / 12)
                + self.bias
            )
        return sample


class MixN2GaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, seed=None, device="cuda"):
        super(MixN2GaussiansSampler, self).__init__(device=device, seed=seed)

        assert dim == 2
        self.dim = 2
        self.std, self.step = std, step

        self.n = n

        grid_1d = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
        xx, yy = np.meshgrid(grid_1d, grid_1d)
        centers = np.stack([xx, yy]).reshape(2, -1).T
        self.centers = torch.tensor(
            centers,
            device=self.device,
        )

    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        indices = random.choices(range(len(self.centers)), k=batch_size)
        with torch.no_grad():
            batch *= self.std
            batch += self.step * self.centers[indices, :]
        return batch


class MixNGaussiansSampler(Sampler):
    def __init__(self, n=5, dim=2, std=1, step=9, seed=None, device="cuda"):
        super(MixNGaussiansSampler, self).__init__(device=device, seed=seed)

        assert dim == 1
        self.dim = 1
        self.std, self.step = std, step

        self.n = n

        grid_1d = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
        self.centers = torch.tensor(
            grid_1d,
            device=self.device,
        )

    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        indices = random.choices(range(len(self.centers)), k=batch_size)
        with torch.no_grad():
            batch *= self.std
            batch += self.step * self.centers[indices, None]
        return batch


class Mix8GaussiansSampler(Sampler):
    def __init__(
        self, with_central=False, std=1, r=12, dim=2, seed=None, device="cuda"
    ):
        super(Mix8GaussiansSampler, self).__init__(device=device, seed=seed)
        assert dim == 2
        self.dim = 2
        self.std, self.r = std, r

        self.with_central = with_central
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        if self.with_central:
            centers.append((0, 0))
        self.centers = torch.tensor(
            centers, device=self.device, dtype=torch.float32
        )

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.randn(batch_size, self.dim, device=self.device)
            indices = random.choices(range(len(self.centers)), k=batch_size)
            batch *= self.std
            batch += self.r * self.centers[indices, :]
        return batch


class SphereUniformSampler(Sampler):
    def __init__(self, dim=1, device="cuda", seed=None):
        super(SphereUniformSampler, self).__init__(device=device, seed=seed)
        self.dim = dim

    def sample(self, batch_size=10):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        batch /= torch.norm(batch, dim=1)[:, None]
        return torch.tensor(batch, device=self.device)


class Transformer(object):
    def __init__(self, device="cuda"):
        self.device = device


class StandardNormalScaler(Transformer):
    def __init__(self, base_sampler, batch_size=1000, device="cuda"):
        super(StandardNormalScaler, self).__init__(device=device)
        self.base_sampler = base_sampler
        batch = self.base_sampler.sample(batch_size).cpu().detach().numpy()

        mean, cov = np.mean(batch, axis=0), np.matrix(np.cov(batch.T))
        self.mean = torch.tensor(mean, device=self.device, dtype=torch.float32)

        multiplier = sqrtm(cov)
        self.multiplier = torch.tensor(
            multiplier, device=self.device, dtype=torch.float32
        )
        self.inv_multiplier = torch.tensor(
            np.linalg.inv(multiplier), device=self.device, dtype=torch.float32
        )
        torch.cuda.empty_cache()

    def sample(self, batch_size=10):
        with torch.no_grad():
            batch = torch.tensor(
                self.base_sampler.sample(batch_size), device=self.device
            )
            batch -= self.mean
            batch @= self.inv_multiplier
        return batch


class LinearTransformer(Transformer):
    def __init__(self, base_sampler, weight, bias=None, device="cuda"):
        super(LinearTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler

        self.weight = torch.tensor(weight, device=device, dtype=torch.float32)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32)
        else:
            self.bias = torch.zeros(
                self.weight.size(0), device=device, dtype=torch.float32
            )

    def sample(self, size=4):
        batch = torch.tensor(
            self.base_sampler.sample(size), device=self.device
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch


class NormalNoiseTransformer(Transformer):
    def __init__(self, base_sampler, std=0.01, device="cuda"):
        super(NormalNoiseTransformer, self).__init__(device=device)
        self.base_sampler = base_sampler
        self.std = std

    def sample(self, batch_size=4):
        batch = self.base_sampler.sample(batch_size)
        with torch.no_grad():
            batch = batch + self.std * torch.randn_like(batch)
        return batch


def dist_sampler(name, *args):
    if name == "uniform":
        return BoxUniformSampler(*args)
    elif name == "bernoulli":
        return Bernoulli(*args)
    elif name == "categorical":
        return Categorical(*args)
    else:
        raise ValueError(f"Unknown distribution {name}")


def sample_latent_noise(
    n_samples, z_dim=1, z_size=4, z_std=0.1, type="gauss", device="cuda"
):
    if type.lower() == "gauss":
        Z = torch.randn(n_samples, z_size, z_dim, device=device) * z_std
    elif type.lower() == "uniform":
        Z = torch.rand(n_samples, z_size, z_dim, device=device)
    elif type.lower() == "bernoulli":
        bernoulli = Bernoulli(
            p=0.5,
            head=torch.ones(z_dim, device=device),
            tail=torch.zeros(z_dim, device=device),
            device=device,
        )
        Z = bernoulli.sample(n_samples * z_size).reshape(
            n_samples, z_size, z_dim
        )
    else:
        raise ValueError(f"Unknown noise latent distribution {type}")
    return Z
