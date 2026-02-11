import torch


def kl_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    p = torch.distributions.Normal(mu_1, sigma_1)
    q = torch.distributions.Normal(mu_2, sigma_2)
    kl = torch.distributions.kl_divergence(p, q)
    return kl.sum(1)


@torch.jit.script
def kldiv_normal_normal(
    mean1: torch.Tensor,
    lnvar1: torch.Tensor,
    mean2: torch.Tensor,
    lnvar2: torch.Tensor,
):
    """
    KL divergence between normal distributions, KL( N(mean1, diag(exp(lnvar1))) || N(mean2, diag(exp(lnvar2))) )
    """
    if lnvar1.ndim == 2 and lnvar2.ndim == 2:
        return 0.5 * torch.sum(
            (lnvar1 - lnvar2).exp()
            - 1.0
            + lnvar2
            - lnvar1
            + (mean2 - mean1).pow(2) / lnvar2.exp(),
            dim=1,
        )
    elif lnvar1.ndim == 1 and lnvar2.ndim == 1:
        d = mean1.shape[1]
        return 0.5 * (
            d * ((lnvar1 - lnvar2).exp() - 1.0 + lnvar2 - lnvar1)
            + torch.sum((mean2 - mean1).pow(2), dim=1) / lnvar2.exp()
        )
    else:
        raise ValueError()
