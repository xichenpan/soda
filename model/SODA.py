import os
from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps  # dtype = torch.float64
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def inverted_cosine_beta_schedule(timesteps, s=0.008):
    """
    inverted cosine schedule
    as proposed in https://arxiv.org/pdf/2311.17901.pdf
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps  # dtype = torch.float64
    alphas_cumprod = (2 * (1 + s) / math.pi) * torch.arccos(torch.sqrt(t)) - s
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def schedules(betas, T, device, type='DDPM'):
    if betas == 'inverted':
        schedule_fn = inverted_cosine_beta_schedule
    elif betas == 'cosine':
        schedule_fn = cosine_beta_schedule
    else:
        beta1, beta2 = betas
        schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SODA(nn.Module):
    def __init__(self, encoder, vae, decoder, device, betas, n_T, drop_prob, **kwargs):
        ''' SODA proposed by "SODA: Bottleneck Diffusion Models for Representation Learning", and \
            DDPM proposed by "Denoising Diffusion Probabilistic Models", as well as \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                encoder: A network (e.g. ResNet) which performs image->latent mapping.
                vae: A network (e.g. VAE) which performs image->latent mapping.
                decoder: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T, drop_prob
        '''
        super(SODA, self).__init__()
        self.encoder = encoder.to(device)
        self.vae = vae.to(device)
        self.decoder = decoder.to(device)

        self.freeze_modules([self.encoder, self.vae])

        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
            print(f"encoder # params: {params:.1f}")
            params = sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6
            print(f"decoder # params: {params:.1f}")

        self.device = device
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss = nn.MSELoss()

    def freeze_modules(self, modules):
        ''' Freeze the specified modules.

            Args:
                modules: The list of modules to be frozen.
        '''
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(0, self.n_T, (x.shape[0],)).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def forward(self, x_source, x_target):
        ''' Training with simple noise prediction loss.

            Args:
                x_source: The augmented image tensor.
                x_target: The augmented image tensor ranged in `[0, 1]`.
            Returns:
                The simple MSE loss.
        '''
        x_target = normalize_to_neg_one_to_one(x_target)
        x_target = self.vae.encode(x_target).latent_dist.sample().mul_(0.18215)
        x_noised, t, noise = self.perturb(x_target, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros(x_noised.shape[0]) + self.drop_prob).to(self.device)

        z = self.encoder(x_source)

        B, C = x_t.shape[:2]
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        terms["vb"] = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=x_start,
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )["output"]

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"]

        return loss

    def encode(self, x, norm=False):
        z = self.encoder(x)
        if norm:
            z = torch.nn.functional.normalize(z)
        return z

    def ddim_sample(self, n_sample, size, z_guide, steps=100, eta=0.0, guide_w=0.3, notqdm=False):
        ''' Sampling with DDIM sampler. Actual NFE is `2 * steps`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                z_guide: The latent code extracted from real images (for guidance).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddim_sche
        model_args = self.prepare_condition_(n_sample, z_guide)
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, model_args, guide_w, alpha)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return unnormalize_to_zero_to_one(x_i)

    def pred_eps_(self, x, t, model_args, guide_w, alpha, clip_x=True):
        def pred_cfg_eps_double_batch():
            # double batch
            x_double = x.repeat(2, 1, 1, 1)
            t_double = t.repeat(2)

            eps = self.decoder(x_double, t_double, *model_args).float()
            n_sample = eps.shape[0] // 2
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            assert eps1.shape == eps2.shape
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            return eps

        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        eps = pred_cfg_eps_double_batch()
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised

    def prepare_condition_(self, n_sample, z_guide):
        z_guide = z_guide.repeat(2, 1)

        # 0 for conditional, 1 for unconditional
        mask = torch.zeros(z_guide.shape[0]).to(self.device)
        mask[n_sample:] = 1.
        return z_guide, mask
