import torch
import torch.nn as nn

import math

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

from utils import log_batch_similarity_stats

noise_schedule = NoiseScheduleVP(schedule="linear")


# ---------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps.to(dtype=torch.float32)

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=torch.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    # if embedding_dim % 2 == 1:  # zero pad
    #    emb = torch.pad(emb, [0,1])
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb


import torch
import torch.nn.functional as F


class DiffCDR(nn.Module):
    def __init__(self, num_steps=200, diff_dim=32, input_dim=32, c_scale=0.1, diff_sample_steps=30, diff_task_lambda=0.1, diff_mask_rate=0.1):
        super(DiffCDR, self).__init__()

        # -------------------------------------------
        # define params
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert (
            self.alphas.shape
            == self.alphas_prod.shape
            == self.alphas_prod_p.shape
            == self.alphas_bar_sqrt.shape
            == self.one_minus_alphas_bar_log.shape
            == self.one_minus_alphas_bar_sqrt.shape
        )

        # -----------------------------------------------
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate
        self.global_step = 0
        self.uni_lambda = 0.1
        # -----------------------------------------------

        self.linears = nn.ModuleList([nn.Linear(input_dim * 3, diff_dim), nn.Linear(diff_dim, diff_dim), nn.Linear(diff_dim, input_dim)])

        self.step_emb_linear = nn.ModuleList([nn.Linear(diff_dim, input_dim)])

        self.cond_emb_linear = nn.ModuleList([nn.Linear(input_dim, input_dim)])

        self.num_layers = 1

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)

    def forward(self, x, t, cond_emb, cond_mask, zero_time=False, zero_cond=False):

        for idx in range(self.num_layers):

            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)

            if zero_time:
                t_embedding = torch.zeros_like(t_embedding)

            cond_embedding = self.cond_emb_linear[idx](cond_emb)

            if zero_cond:
                cond_embedding = torch.zeros_like(cond_embedding)

            # t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            # x = x + t_c_emb

            # -------------------------
            # concat
            # -------------------------
            x = torch.cat([x, t_embedding, cond_embedding], dim=1)

            x = self.linears[0](x)
            x = self.linears[1](x)
            x = self.linears[2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


# ---------------------------------------------------------
# loss
import torch.nn.functional as F


def q_x_fn(model, x_0, t, device):
    """
    x_0: [B, D]
    t:   [B, 1] or [B]
    """
    if t.dim() > 1:
        t = t.squeeze(-1)

    noise = torch.randn_like(x_0).to(device)

    a_bar_sqrt_t = extract(model.alphas_bar_sqrt.to(device), t, x_0.shape)
    one_minus_a_bar_sqrt_t = extract(model.one_minus_alphas_bar_sqrt.to(device), t, x_0.shape)

    x_t = a_bar_sqrt_t * x_0 + one_minus_a_bar_sqrt_t * noise
    return x_t, noise


def diffusion_loss_fn(model, x_0, cond_emb, iid_emb, y_input, device, is_task):

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:

        # ------------------------
        # sampling t
        # ------------------------
        batch_size = x_0.shape[0]

        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)

        # ------------------------
        # forward diffusion
        # ------------------------
        x_t, e = q_x_fn(model, x_0, t, device)

        # ------------------------
        # random mask
        # ------------------------
        cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
        cond_mask = 1 - cond_mask.int()

        # ------------------------
        # predict x0 directly
        # ------------------------
        x0_pred = model(x_t, t, cond_emb, cond_mask)

        # clamp는 선택
        # x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

        return F.smooth_l1_loss(x_0, x0_pred)

    elif is_task:
        # final_output_raw, iid_emb = p_sample_loop_x0(model, cond_emb, iid_emb, device, start_mode="cond")
        final_output_raw, iid_emb = p_sample_loop_x0_solver(
            model=model, cond_emb=cond_emb, iid_emb=iid_emb, device=device, start_mode="noise", sample_steps=20, eta=0.0
        )

        log_batch_similarity_stats(iid_emb, global_step=model.global_step, log_every=200, prefix="iid_emb")

        log_batch_similarity_stats(final_output_raw, global_step=model.global_step, log_every=200, prefix="final_output_raw")

        # final_output_proj = model.al_linear(final_output_raw)

        # log_batch_similarity_stats(final_output_proj, global_step=model.global_step, log_every=200, prefix="final_output_proj")

        # -------------------------------------------------
        # debug log
        # -------------------------------------------------
        # if model.global_step % 200 == 0:
        #     with torch.no_grad():
        #         target = y_input.squeeze().float()

        #         # -------------------------
        #         # norm stats
        #         # -------------------------
        #         x0_norm = x_0.norm(dim=1)
        #         raw_norm = final_output_raw.norm(dim=1)
        #         proj_norm = final_output_proj.norm(dim=1)
        #         iid_norm = iid_emb.norm(dim=1)

        #         print(f"\n[step {model.global_step}] ================= DEBUG =================")
        #         print(
        #             f"x_0 norm           | mean={x0_norm.mean().item():.4f}, "
        #             f"std={x0_norm.std().item():.4f}, "
        #             f"min={x0_norm.min().item():.4f}, "
        #             f"max={x0_norm.max().item():.4f}"
        #         )
        #         print(
        #             f"final_output_raw   | mean={raw_norm.mean().item():.4f}, "
        #             f"std={raw_norm.std().item():.4f}, "
        #             f"min={raw_norm.min().item():.4f}, "
        #             f"max={raw_norm.max().item():.4f}"
        #         )
        #         print(
        #             f"final_output_proj  | mean={proj_norm.mean().item():.4f}, "
        #             f"std={proj_norm.std().item():.4f}, "
        #             f"min={proj_norm.min().item():.4f}, "
        #             f"max={proj_norm.max().item():.4f}"
        #         )
        #         print(
        #             f"iid_emb norm       | mean={iid_norm.mean().item():.4f}, "
        #             f"std={iid_norm.std().item():.4f}, "
        #             f"min={iid_norm.min().item():.4f}, "
        #             f"max={iid_norm.max().item():.4f}"
        #         )

        #         # -------------------------
        #         # cosine similarity
        #         # -------------------------
        #         cos_x0_raw = F.cosine_similarity(x_0, final_output_raw, dim=1)
        #         cos_x0_proj = F.cosine_similarity(x_0, final_output_proj, dim=1)

        #         print(
        #             f"x0 vs raw cosine   | mean={cos_x0_raw.mean().item():.4f}, "
        #             f"std={cos_x0_raw.std().item():.4f}, "
        #             f"min={cos_x0_raw.min().item():.4f}, "
        #             f"max={cos_x0_raw.max().item():.4f}"
        #         )
        #         print(
        #             f"x0 vs proj cosine  | mean={cos_x0_proj.mean().item():.4f}, "
        #             f"std={cos_x0_proj.std().item():.4f}, "
        #             f"min={cos_x0_proj.min().item():.4f}, "
        #             f"max={cos_x0_proj.max().item():.4f}"
        #         )

        #         # -------------------------
        #         # prediction stats
        #         # -------------------------
        #         y_pred_x0 = torch.sum(x_0 * iid_emb, dim=1)
        #         y_pred_raw = torch.sum(final_output_raw * iid_emb, dim=1)
        #         y_pred_proj = torch.sum(final_output_proj * iid_emb, dim=1)

        #         print(
        #             f"target             | mean={target.mean().item():.4f}, "
        #             f"std={target.std().item():.4f}, "
        #             f"min={target.min().item():.4f}, "
        #             f"max={target.max().item():.4f}"
        #         )

        #         print(
        #             f"pred(x0, iid)      | mean={y_pred_x0.mean().item():.4f}, "
        #             f"std={y_pred_x0.std().item():.4f}, "
        #             f"min={y_pred_x0.min().item():.4f}, "
        #             f"max={y_pred_x0.max().item():.4f}"
        #         )
        #         print(
        #             f"pred(raw, iid)     | mean={y_pred_raw.mean().item():.4f}, "
        #             f"std={y_pred_raw.std().item():.4f}, "
        #             f"min={y_pred_raw.min().item():.4f}, "
        #             f"max={y_pred_raw.max().item():.4f}"
        #         )
        #         print(
        #             f"pred(proj, iid)    | mean={y_pred_proj.mean().item():.4f}, "
        #             f"std={y_pred_proj.std().item():.4f}, "
        #             f"min={y_pred_proj.min().item():.4f}, "
        #             f"max={y_pred_proj.max().item():.4f}"
        #         )

        #         # -------------------------
        #         # loss stats
        #         # -------------------------
        #         recon_raw = F.smooth_l1_loss(x_0, final_output_raw)
        #         recon_proj = F.smooth_l1_loss(x_0, final_output_proj)

        #         task_loss_x0 = (y_pred_x0 - target).square().mean()
        #         task_loss_raw = (y_pred_raw - target).square().mean()
        #         task_loss_proj = (y_pred_proj - target).square().mean()

        #         mae_x0 = torch.abs(y_pred_x0 - target).mean()
        #         mae_raw = torch.abs(y_pred_raw - target).mean()
        #         mae_proj = torch.abs(y_pred_proj - target).mean()

        #         print(f"recon raw loss     | {recon_raw.item():.6f}")
        #         print(f"recon proj loss    | {recon_proj.item():.6f}")
        #         print(f"task mse x0        | {task_loss_x0.item():.6f}")
        #         print(f"task mse raw       | {task_loss_raw.item():.6f}")
        #         print(f"task mse proj      | {task_loss_proj.item():.6f}")
        #         print(f"task mae x0        | {mae_x0.item():.6f}")
        #         print(f"task mae raw       | {mae_raw.item():.6f}")
        #         print(f"task mae proj      | {mae_proj.item():.6f}")
        #         print("===================================================\n")

        model.global_step += 1

        # uni_loss_proj = uniformity_loss(final_output_proj)
        uni_loss_proj = uniformity_loss(final_output_raw)

        y_pred = torch.sum(final_output_raw * iid_emb, dim=1)

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()

        return F.smooth_l1_loss(x_0, final_output_raw), model.task_lambda * task_loss, model.uni_lambda * uni_loss_proj


# generation fun
def p_sample(model, cond_emb, x, iid_emb, device):
    # wrap for dpm_solver
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    model_kwargs = {"cond_emb": cond_emb, "cond_mask": torch.zeros(cond_emb.size()[0], device=device)}

    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)

    sample = dpm_solver.sample(x, steps=dmp_sample_steps, eps=1e-4, adaptive_step_size=False, fast_version=True)

    # return model.get_al_emb(sample).to(device), iid_emb
    return sample.to(device), iid_emb


def p_sample_loop(model, cond_emb, iid_input, device):
    # source emb input
    # cur_x = cond_emb

    # noise input
    cur_x = torch.normal(0, 1, size=cond_emb.size(), device=device)

    # reversing
    cur_x, iid_emb_out = p_sample(model, cond_emb, cur_x, iid_input, device)

    return cur_x, iid_emb_out


import torch
import torch.nn.functional as F


@torch.no_grad()
def p_sample_naive_step(model, x_t, t, cond_emb, device):
    """
    x_t:      [B, D]
    t:        int
    cond_emb: [B, D]
    return:   x_{t-1}
    """

    bsz = x_t.size(0)

    # ----------------------------------
    # coefficients
    # ----------------------------------
    beta_t = model.betas[t].to(device)  # scalar
    alpha_t = model.alphas[t].to(device)  # scalar
    alpha_bar_t = model.alphas_prod[t].to(device)  # scalar
    alpha_bar_prev = model.alphas_prod_p[t].to(device)  # scalar

    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

    # posterior variance
    posterior_var_t = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    posterior_var_t = torch.clamp(posterior_var_t, min=1e-20)

    # ----------------------------------
    # predict epsilon
    # ----------------------------------
    t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
    cond_mask = torch.zeros(bsz, device=device)

    eps_theta = model(x_t, t_batch, cond_emb, cond_mask)

    # ----------------------------------
    # DDPM reverse mean
    # mu_theta(x_t, t)
    # = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
    # ----------------------------------
    model_mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta)

    # ----------------------------------
    # sample x_{t-1}
    # ----------------------------------
    if t > 0:
        noise = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_var_t) * noise
    else:
        x_prev = model_mean

    return x_prev


@torch.no_grad()
def p_sample_loop_naive(model, cond_emb, iid_input, device, start_mode="noise", x0_ref=None, log_every=0):

    if start_mode == "noise":
        cur_x = torch.randn_like(cond_emb)

    elif start_mode == "cond":  # "cond_noise"
        # cur_x = cond_emb + 0.1 * torch.randn_like(cond_emb)
        cur_x = cond_emb

    elif start_mode == "x0_forward":
        if x0_ref is None:
            raise ValueError("start_mode='x0_forward' requires x0_ref")

        t_full = torch.full((x0_ref.size(0), 1), fill_value=model.num_steps - 1, device=device, dtype=torch.long)
        cur_x, _ = q_x_fn(model, x0_ref, t_full, device)

    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    for t in reversed(range(model.num_steps)):
        cur_x = p_sample_naive_step(model, cur_x, t, cond_emb, device)

        if log_every > 0 and (t % log_every == 0 or t == model.num_steps - 1 or t == 0):
            norm = cur_x.norm(dim=1)
            # print(
            #     f"[naive reverse] t={t:03d} | "
            #     f"norm mean={norm.mean().item():.4f}, "
            #     f"std={norm.std().item():.4f}, "
            #     f"min={norm.min().item():.4f}, "
            #     f"max={norm.max().item():.4f}"
            # )

    return cur_x, iid_input


def uniformity_loss(z, t=2.0):
    z = F.normalize(z, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)


def p_sample_x0(model, x_t, t, cond_emb, device, cond_mask=None):
    """
    model output: x0_pred
    x_t: [B, D]
    t:   [B]
    """
    if cond_mask is None:
        cond_mask = torch.ones(x_t.shape[0], device=device, dtype=torch.int)

    betas_t = extract(model.betas.to(device), t, x_t.shape)
    alphas_t = extract(model.alphas.to(device), t, x_t.shape)
    alphas_bar_t = extract(model.alphas_prod.to(device), t, x_t.shape)
    alphas_bar_prev_t = extract(model.alphas_prod_p.to(device), t, x_t.shape)

    # ------------------------
    # predict x0
    # ------------------------
    x0_pred = model(x_t, t, cond_emb, cond_mask)

    # optional clamp
    # x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

    # ------------------------
    # posterior mean
    # mu = c1 * x0_pred + c2 * x_t
    # ------------------------
    coef1 = betas_t * torch.sqrt(alphas_bar_prev_t) / (1.0 - alphas_bar_t)
    coef2 = torch.sqrt(alphas_t) * (1.0 - alphas_bar_prev_t) / (1.0 - alphas_bar_t)
    mean = coef1 * x0_pred + coef2 * x_t

    # posterior variance
    var = betas_t * (1.0 - alphas_bar_prev_t) / (1.0 - alphas_bar_t)

    noise = torch.randn_like(x_t)

    # t == 0이면 noise 없이 mean 반환
    nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.dim() - 1)))
    x_prev = mean + nonzero_mask * torch.sqrt(var) * noise

    return x_prev, x0_pred


def p_sample_loop_x0(model, cond_emb, iid_emb, device, start_mode="noise"):
    """
    start_mode:
        - "noise": pure Gaussian에서 시작
        - "cond" : cond_emb에서 시작
    """
    batch_size = cond_emb.shape[0]

    if start_mode == "noise":
        x_t = torch.randn_like(cond_emb).to(device)
    elif start_mode == "cond":
        x_t = cond_emb.clone().to(device)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    cond_mask = torch.ones(batch_size, device=device, dtype=torch.int)

    final_x0_pred = None

    for time_step in reversed(range(model.num_steps)):
        t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
        x_t, x0_pred = p_sample_x0(model, x_t, t, cond_emb, device, cond_mask)
        final_x0_pred = x0_pred

    return final_x0_pred, iid_emb


import torch


def extract(a, t, x_shape):
    """
    a: [T]
    t: [B]
    return: [B, 1, ..., 1] broadcastable to x_shape
    """
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def predict_eps_from_x0(model, x_t, t, cond_emb, device, cond_mask=None):
    """
    model predicts x0, then convert to eps

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    => eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
    """
    if cond_mask is None:
        cond_mask = torch.ones(x_t.shape[0], device=device, dtype=torch.int)

    alpha_bar_t = extract(model.alphas_prod.to(device), t, x_t.shape)

    # x0_pred = model(x_t, t, cond_emb, cond_mask)
    x0_pred = model(x_t, t, cond_emb, cond_mask, zero_cond=True)
    # x0_pred = model(x_t, t, cond_emb, cond_mask, zero_time=True, zero_cond=True)

    eps_pred = (x_t - torch.sqrt(alpha_bar_t) * x0_pred) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)

    return x0_pred, eps_pred


@torch.no_grad()
def ddim_step_from_x0(
    model,
    x_t,
    t,
    t_prev,
    cond_emb,
    device,
    cond_mask=None,
    eta=0.0,
):
    """
    x0-prediction model + DDIM step

    eta = 0.0 이면 deterministic ODE-like sampling
    eta > 0.0 이면 stochastic DDIM
    """
    if cond_mask is None:
        cond_mask = torch.ones(x_t.shape[0], device=device, dtype=torch.int)

    x0_pred, eps_pred = predict_eps_from_x0(model, x_t, t, cond_emb, device, cond_mask)

    alpha_bar_t = extract(model.alphas_prod.to(device), t, x_t.shape)
    alpha_bar_prev = extract(model.alphas_prod.to(device), t_prev, x_t.shape)

    # DDIM sigma
    sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)

    noise = torch.randn_like(x_t)

    # direction term
    dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_pred

    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

    return x_prev, x0_pred


def make_ddim_timesteps(num_steps, sample_steps, device):
    """
    ex) num_steps=200, sample_steps=20
    -> [199, 188, 178, ..., 0]
    """
    if sample_steps > num_steps:
        raise ValueError("sample_steps must be <= num_steps")

    step_indices = torch.linspace(0, num_steps - 1, sample_steps, device=device).long()
    step_indices = torch.unique(step_indices)
    step_indices = torch.flip(step_indices, dims=[0])

    if step_indices[-1].item() != 0:
        step_indices = torch.cat([step_indices, torch.zeros(1, device=device, dtype=torch.long)], dim=0)

    return step_indices


@torch.no_grad()
def p_sample_loop_x0_solver(model, cond_emb, iid_emb, device, start_mode="noise", sample_steps=20, eta=0.0):
    """
    solver-style sampling for x0-prediction model

    start_mode:
        - "noise": pure Gaussian에서 시작
        - "cond" : cond_emb에서 시작

    sample_steps:
        전체 diffusion step(num_steps)보다 적게 두면 빠른 샘플링 가능

    eta:
        0.0 -> deterministic DDIM / ODE-like
        >0  -> stochastic DDIM
    """
    batch_size = cond_emb.shape[0]

    if start_mode == "noise":
        x_t = torch.randn_like(cond_emb).to(device)
    elif start_mode == "cond":
        x_t = cond_emb.clone().to(device)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    cond_mask = torch.ones(batch_size, device=device, dtype=torch.int)

    timesteps = make_ddim_timesteps(model.num_steps, sample_steps, device)

    final_x0_pred = None

    for i in range(len(timesteps) - 1):
        t = torch.full((batch_size,), timesteps[i].item(), device=device, dtype=torch.long)
        t_prev = torch.full((batch_size,), timesteps[i + 1].item(), device=device, dtype=torch.long)

        x_t, x0_pred = ddim_step_from_x0(
            model=model,
            x_t=x_t,
            t=t,
            t_prev=t_prev,
            cond_emb=cond_emb,
            device=device,
            cond_mask=cond_mask,
            eta=eta,
        )
        final_x0_pred = x0_pred

    # 마지막 t=0에서 한 번 더 x0 prediction 정리
    t0 = torch.zeros(batch_size, device=device, dtype=torch.long)
    # final_x0_pred = model(x_t, t0, cond_emb, cond_mask)
    final_x0_pred = model(x_t, t0, cond_emb, cond_mask, zero_cond=True)

    return final_x0_pred, iid_emb
