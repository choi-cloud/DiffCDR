import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dpm_solver_pytorch import model_wrapper, model_wrapper_hierarchical_cond, NoiseScheduleVP, DPM_Solver, hierarchical_cond_from_levels
from utils import AttentionLayer, SimilarityProjector

from rqvae import ResidualQuantizer

noise_schedule = NoiseScheduleVP(schedule="linear")


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
        # -----------------------------------------------

        self.linears = nn.ModuleList([nn.Linear(input_dim, diff_dim), nn.Linear(diff_dim, diff_dim), nn.Linear(diff_dim, input_dim)])

        self.step_emb_linear = nn.ModuleList([nn.Linear(diff_dim, input_dim)])

        self.cond_emb_linear = nn.ModuleList([nn.Linear(input_dim, input_dim)])

        self.num_layers = 1

        self.attn_layer = AttentionLayer(in_dim=input_dim, out_dim=input_dim)

        self.linear_m = nn.Linear(input_dim, input_dim, False)
        self.ln_iid = nn.LayerNorm(input_dim)
        self.ln_m = nn.LayerNorm(input_dim)

        self.style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.style_ln = nn.LayerNorm(input_dim)
        self.style_scale = nn.Parameter(torch.tensor(0.1))

        self.item_style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.item_style_ln = nn.LayerNorm(input_dim)
        self.item_style_scale = nn.Parameter(torch.tensor(0.1))

        self.tgt_global_bias = nn.Parameter(torch.tensor(0.0))

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)

    def forward(self, x, t, cond_emb, cond_mask):

        for idx in range(self.num_layers):

            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)

            cond_embedding = self.cond_emb_linear[idx](cond_emb)

            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb
            # x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)

            x = self.linears[0](x)
            x = self.linears[1](x)
            x = self.linears[2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffParallel(nn.Module):
    def __init__(
        self,
        num_steps=200,
        diff_dim=32,
        input_dim=32,
        c_scale=0.1,
        diff_sample_steps=30,
        diff_task_lambda=0.1,
        diff_mask_rate=0.1,
        parallel=None,
        rqvae=None,
    ):
        super(DiffParallel, self).__init__()

        # -------------------------------------------
        # define params
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.global_step = 0

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
        # -----------------------------------------------

        # Parallel setting
        self.parallel = parallel
        self.aggregation = parallel.get("aggregation", "aggregation")  # 'aggregation', 'aggregation_ab1', 'aggregation_ab2'

        # RQVAE setting
        self.rqvae = rqvae

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim * 2),
            nn.GELU(),
            nn.Linear(self.input_dim * 2, self.input_dim),
        )

        # time, condition, noised emb -> reverse 하는 3FC diffusion solver
        self.diff_models = nn.ModuleList()
        self.cond_emb_linear = nn.ModuleList()

        if self.aggregation in ["aggregation", "aggregation_ab1"]:
            self.diff_models.append(nn.ModuleList([nn.Linear(input_dim * 3, input_dim)]))
            self.cond_emb_linear.append(nn.Linear(input_dim, input_dim))
            # self.linear_m = nn.Linear(input_dim, input_dim, False)
            if self.parallel["batch_norm"]:
                self.ln_m = nn.BatchNorm1d(input_dim)

        if self.aggregation in ["aggregation", "aggregation_ab2"]:
            self.diff_models.append(nn.ModuleList([nn.Linear(input_dim * 3, input_dim)]))
            self.cond_emb_linear.append(nn.Linear(input_dim, input_dim))
            # self.linear_g = nn.Linear(input_dim, input_dim, False)
            if self.parallel["batch_norm"]:
                self.ln_g = nn.BatchNorm1d(input_dim)

        self.num_layers = 1
        if self.parallel["batch_norm"]:
            self.ln_iid = nn.BatchNorm1d(input_dim)

        self.attn_layer = AttentionLayer(in_dim=input_dim, out_dim=input_dim)

        if self.parallel["bias_mapping"] == "user":
            self.user_style_mapper = nn.Sequential(nn.Linear(2, 2))

        if self.parallel["set_aggr"] == "item_i":
            self.item_style_encoder = nn.Sequential(nn.Linear(2, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
            self.item_style_ln = nn.LayerNorm(input_dim)
            self.item_style_scale = nn.Parameter(torch.tensor(0.1))

        elif self.parallel["set_aggr"] == "item_iu":
            self.style_encoder = nn.Sequential(nn.Linear(2, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
            self.style_ln = nn.LayerNorm(input_dim)
            self.style_scale = nn.Parameter(torch.tensor(0.1))

            self.item_style_encoder = nn.Sequential(nn.Linear(2, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
            self.item_style_ln = nn.LayerNorm(input_dim)
            self.item_style_scale = nn.Parameter(torch.tensor(0.1))

        elif self.parallel["set_aggr"] == "item_u":
            self.style_encoder = nn.Sequential(nn.Linear(2, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
            self.style_ln = nn.LayerNorm(input_dim)
            self.style_scale = nn.Parameter(torch.tensor(0.1))

        elif self.parallel["set_aggr"] == "item":
            pass

        if self.rqvae["RQVAE"]:
            self.rq_mf = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])
            self.rq_aggr = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])

    def forward(
        self,
        x,
        t,
        cond_emb,
        cond_mask,
        diff_id=0,
        zero_cond=None,
    ):

        for idx in range(self.num_layers):
            t_embedding = self.step_mlp(t)

            cond_embedding = self.cond_emb_linear[diff_id](cond_emb)

            if zero_cond:
                cond_embedding = torch.zeros_like(cond_embedding)
            x = torch.cat([t_embedding, cond_embedding * cond_mask.unsqueeze(-1), x], axis=1)  # * cond_mask.unsqueeze(-1)

            x = self.diff_models[diff_id][0](x)  # reverse -- 3 FC를 통해 denosing.

        return x


def diffusion_loss_fn(model, x_0, cond_emb, iid_emb, y_input, device, is_task, style_src=None, uid=None, iid=None):

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0.shape[0]
        # sample t
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        x, e = q_x_fn(model, x_0, t, device)

        # random mask
        cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
        cond_mask = 1 - cond_mask.int()

        # pred noise
        output = model(x, t.squeeze(-1), cond_emb, cond_mask)

        return F.smooth_l1_loss(e, output)

    elif is_task:
        final_output, iid_emb = p_sample_loop(model, cond_emb, iid_emb, device)

        iid_emb = model.ln_iid(iid_emb)
        final_output_m = model.ln_m(model.linear_m(final_output))

        uid = uid.long()  # (B,)

        style_src = style_src.to(final_output_m.device)
        style_u = style_src[uid]  # (B, F)
        style_tok = model.style_encoder(style_u)  # (B, D)
        style_tok = model.style_ln(style_tok)  # (B, D)
        style_tok_u = model.style_scale * style_tok  # (B, D)

        style_tgt_item = model.style_tgt_item.to(final_output_m.device)  # [I_total, F_item]
        style_i = style_tgt_item[iid.squeeze(1)]  # (B, F_item)
        item_style_tok = model.item_style_encoder(style_i)  # (B, D)
        item_style_tok = model.item_style_ln(item_style_tok)  # (B, D)
        item_style_tok = model.item_style_scale * item_style_tok  # (B, D)

        tokens = torch.stack([iid_emb, final_output_m, style_tok_u, item_style_tok], dim=1)
        out = model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
        final_output = out[:, 0, :]  # (B, D)

        y_pred = torch.sum(final_output * iid_emb, dim=1)

        # domain bias
        mu_t = model.tgt_global_bias
        y_pred = y_pred + mu_t

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()
        # RMSE
        # task_loss =   (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]

        return F.smooth_l1_loss(x_0, final_output) + model.task_lambda * task_loss


def diffusion_loss_fn_parallel(
    model,
    x_0_m,
    x_0_g,
    cond_emb1,
    cond_emb2,
    iid_emb,
    y_input,
    device,
    is_task,
    q_embs1=None,
    q_embs2=None,
    style_src=None,
    uid=None,
    iid=None,
    Q_emb1=None,
    Q_emb2=None,
):

    num_steps = model.num_steps
    mask_rate = model.mask_rate
    if is_task == False:  # DIM loss 먼저

        batch_size = x_0_m.shape[0]

        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        if model.aggregation in ["aggregation", "aggregation_ab1"]:
            x_m, e_m = q_x_fn(model, x_0_m, t, device)
        if model.aggregation in ["aggregation", "aggregation_ab2"]:
            x_g, e_g = q_x_fn(model, x_0_g, t, device)

        # random mask
        cond_mask1 = 1 * (torch.rand(cond_emb1.shape[0], device=device) <= mask_rate)
        cond_mask1 = 1 - cond_mask1.int()

        cond_mask2 = 1 * (torch.rand(cond_emb2.shape[0], device=device) <= mask_rate)
        cond_mask2 = 1 - cond_mask2.int()

        if model.rqvae["RQVAE"] == True and q_embs1 is not None:
            ns = NoiseScheduleVP(schedule="linear")
            t_cont = t.squeeze(-1).float() / model.num_steps
            c1 = hierarchical_cond_from_levels(q_embs1, t_cont, ns)
            c2 = hierarchical_cond_from_levels(q_embs2, t_cont, ns) if q_embs2 is not None else cond_emb2
        else:
            c1, c2 = cond_emb1, cond_emb2

        if model.aggregation == "aggregation":
            output1 = model(x_m, t.squeeze(-1), c1, cond_mask1, diff_id=0)
            output2 = model(x_g, t.squeeze(-1), c2, cond_mask2, diff_id=1)
            return F.smooth_l1_loss(x_0_m, output1) + F.smooth_l1_loss(x_0_g, output2)
        elif model.aggregation == "aggregation_ab1":
            output1 = model(x_m, t.squeeze(-1), c1, cond_mask1, diff_id=0)
            return F.smooth_l1_loss(x_0_m, output1)
        elif model.aggregation == "aggregation_ab2":
            output1 = model(x_g, t.squeeze(-1), c2, cond_mask2, diff_id=0)
            return F.smooth_l1_loss(x_0_g, output1)

    elif is_task:
        if model.rqvae["start_point"] == "src_u":
            start1, start2 = cond_emb1, cond_emb2
        elif model.rqvae["start_point"] == "quant_u":
            start1, start2 = Q_emb1, Q_emb2
        elif model.rqvae["start_point"] == "noise":
            start1, start2 = torch.randn_like(cond_emb1), torch.randn_like(cond_emb2)

        if model.rqvae["RQVAE"] == True:
            cond1, cond2 = q_embs1, q_embs2
            p_sample = p_sample_loop_x0_solver
        else:
            cond1, cond2 = cond_emb1, cond_emb2
            p_sample = p_sample_loop_x0_solver

        # log_embedding_stats("item_raw", iid_emb, model.global_step)
        if model.parallel["batch_norm"]:
            iid_emb = model.ln_iid(iid_emb)
        # log_embedding_stats("item_norm", iid_emb, model.global_step)

        if model.aggregation == "aggregation":
            final_output_m, iid_emb = p_sample(model, cond1, iid_emb, device, diff_id=0)
            final_output_g, iid_emb = p_sample(model, cond2, iid_emb, device, diff_id=1)
            # -------------------------
            # Raw
            # -------------------------
            # log_embedding_stats("user_m_raw", final_output_m, model.global_step)
            # log_embedding_stats("user_g_raw", final_output_g, model.global_step)

            # final_output_m_proj = model.linear_m(final_output_m)
            # final_output_g_proj = model.linear_g(final_output_g)

            # -------------------------
            # Proj
            # -------------------------
            # log_embedding_stats("user_m_proj", final_output_m_proj, model.global_step)
            # log_embedding_stats("user_g_proj", final_output_g_proj, model.global_step)
            if model.parallel["batch_norm"]:
                final_output_m = model.ln_m(final_output_m)
                final_output_g = model.ln_g(final_output_g)

            # -------------------------
            # Norm
            # -------------------------
            # log_embedding_stats("user_m_norm", final_output_m, model.global_step)
            # log_embedding_stats("user_g_norm", final_output_g, model.global_step)

            base_tokens = torch.stack([final_output_m, final_output_g], dim=1)

            # uni_loss_m = uniformity_loss(final_output_m, t=2.0)
            # uni_loss_g = uniformity_loss(final_output_g, t=2.0)
            # uni_loss = uni_loss_m + uni_loss_g

        elif model.aggregation == "aggregation_ab1":
            # final_output_m, iid_emb = p_sample(model, start1, cond1, iid_emb, device, diff_id=0)
            final_output_m, iid_emb = p_sample(model, cond1, iid_emb, device, diff_id=0)
            # final_output_m_proj = model.linear_m(final_output_m)
            if model.parallel["batch_norm"]:
                final_output_m = model.ln_m(final_output_m)
            base_tokens = torch.stack([final_output_m], dim=1)

        elif model.aggregation == "aggregation_ab2":
            final_output_g, iid_emb = p_sample(model, start2, cond2, iid_emb, device, diff_id=0)
            final_output_g_proj = model.linear_g(final_output_g)
            final_output_g = model.ln_g(final_output_g_proj)
            base_tokens = torch.stack([final_output_g], dim=1)

        if model.parallel["set_aggr"] == "item":
            tokens = base_tokens

        elif model.parallel["set_aggr"] == "item_i":
            iid = iid.squeeze(1)

            style_tgt_item = model.style_tgt_item.to(base_tokens.device)  # [I_total, F_item]
            style_i = style_tgt_item[iid][:, :2]  # (B, F_item)
            item_style_tok = model.item_style_encoder(style_i)  # (B, D)
            item_style_tok = model.item_style_ln(item_style_tok)  # (B, D)
            item_style_tok = model.item_style_scale * item_style_tok  # (B, D)

            tokens = torch.cat([base_tokens, item_style_tok.unsqueeze(1)], dim=1)

        elif model.parallel["set_aggr"] == "item_iu":
            uid = uid.long()  # (B,)
            iid = iid.squeeze(1)

            style_src = style_src.to(base_tokens.device)
            style_u = style_src[uid][:, :2]  # (B, F)

            if model.parallel["bias_mapping"] == "user":
                style_u = model.user_style_mapper(style_u)
                mapping_loss = F.mse_loss(style_u, model.style_tgt_user[uid, :2])
                style_u = style_u.detach()

            style_tok = model.style_encoder(style_u)  # (B, D)
            style_tok = model.style_ln(style_tok)  # (B, D)
            style_tok_u = model.style_scale * style_tok  # (B, D)

            style_tgt_item = model.style_tgt_item.to(base_tokens.device)  # [I_total, F_item]
            style_i = style_tgt_item[iid][:, :2]  # (B, F_item)
            item_style_tok = model.item_style_encoder(style_i)  # (B, D)
            item_style_tok = model.item_style_ln(item_style_tok)  # (B, D)
            item_style_tok = model.item_style_scale * item_style_tok  # (B, D)

            tokens = torch.cat([base_tokens, style_tok_u.unsqueeze(1), item_style_tok.unsqueeze(1)], dim=1)

        elif model.parallel["set_aggr"] == "item_u":
            uid = uid.long()  # (B,)

            style_src = style_src.to(base_tokens.device)
            style_u = style_src[uid][:, :2]  # (B, F)

            if model.parallel["bias_mapping"] == "user":
                style_u = model.user_style_mapper(style_u)
                mapping_loss = F.mse_loss(style_u, model.style_tgt_user[uid, :2])
                style_u = style_u.detach()

            style_tok = model.style_encoder(style_u)  # (B, D)
            style_tok = model.style_ln(style_tok)  # (B, D)
            style_tok_u = model.style_scale * style_tok  # (B, D)

            tokens = torch.cat([base_tokens, style_tok_u.unsqueeze(1)], dim=1)

        elif model.parallel["set_aggr"] == "item":
            tokens = base_tokens

        out = model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
        final_output = out[:, 0, :]  # (B, D)

        uni_loss = uniformity_loss(final_output, t=2.0)  ###############0414 uniformity 실험을 위해 추가

        y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측

        model.global_step += 1

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()

        if model.parallel["set_aggr"] != "item":
            task_loss += model.parallel["mapping_lambda"] * mapping_loss

        if model.aggregation == "aggregation":
            return model.task_lambda * task_loss, model.parallel["uniformity_loss"] * uni_loss
        elif model.aggregation == "aggregation_ab1":
            return model.task_lambda * task_loss, model.parallel["uniformity_loss"] * uni_loss
        elif model.aggregation == "aggregation_ab2":
            return model.task_lambda * task_loss, model.parallel["uniformity_loss"] * uni_loss


def log_prediction_stats(name, pred, global_step, log_every=200):
    if global_step % log_every != 0:
        return

    with torch.no_grad():
        print(f"[{name}] pred")
        print(f" mean={pred.mean().item():.4f}, std={pred.std().item():.4f}, " f"min={pred.min().item():.4f}, max={pred.max().item():.4f}")
        print("")


def log_embedding_stats(name, emb, global_step, log_every=200):
    if global_step % log_every != 0:
        return

    with torch.no_grad():
        B, D = emb.shape

        # -----------------------------
        # norm stats
        # -----------------------------
        norm = emb.norm(dim=1)  # [B]
        norm_mean = norm.mean().item()
        norm_std = norm.std().item()
        norm_min = norm.min().item()
        norm_max = norm.max().item()

        # -----------------------------
        # per-sample mean/std (중요!!)
        # -----------------------------
        mean = emb.mean(dim=1)
        std = emb.std(dim=1)

        mean_mean = mean.mean().item()
        mean_std = mean.std().item()

        std_mean = std.mean().item()
        std_std = std.std().item()

        # -----------------------------
        # cosine similarity (batch 내)
        # -----------------------------
        emb_unit = F.normalize(emb, p=2, dim=1)
        sim_matrix = torch.matmul(emb_unit, emb_unit.t())  # [B, B]

        offdiag = sim_matrix[~torch.eye(B, dtype=bool, device=emb.device)]
        sim_mean = offdiag.mean().item()
        sim_std = offdiag.std().item()
        sim_min = offdiag.min().item()
        sim_max = offdiag.max().item()

        print(f"[{name}]")
        print(f" norm     | mean={norm_mean:.4f}, std={norm_std:.4f}, min={norm_min:.4f}, max={norm_max:.4f}")
        print(f" mean     | mean={mean_mean:.4f}, std={mean_std:.4f}")
        print(f" std      | mean={std_mean:.4f}, std={std_std:.4f}")
        print(f" cosine   | mean={sim_mean:.4f}, std={sim_std:.4f}, min={sim_min:.4f}, max={sim_max:.4f}")
        print("")


def uniformity_loss(z, t=2.0):
    z = F.normalize(z, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)


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


def extract(a, t, x_shape):
    """
    a: [T]
    t: [B]
    return: [B, 1, ..., 1] broadcastable to x_shape
    """
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def predict_eps_from_x0(model, x_t, t, cond_emb, device, cond_mask=None, diff_id=None):
    """
    model predicts x0, then convert to eps

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    => eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
    """
    if cond_mask is None:
        cond_mask = torch.ones(x_t.shape[0], device=device, dtype=torch.int)

    alpha_bar_t = extract(model.alphas_prod.to(device), t, x_t.shape)

    if model.parallel["zero_cond"]:
        x0_pred = model(x_t, t, cond_emb, cond_mask, diff_id=diff_id, zero_cond=True)
    else:
        x0_pred = model(x_t, t, cond_emb, cond_mask, diff_id=diff_id)

    eps_pred = (x_t - torch.sqrt(alpha_bar_t) * x0_pred) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)

    return x0_pred, eps_pred


@torch.no_grad()
def ddim_step_from_x0(model, x_t, t, t_prev, cond_emb, device, cond_mask=None, eta=0.0, diff_id=None):
    """
    x0-prediction model + DDIM step

    eta = 0.0 이면 deterministic ODE-like sampling
    eta > 0.0 이면 stochastic DDIM
    """
    if cond_mask is None:
        cond_mask = torch.ones(x_t.shape[0], device=device, dtype=torch.int)

    x0_pred, eps_pred = predict_eps_from_x0(model, x_t, t, cond_emb, device, cond_mask, diff_id=diff_id)

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
def p_sample_loop_x0_solver(model, cond_emb, iid_emb, device, start_mode="noise", sample_steps=20, eta=0.0, diff_id=0):
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
    if model.rqvae["RQVAE"]:  # cond_emb:  [L, B, D]
        q_embs = cond_emb  # [L, B, D] 원본 보존
        x_init = cond_emb[0]  # [B, D]
    else:  # cond_emb: [B, D]
        x_init = cond_emb

    batch_size = x_init.shape[0]

    if start_mode == "noise":
        x_t = torch.randn_like(x_init).to(device)  # [B, D]
    elif start_mode == "cond":
        x_t = x_init.clone().to(device)
    else:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    cond_mask = torch.ones(batch_size, device=device, dtype=torch.int)

    timesteps = make_ddim_timesteps(model.num_steps, sample_steps, device)

    final_x0_pred = None

    for i in range(len(timesteps) - 1):
        t = torch.full((batch_size,), timesteps[i].item(), device=device, dtype=torch.long)
        t_prev = torch.full((batch_size,), timesteps[i + 1].item(), device=device, dtype=torch.long)

        if model.rqvae["RQVAE"]:
            ns = NoiseScheduleVP(schedule="linear")
            t_cont = t.float() / model.num_steps
            cond_emb = hierarchical_cond_from_levels(q_embs, t_cont, ns)  # [L, B, D] -> [B, D]

        x_t, x0_pred = ddim_step_from_x0(
            model=model,
            x_t=x_t,
            t=t,
            t_prev=t_prev,
            cond_emb=cond_emb,
            device=device,
            cond_mask=cond_mask,
            eta=eta,
            diff_id=diff_id,
        )
        final_x0_pred = x0_pred

    # 마지막 t=0에서 한 번 더 x0 prediction 정리
    t0 = torch.zeros(batch_size, device=device, dtype=torch.long)

    if model.rqvae["RQVAE"]:
        ns = NoiseScheduleVP(schedule="linear")
        t_cont = t0.float() / model.num_steps
        cond_emb = hierarchical_cond_from_levels(q_embs, t_cont, ns)  # [L, B, D] -> [B, D]

    if model.parallel["zero_cond"]:
        final_x0_pred = model(x_t, t0, cond_emb, cond_mask, diff_id=diff_id, zero_cond=True)
    else:
        final_x0_pred = model(x_t, t0, cond_emb, cond_mask, diff_id=diff_id)

    return final_x0_pred, iid_emb
