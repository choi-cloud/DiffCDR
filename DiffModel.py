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
        self.test_users_degree = None
        self.test_users_pop_group = None
        self.degree_by_uid = None
        self.diff_step = 0
        self.task_step = 0
        # -----------------------------------------------

        # Parallel setting
        self.parallel = parallel
        self.aggregation = parallel.get("aggregation", "aggregation")  # 'aggregation', 'aggregation_ab1', 'aggregation_ab2'

        # RQVAE setting
        self.rqvae = rqvae

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.input_dim),
            # nn.Linear(self.input_dim, self.input_dim),
            # nn.GELU(),
            # nn.Linear(self.input_dim * 2, self.input_dim),
        )

        # time, condition, noised emb -> reverse 하는 3FC diffusion solver
        self.diff_models = nn.ModuleList()
        self.step_emb_linear = nn.ModuleList([nn.Linear(1, input_dim, bias=False)])

        self.cond_emb_linear = nn.ModuleList()
        # self.degree_encoder = nn.Sequential(nn.Linear(1, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        # self.degree_scale = nn.Parameter(torch.tensor(0.1))

        if self.aggregation in ["aggregation", "aggregation_ab1"]:
            self.diff_models.append(nn.ModuleList([nn.Linear(input_dim * 3, input_dim, bias=False)]))
            self.cond_emb_linear.append(nn.Linear(input_dim, input_dim, bias=False))
            self.mf_proj = nn.Linear(input_dim, input_dim)
            self.mf_norm = nn.LayerNorm(input_dim)

            if self.parallel["batch_norm"]:
                self.ln_m = nn.BatchNorm1d(input_dim)

            if self.aggregation == "aggregation":
                self.query_proj = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))

        if self.aggregation in ["aggregation", "aggregation_ab2"]:
            self.diff_models.append(nn.ModuleList([nn.Linear(input_dim * 3, input_dim, bias=False)]))
            self.cond_emb_linear.append(nn.Linear(input_dim, input_dim, bias=False))
            self.linear_g = nn.Linear(input_dim, input_dim, False)
            self.aggr_proj = nn.Linear(input_dim, input_dim)
            self.aggr_norm = nn.LayerNorm(input_dim)

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

            self.style_decoder = nn.Linear(input_dim, 2)
            self.item_style_decoder = nn.Linear(input_dim, 2)

        elif self.parallel["set_aggr"] == "item_u":
            self.style_encoder = nn.Sequential(nn.Linear(2, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
            self.style_ln = nn.LayerNorm(input_dim)
            self.style_scale = nn.Parameter(torch.tensor(0.1))

        elif self.parallel["set_aggr"] == "item":
            pass

        if self.rqvae["RQVAE"]:
            self.rq_mf = ResidualQuantizer(
                code_dim=input_dim,
                num_levels=rqvae["codebook_num"],
                codebook_size=rqvae["codebook_size"],
                level_loss_weights=[0.25, 1.0, 1.0, 1.0],
                recon_lambda=1.0,
            )
            self.rq_aggr = ResidualQuantizer(
                code_dim=input_dim,
                num_levels=rqvae["codebook_num"],
                codebook_size=rqvae["codebook_size"],
                level_loss_weights=[0.25, 1.0, 1.0, 1.0],
                recon_lambda=1.0,
            )

        self.cond_scale = 1.0

    def forward(self, x, t, cond_emb, cond_mask, diff_id=0, zero_cond=None):

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if cond_emb.dim() == 1:
            cond_emb = cond_emb.unsqueeze(0)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        if cond_mask is not None and cond_mask.dim() == 0:
            cond_mask = cond_mask.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(f"x must be [B,D], got {tuple(x.shape)}")

        if cond_emb.dim() != 2:
            raise ValueError(f"cond_emb must be [B,D], got {tuple(cond_emb.shape)}")

        if x.size(0) != cond_emb.size(0):
            raise ValueError(f"batch mismatch: x {tuple(x.shape)}, cond_emb {tuple(cond_emb.shape)}")

        # if zero_cond is True:
        #     cond_emb = torch.zeros_like(cond_emb)

        # if cond_mask is not None:
        #     cond_emb = cond_emb * cond_mask.float().view(-1, 1)

        for idx in range(self.num_layers):

            t_embedding = t.float().unsqueeze(-1) / self.num_steps
            t_embedding = self.step_emb_linear[idx](t_embedding)

            cond_embedding = self.cond_emb_linear[diff_id](cond_emb)

            if hasattr(self, "cond_scale"):
                cond_embedding = self.cond_scale * cond_embedding

            # cond_embedding = torch.zeros_like(cond_embedding)
            # perm = torch.randperm(cond_emb.size(0), device=cond_emb.device)
            # cond_emb = cond_emb[perm]

            x = torch.cat([x, t_embedding, cond_embedding], dim=1)

            x = self.diff_models[diff_id][idx](x)

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

        model.diff_step += 1

        batch_size = x_0_m.shape[0]

        # -------------------------------------------------
        # high-t biased sampling
        # cond_emb 의존도를 높이기 위해 noisy timestep을 더 많이 샘플링
        # -------------------------------------------------
        if model.aggregation in ["aggregation"]:
            high_ratio = 0.7
            high_start = int(num_steps * 0.6)

            num_high = int(batch_size * high_ratio)
            num_rand = batch_size - num_high

            t_high = torch.randint(low=high_start, high=num_steps, size=(num_high,), device=device)

            t_rand = torch.randint(low=0, high=num_steps, size=(num_rand,), device=device)

            t = torch.cat([t_high, t_rand], dim=0)

            t = t.unsqueeze(-1)

        if model.aggregation in ["aggregation_ab1", "aggregation_ab2"]:
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
            c1 = hierarchical_cond_from_levels(q_embs1, t_cont, ns, model.rqvae["rq_num"])
            c2 = hierarchical_cond_from_levels(q_embs2, t_cont, ns, model.rqvae["rq_num"]) if q_embs2 is not None else cond_emb2

        else:
            c1, c2 = cond_emb1, cond_emb2

        if model.aggregation == "aggregation":
            output1 = model(x_m, t.squeeze(-1), c1, cond_mask1, diff_id=0)
            output2 = model(x_g, t.squeeze(-1), c2, cond_mask2, diff_id=1)
            return F.smooth_l1_loss(x_0_m, output1) + F.smooth_l1_loss(x_0_g, output2)

        elif model.aggregation == "aggregation_ab1":
            output1 = model(x_m, t.squeeze(-1), c1, cond_mask1, diff_id=0)

            rec_loss = F.smooth_l1_loss(output1, x_0_m)

            cos_align_loss = 1.0 - F.cosine_similarity(output1, x_0_m, dim=1, eps=1e-8).mean()

            geo_loss = batch_cosine_matrix_loss(output1, x_0_m)

            norm_loss = sphere_match_norm_loss(output1, x_0_m)

            # -------------------------------------------------
            # shuffled condition negative
            # -------------------------------------------------
            perm = torch.randperm(c1.size(0), device=c1.device)
            c1_shuffle = c1[perm]

            output_neg = model(
                x_m,
                t.squeeze(-1),
                c1_shuffle,
                cond_mask1,
                diff_id=0,
            )

            pos_dist = 1.0 - F.cosine_similarity(output1, x_0_m, dim=1, eps=1e-8)

            neg_dist = 1.0 - F.cosine_similarity(output_neg, x_0_m, dim=1, eps=1e-8)

            margin_loss = F.relu(0.1 + pos_dist - neg_dist).mean()

            loss = rec_loss + 0.3 * cos_align_loss + 0.7 * geo_loss + 0.3 * norm_loss + 0.1 * margin_loss

            return loss

        elif model.aggregation == "aggregation_ab2":
            output1 = model(x_g, t.squeeze(-1), c2, cond_mask2, diff_id=0)
            return F.smooth_l1_loss(x_0_g, output1)

    elif is_task:
        model.task_step += 1

        if model.rqvae["RQVAE"] == True:
            cond1, cond2 = q_embs1, q_embs2
            p_sample = p_sample_loop_x0_solver
        else:
            cond1, cond2 = cond_emb1, cond_emb2
            p_sample = p_sample_loop_x0_solver

        if model.parallel["batch_norm"]:
            iid_emb = model.ln_iid(iid_emb)

        if model.parallel["set_aggr"] != "item_iu":
            if model.aggregation == "aggregation":
                query = model.query_proj(iid_emb).unsqueeze(1)
            else:
                query = iid_emb.unsqueeze(1)

        if model.aggregation == "aggregation":
            final_output_m, iid_emb = p_sample(model, cond1, iid_emb, device, diff_id=0)
            final_output_g, iid_emb = p_sample(model, cond2, iid_emb, device, diff_id=1)

            final_output_m = model.mf_proj(final_output_m)
            final_output_g = model.aggr_proj(final_output_g)

            if model.parallel["set_aggr"] == "item":
                final_output = (final_output_m + final_output_g) / 2
                uni_loss = uniformity_loss(final_output, t=2.0)
                y_pred = torch.sum(final_output * iid_emb, dim=1)
                task_loss = (y_pred - y_input.squeeze().float()).square().mean()
                return task_loss, model.parallel["uniformity_loss"] * uni_loss

            final_output_m = model.mf_norm(final_output_m)
            final_output_g = model.aggr_norm(final_output_g)

            if model.parallel["batch_norm"]:
                final_output_m = model.ln_m(final_output_m)
                final_output_g = model.ln_g(final_output_g)

            base_tokens = torch.stack([final_output_m, final_output_g], dim=1)

        elif model.aggregation == "aggregation_ab1":

            # -------------------------------------------------
            # raw condition
            # -------------------------------------------------
            out_raw, iid_emb = p_sample(model, cond_emb1, iid_emb, device, diff_id=0)  # raw cond

            # -------------------------------------------------
            # q1 condition
            # -------------------------------------------------
            # all_level_vectors가 [L, B, D]일 때만 q1 추출
            if q_embs1.dim() == 3:
                q1_cond = q_embs1[0]  # [B, D]
            else:
                raise ValueError(f"Expected q_embs1 [L,B,D], got {tuple(q_embs1.shape)}")

            out_q1, _ = p_sample(model, q1_cond, iid_emb, device, diff_id=0)

            if model.task_step % 500 == 0:

                print("\n" + "=" * 80)
                print(f"Raw Cond vs Q1 Cond Check @ step {model.task_step}")
                print("=" * 80)

                # 1. condition 자체 비교
                print_geometry("raw_cond", cond_emb1, task_step=model.task_step)
                print_geometry("q1_cond", q1_cond, task_step=model.task_step)
                print_geometry("raw_minus_q1", cond_emb1 - q1_cond, task_step=model.task_step)

                print_cross_cos("q1_cond vs raw_cond", q1_cond, cond_emb1, task_step=model.task_step)

                # 2. output 비교
                print_geometry("out_raw", out_raw, task_step=model.task_step)
                print_geometry("out_q1", out_q1, task_step=model.task_step)
                print_geometry("out_raw_minus_out_q1", out_raw - out_q1, task_step=model.task_step)

                print_cross_cos("out_raw vs out_q1", out_raw, out_q1, task_step=model.task_step)

                # 3. rating prediction 차이
                y_raw = torch.sum(out_raw * iid_emb, dim=1)
                y_q1 = torch.sum(out_q1 * iid_emb, dim=1)

                print("-" * 80)
                print(
                    f"[pred raw vs q1] "
                    f"diff_MAE={(y_raw - y_q1).abs().mean().item():.6f} | "
                    f"raw_mean={y_raw.mean().item():.6f} | "
                    f"raw_std={y_raw.std(unbiased=False).item():.6f} | "
                    f"q1_mean={y_q1.mean().item():.6f} | "
                    f"q1_std={y_q1.std(unbiased=False).item():.6f}"
                )

                rating_mae_rmse("raw cond", out_raw, iid_emb, y_input)
                rating_mae_rmse("q1 cond", out_q1, iid_emb, y_input)

                print("=" * 80 + "\n")

            # final_output_m = model.mf_norm(model.mf_proj(final_output_m))
            # final_output_m = model.mf_proj(final_output_m)

            final_output_m = out_raw
            y_pred = torch.sum(final_output_m * iid_emb, dim=1)
            task_loss = (y_pred - y_input.squeeze().float()).square().mean()
            uni_loss = uniformity_loss(final_output_m, t=2.0)

            return task_loss, model.parallel["uniformity_loss"] * uni_loss

        elif model.aggregation == "aggregation_ab2":
            final_output_g, iid_emb = p_sample(model, cond2, iid_emb, device, diff_id=0)

            # print_batch_node_similarity(emb=final_output_g, step=model.task_step, prefix="final_output_g", interval=200)

            # final_output_g = model.aggr_norm(model.aggr_proj(final_output_g))
            final_output_g = model.aggr_proj(final_output_g)

            y_pred = torch.sum(final_output_g * iid_emb, dim=1)
            task_loss = (y_pred - y_input.squeeze().float()).square().mean()
            uni_loss = uniformity_loss(final_output_g, t=2.0)

            return task_loss, model.parallel["uniformity_loss"] * uni_loss

        if model.parallel["set_aggr"] == "item_iu":
            uid = uid.long()  # (B,)
            iid = iid.squeeze(1)

            style_src = style_src.to(base_tokens.device)

            # --------------------------------------------------
            # user bias: raw -> normalize
            # --------------------------------------------------
            style_u_raw = style_src[uid][:, :2]  # (B, 2)

            style_u_mean = style_u_raw.mean(dim=0, keepdim=True)
            style_u_std = style_u_raw.std(dim=0, keepdim=True)
            style_u_norm = (style_u_raw - style_u_mean) / (style_u_std + 1e-8)

            if model.parallel["bias_mapping"] == "user":
                style_u = model.user_style_mapper(style_u_norm)

                target_u_raw = model.style_tgt_user[uid, :2].to(base_tokens.device)

                target_u_mean = target_u_raw.mean(dim=0, keepdim=True)
                target_u_std = target_u_raw.std(dim=0, keepdim=True)
                target_u_norm = (target_u_raw - target_u_mean) / (target_u_std + 1e-8)

                mapping_loss = F.mse_loss(style_u, target_u_norm)

                style_u = style_u.detach()
            else:
                style_u = style_u_norm

            # --------------------------------------------------
            # user encoder-decoder
            # --------------------------------------------------
            style_tok = model.style_encoder(style_u)  # (B, D)
            style_recon = model.style_decoder(style_tok)  # (B, 2)

            user_style_recon_loss = F.mse_loss(style_recon, style_u.detach())

            style_tok = model.style_ln(style_tok.detach())  # (B, D)
            style_tok_u = model.style_scale * style_tok  # (B, D)

            # --------------------------------------------------
            # item bias: raw -> normalize
            # --------------------------------------------------
            style_tgt_item = model.style_tgt_item.to(base_tokens.device)
            style_i_raw = style_tgt_item[iid][:, :2]  # (B, 2)

            style_i_mean = style_i_raw.mean(dim=0, keepdim=True)
            style_i_std = style_i_raw.std(dim=0, keepdim=True)
            style_i_norm = (style_i_raw - style_i_mean) / (style_i_std + 1e-8)

            # --------------------------------------------------
            # item encoder-decoder
            # --------------------------------------------------
            item_style_tok = model.item_style_encoder(style_i_norm)  # (B, D)
            item_style_recon = model.item_style_decoder(item_style_tok)  # (B, 2)

            item_style_recon_loss = F.mse_loss(item_style_recon, style_i_norm.detach())

            item_style_tok = model.item_style_ln(item_style_tok.detach())  # (B, D)
            item_style_tok = model.item_style_scale * item_style_tok  # (B, D)

            # --------------------------------------------------
            # total style reconstruction loss
            # --------------------------------------------------
            style_recon_loss = user_style_recon_loss + item_style_recon_loss

            # --------------------------------------------------
            # key/value tokens: MF, AGGR only
            # --------------------------------------------------
            tokens = base_tokens  # (B, 2, D)

            # --------------------------------------------------
            # query: encoded user bias + encoded item bias
            # --------------------------------------------------
            query_bias = style_tok_u + item_style_tok  # (B, D)

            query = model.query_proj(query_bias).unsqueeze(1)  # (B, 1, D)

        out, score, attn = model.attn_layer(tokens, query=query, return_score=True)  # (B, 1, D)

        final_output = out[:, 0, :]  # (B, D)

        # print_batch_node_similarity(emb=final_output, step=model.task_step, prefix="final_output", interval=200)

        uni_loss = uniformity_loss(final_output, t=2.0)

        if model.aggregation == "aggregation":
            y_pred = torch.sum(final_output * iid_emb, dim=1)

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()

        if model.aggregation == "aggregation":
            if model.parallel["set_aggr"] == "item_iu":
                return (
                    task_loss + model.parallel["mapping_lambda"] * mapping_loss + model.parallel["recon_loss"] * style_recon_loss,
                    model.parallel["uniformity_loss"] * uni_loss,
                )
            else:
                return task_loss, model.parallel["uniformity_loss"] * uni_loss


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

    # if model.rqvae["RQVAE"]:  # cond_emb:  [L, B, D]
    #     q_embs = cond_emb  # [L, B, D] 원본 보존
    #     x_init = cond_emb[0]  # [B, D]
    # else:  # cond_emb: [B, D]
    #     x_init = cond_emb

    # 올바른 처리
    if cond_emb.dim() == 3:
        # RQ: [L, B, D]
        x_init = cond_emb[0]  # q1, [B, D]
    elif cond_emb.dim() == 2:
        # raw or q1 condition: [B, D]
        x_init = cond_emb
    else:
        raise ValueError(f"bad cond_emb shape: {tuple(cond_emb.shape)}")

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
        #     cond_emb = hierarchical_cond_from_levels(q_embs, t_cont, ns, model.rqvae["rq_num"])  # [L, B, D] -> [B, D]

        if cond_emb.dim() == 3:
            # RQ condition: [L, B, D]
            q_embs = cond_emb
            cond_emb_step = hierarchical_cond_from_levels(q_embs, t_cont, ns, model.rqvae["rq_num"])  # [B, D]

        elif cond_emb.dim() == 2:
            # raw condition: [B, D]
            cond_emb_step = cond_emb

        else:
            raise ValueError(f"Unexpected cond_emb shape: {tuple(cond_emb.shape)}")

        x_t, x0_pred = ddim_step_from_x0(
            model=model, x_t=x_t, t=t, t_prev=t_prev, cond_emb=cond_emb_step, device=device, cond_mask=cond_mask, eta=eta, diff_id=diff_id
        )
        final_x0_pred = x_t

    t0 = torch.zeros(batch_size, device=device, dtype=torch.long)

    # if model.rqvae["RQVAE"]:
    #     ns = NoiseScheduleVP(schedule="linear")
    #     t_cont = t0.float() / model.num_steps
    #     cond_emb = hierarchical_cond_from_levels(q_embs, t_cont, ns, model.rqvae["rq_num"])  # [L, B, D] -> [B, D]

    # if model.parallel["zero_cond"]:
    #     final_x0_pred = model(x_t, t0, cond_emb, cond_mask, diff_id=diff_id, zero_cond=True)
    # else:
    #     final_x0_pred = model(x_t, t0, cond_emb, cond_mask, diff_id=diff_id)

    return final_x0_pred, iid_emb


@torch.no_grad()
def print_debug_metrics(step, x_0_m, x_0_g, final_output_m, final_output_g, iid_emb, y_input, interval=200):
    """
    200 step마다:
    - x_0_m / x_0_g 와 iid_emb로 예측한 MAE
    - final_output_m / final_output_g 와 iid_emb로 예측한 MAE
    - 배치 내 노드 간 cosine similarity 출력
    """

    if step % interval != 0:
        return

    y_true = y_input.squeeze().float()

    # 혹시 iid_emb가 (B, 1, D)면 (B, D)로 정리
    if iid_emb.dim() == 3:
        iid_emb_ = iid_emb.squeeze(1)
    else:
        iid_emb_ = iid_emb

    def pred_mae(user_emb, item_emb, y_true):
        if user_emb.dim() == 3:
            user_emb = user_emb[:, 0, :]  # diffusion output이 (B, token, D)인 경우

        y_pred = torch.sum(user_emb * item_emb, dim=-1)
        mae = torch.mean(torch.abs(y_pred - y_true))
        return mae.item()

    def batch_cos_sim(x):
        if x.dim() == 3:
            x = x[:, 0, :]

        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, x.t())  # (B, B)

        B = sim.size(0)
        if B <= 1:
            return 0.0

        # diagonal 제외 평균
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        return sim[mask].mean().item()

    # MAE
    mae_x0_m = pred_mae(x_0_m, iid_emb_, y_true)
    mae_x0_g = pred_mae(x_0_g, iid_emb_, y_true)

    mae_final_m = pred_mae(final_output_m, iid_emb_, y_true)
    mae_final_g = pred_mae(final_output_g, iid_emb_, y_true)

    # batch node similarity
    sim_x0_m = batch_cos_sim(x_0_m)
    sim_x0_g = batch_cos_sim(x_0_g)

    sim_final_m = batch_cos_sim(final_output_m)
    sim_final_g = batch_cos_sim(final_output_g)

    print(
        f"[Step {step}] "
        f"x0_m MAE: {mae_x0_m:.4f} | "
        f"x0_g MAE: {mae_x0_g:.4f} | "
        f"final_m MAE: {mae_final_m:.4f} | "
        f"final_g MAE: {mae_final_g:.4f} || "
        f"sim x0_m: {sim_x0_m:.4f} | "
        f"sim x0_g: {sim_x0_g:.4f} | "
        f"sim final_m: {sim_final_m:.4f} | "
        f"sim final_g: {sim_final_g:.4f}"
    )


def print_batch_node_similarity(emb, step, prefix="Embedding", interval=200):
    """
    emb    : [B, D]
    step   : 현재 step, 예: model.task_step
    prefix : 출력 이름
    """
    if step % interval != 0:
        return

    with torch.no_grad():
        normed = F.normalize(emb, dim=1)
        sim_matrix = torch.matmul(normed, normed.t())  # [B, B]

        batch_size = sim_matrix.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

        sims = sim_matrix[mask]

        print(f"\n[Step {step}] {prefix} Similarity")
        print(f"Mean : {sims.mean().item():.6f}")
        print(f"Std  : {sims.std().item():.6f}")
        print(f"Min  : {sims.min().item():.6f}")
        print(f"Max  : {sims.max().item():.6f}")


def log_embedding_geometry(name, emb, target_emb=None, step=None):
    """
    emb: [B, D]
    target_emb: [B, D] or None
    """

    with torch.no_grad():

        emb = emb.detach()

        # -------------------------------------------------
        # node-node cosine
        # -------------------------------------------------
        emb_normed = F.normalize(emb, p=2, dim=1)

        sim_mat = emb_normed @ emb_normed.t()

        B = sim_mat.size(0)

        mask = ~torch.eye(B, dtype=torch.bool, device=emb.device)

        off_diag_sim = sim_mat[mask]

        # -------------------------------------------------
        # norm
        # -------------------------------------------------
        emb_norm = emb.norm(dim=1)

        print("\n" + "=" * 60)
        print(f"{name} Geometry Check")
        print("=" * 60)

        if step is not None:
            print(f"step : {step}")

        # -------------------------------------------------
        # target cosine
        # -------------------------------------------------
        if target_emb is not None:

            target_emb = target_emb.detach()

            cos_sim = F.cosine_similarity(emb, target_emb, dim=1)

            target_norm = target_emb.norm(dim=1)

            print(
                f"cosine({name}, target) | "
                f"mean: {cos_sim.mean().item():.6f} | "
                f"std : {cos_sim.std().item():.6f} | "
                f"min : {cos_sim.min().item():.6f} | "
                f"max : {cos_sim.max().item():.6f}"
            )

            print(
                "target norm | "
                f"mean: {target_norm.mean().item():.6f} | "
                f"std : {target_norm.std().item():.6f} | "
                f"min : {target_norm.min().item():.6f} | "
                f"max : {target_norm.max().item():.6f}"
            )

        # -------------------------------------------------
        # node-node cosine
        # -------------------------------------------------
        print(
            f"{name} node-node cosine | "
            f"mean: {off_diag_sim.mean().item():.6f} | "
            f"std : {off_diag_sim.std().item():.6f} | "
            f"min : {off_diag_sim.min().item():.6f} | "
            f"max : {off_diag_sim.max().item():.6f}"
        )

        # -------------------------------------------------
        # emb norm
        # -------------------------------------------------
        print(
            f"{name} norm | "
            f"mean: {emb_norm.mean().item():.6f} | "
            f"std : {emb_norm.std().item():.6f} | "
            f"min : {emb_norm.min().item():.6f} | "
            f"max : {emb_norm.max().item():.6f}"
        )

        print("=" * 60)


def batch_cosine_matrix_loss(pred, target):
    pred_n = F.normalize(pred, dim=1, eps=1e-8)
    target_n = F.normalize(target, dim=1, eps=1e-8)

    pred_sim = pred_n @ pred_n.t()
    target_sim = target_n @ target_n.t()

    bsz = pred.size(0)
    if bsz <= 1:
        return pred.new_tensor(0.0)

    mask = ~torch.eye(bsz, dtype=torch.bool, device=pred.device)

    return F.smooth_l1_loss(pred_sim[mask], target_sim[mask])


def sphere_match_norm_loss(pred, target):
    pred_norm = pred.norm(dim=1)
    target_norm = target.norm(dim=1)
    return F.smooth_l1_loss(pred_norm, target_norm)


@torch.no_grad()
def rating_mae_rmse(name, out, iid_emb, y_true):
    y_true = y_true.view(-1).to(out.device).float()
    y_pred = torch.sum(out * iid_emb, dim=1)

    mae = torch.mean(torch.abs(y_pred - y_true))
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))

    print(
        f"[{name}] "
        f"MAE={mae.item():.6f} | "
        f"RMSE={rmse.item():.6f} | "
        f"pred_mean={y_pred.mean().item():.6f} | "
        f"pred_std={y_pred.std(unbiased=False).item():.6f} | "
        f"true_mean={y_true.mean().item():.6f}"
    )


@torch.no_grad()
def print_geometry(name, x, task_step=None):
    """
    x: [B, D]
    batch 내 node-node cosine과 norm 통계 출력
    """
    if x is None:
        return

    x = x.detach()

    if x.dim() != 2:
        print(f"[{name}] skip: expected [B, D], got {tuple(x.shape)}")
        return

    bsz = x.size(0)
    norm = x.norm(dim=1)

    x_n = F.normalize(x, dim=1, eps=1e-8)
    sim = x_n @ x_n.t()

    if bsz > 1:
        mask = ~torch.eye(bsz, dtype=torch.bool, device=x.device)
        vals = sim[mask]
    else:
        vals = sim.reshape(-1)

    step_str = f" @ task_step {task_step}" if task_step is not None else ""

    print("-" * 80)
    print(f"[Geometry] {name}{step_str}")
    print(
        f"node-node cosine | "
        f"mean={vals.mean().item():.6f} | "
        f"std={vals.std(unbiased=False).item():.6f} | "
        f"min={vals.min().item():.6f} | "
        f"max={vals.max().item():.6f}"
    )
    print(
        f"norm             | "
        f"mean={norm.mean().item():.6f} | "
        f"std={norm.std(unbiased=False).item():.6f} | "
        f"min={norm.min().item():.6f} | "
        f"max={norm.max().item():.6f}"
    )


@torch.no_grad()
def print_cross_cos(name, a, b, task_step=None):
    """
    a, b: [B, D]
    같은 index끼리 cosine similarity 출력
    """
    if a is None or b is None:
        return

    a = a.detach()
    b = b.detach()

    if a.dim() != 2 or b.dim() != 2:
        print(f"[{name}] skip: expected [B, D], got {tuple(a.shape)} and {tuple(b.shape)}")
        return

    if a.shape != b.shape:
        print(f"[{name}] skip: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return

    cos = F.cosine_similarity(a, b, dim=1, eps=1e-8)

    step_str = f" @ task_step {task_step}" if task_step is not None else ""

    print("-" * 80)
    print(f"[Cross Cos] {name}{step_str}")
    print(
        f"mean={cos.mean().item():.6f} | "
        f"std={cos.std(unbiased=False).item():.6f} | "
        f"min={cos.min().item():.6f} | "
        f"max={cos.max().item():.6f}"
    )
