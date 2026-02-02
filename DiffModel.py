import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dpm_solver_pytorch import model_wrapper, model_wrapper_hierarchical_cond, NoiseScheduleVP, DPM_Solver
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

        # RQVAE setting
        self.rqvae = rqvae

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim * 2),
            nn.GELU(),
            nn.Linear(self.input_dim * 2, self.input_dim),
        )

        # time, condition, noised emb -> reverse 하는 3FC diffusion solver
        self.diff_models = nn.ModuleList(
            [
                nn.ModuleList(
                    [  # diff model 1 -- MF condition
                        nn.Linear(input_dim * 3, input_dim),
                        # nn.Linear(diff_dim, diff_dim),
                        # nn.Linear(diff_dim, input_dim),
                    ]
                ),
                nn.ModuleList(
                    [  # diff model 2 -- Aggr condition
                        nn.Linear(input_dim * 3, input_dim),
                        # nn.Linear(diff_dim, diff_dim),
                        # nn.Linear(diff_dim, input_dim),
                    ]
                ),
            ]
        )
        # time embedding
        # self.step_emb_linear = nn.ModuleList([nn.Linear(diff_dim, input_dim)])

        self.cond_emb_linear = nn.ModuleList([nn.Linear(input_dim, input_dim)])

        self.num_layers = 1

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)

        self.linear_m = nn.Linear(input_dim, input_dim, False)
        self.linear_g = nn.Linear(input_dim, input_dim, False)

        self.ln_iid = nn.LayerNorm(input_dim)
        self.ln_m = nn.LayerNorm(input_dim)
        self.ln_g = nn.LayerNorm(input_dim)

        if self.parallel["set_aggr"] in ["attn", "item_attn"]:
            self.attn_layer = AttentionLayer(in_dim=input_dim * 2, out_dim=input_dim)
        elif self.parallel["set_aggr"] == "item_cls":
            self.attn_layer = AttentionLayer(in_dim=input_dim, out_dim=input_dim)

        self.rq_mf = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])
        self.rq_aggr = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])

        self.style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.style_ln = nn.LayerNorm(input_dim)
        self.style_scale = nn.Parameter(torch.tensor(0.1))

        self.item_style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.item_style_ln = nn.LayerNorm(input_dim)
        self.item_style_scale = nn.Parameter(torch.tensor(0.1))

        self.tgt_global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t, cond_emb, cond_mask, diff_id):

        for idx in range(self.num_layers):
            # t_embedding = get_timestep_embedding(t, self.diff_dim)  # sin파 기반의 position embedding 얻고
            # t_embedding = self.step_emb_linear[idx](t_embedding)  # linear 통과 -> time embedding
            t_embedding = self.step_mlp(t)

            cond_embedding = self.cond_emb_linear[idx](cond_emb)  # condition(user emb from src) -> linear 통과

            x = torch.cat([t_embedding, cond_embedding * cond_mask.unsqueeze(-1), x], axis=1)  # * cond_mask.unsqueeze(-1)

            x = self.diff_models[diff_id][0](x)  # reverse -- 3 FC를 통해 denosing.
            # x = self.diff_models[diff_id][1](x)
            # x = self.diff_models[diff_id][2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


def q_x_fn(model, x_0, t, device):  # forward
    # eq(4)
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise  # x0에 노이즈를 더함.


def diffusion_loss_fn_parallel(
    model, x_0_m, x_0_g, cond_emb1, cond_emb2, iid_emb, y_input, device, is_task, q_embs1=None, q_embs2=None, style_src=None, uid=None, iid=None
):

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:  # DIM loss 먼저

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0_m.shape[0]

        ### [TRAIN-DIM] 1. sample t, timestep t를 랜덤하게 추출.
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        ### [TRAIN-DIM] 2. Diff1, Diff2 noised x_0, noise (e) 생성
        if model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
            x_m, e_m = q_x_fn(model, x_0_m, t, device)
            x_g, e_g = x_m, e_m
        elif model.parallel["set_init"] == 1:  # 각각 MF, Aggr
            x_m, e_m = q_x_fn(model, x_0_m, t, device)
            x_g, e_g = q_x_fn(model, x_0_g, t, device)

        # random mask
        cond_mask1 = 1 * (torch.rand(cond_emb1.shape[0], device=device) <= mask_rate)
        cond_mask1 = 1 - cond_mask1.int()

        cond_mask2 = 1 * (torch.rand(cond_emb2.shape[0], device=device) <= mask_rate)
        cond_mask2 = 1 - cond_mask2.int()

        # [TRAIN-DIM] 3. Diff1, Diff2 -> noise 예측
        output1 = model(x_m, t.squeeze(-1), cond_emb1, cond_mask1, diff_id=0)  # x_t, c1 -> noise
        output2 = model(x_g, t.squeeze(-1), cond_emb2, cond_mask2, diff_id=1)  # x_t, c2 -> noise

        return F.mse_loss(x_m, output1) + F.mse_loss(x_g, output2)  # 예측 노이즈와 실제 노이즈 비교 L1 loss

    elif is_task:  # task loss ALM 수행

        ### [TRAIN-ALM] 1. noised x_0 설정에 따라 denoising
        if model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
            final_output_m, iid_emb = p_sample_loop_parallel(model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0)
            final_output_g, iid_emb = p_sample_loop_parallel(model, cond_emb1, cond_emb2, iid_emb, device, diff_id=1)
        elif model.parallel["set_init"] == 1:  # 각각 MF, Aggr
            # ! 각각 MF, Aggr인 파트만 수정
            final_output_m, iid_emb = p_sample_loop_parallel(model, cond_emb1, q_embs1, iid_emb, device, diff_id=0)
            final_output_g, iid_emb = p_sample_loop_parallel(model, cond_emb2, q_embs2, iid_emb, device, diff_id=1)

        ### [TRAIN-ALM] 2. Diff1, Diff2 결과 aggregation
        if model.parallel["set_aggr"] == "attn":
            # ! 어텐션으로 최종 임베딩 종합
            final_output = model.attn_layer(torch.cat([final_output_m, final_output_g], dim=1))

        elif model.parallel["set_aggr"] == "item_attn":
            # 아이템을 쿼리로 사용
            final_output = model.attn_layer(torch.cat([final_output_m, final_output_g], dim=1), query=torch.cat([iid_emb, iid_emb], dim=1))

        elif model.parallel["set_aggr"] == "item_cls":
            iid_emb = model.ln_iid(iid_emb)
            final_output_m = model.ln_m(model.linear_m(final_output_m))
            final_output_g = model.ln_g(model.linear_g(final_output_g))

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

            tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u, item_style_tok], dim=1)
            out = model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
            final_output = out[:, 0, :]  # (B, D)

        y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
        mu_t = model.tgt_global_bias
        y_pred = y_pred + mu_t

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()

        # RMSE
        # task_loss =   (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]

        if model.parallel["set_loss"] == 0:
            # ! mf 임베딩과 유사해지도록 통일
            # return F.smooth_l1_loss(x_0_m, final_output) + model.task_lambda * task_loss
            return F.mse_loss(x_0_m, final_output_m) + F.mse_loss(x_0_g, final_output_g) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 1:
            return F.mse_loss(x_0_g, final_output) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 2:
            return F.mse_loss((x_0_m + x_0_g) / 2, final_output) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 3:
            return F.mse_loss(x_0_m, final_output_m) + F.mse_loss(x_0_g, final_output_g) + model.task_lambda * task_loss  # ALM 로스 + task loss


def _get_ddpm_sampler(model, device):
    """
    model.num_steps, model.betas를 그대로 사용해서 diffusion sampler를 구성.
    w는 0으로 둬서 guidance 끔 (원하면 model.parallel['w'] 같은 걸로 바꿔도 됨)
    """

    class _Diffusion:
        def __init__(self, betas, w, device):
            self.device = device
            self.w = w
            self.betas = betas.to(device)

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

            self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
            self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        def _extract(self, a, t, x_shape):
            # t: (B,) long
            out = a.gather(0, t)
            return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

        @torch.no_grad()
        def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
            # x_start = predicted x0
            x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)

            model_mean = self._extract(self.posterior_mean_coef1, t, x.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x.shape) * x

            if t_index == 0:
                return model_mean
            noise = torch.randn_like(x)
            var = self._extract(self.posterior_variance, t, x.shape)
            return model_mean + torch.sqrt(var) * noise

        @torch.no_grad()
        def sample(self, model_forward, model_forward_uncon, h, x_t):
            x = x_t
            T = self.betas.shape[0]

            step_indices = torch.linspace(T - 1, 0, h.shape[0], device=self.device).long()

            for i, n in enumerate(step_indices):
                t = torch.full((x.shape[0],), n, device=self.device, dtype=torch.long)
                h_i = h[i] if h.dim() == 3 else h
                x = self.p_sample(model_forward, model_forward_uncon, x, h_i, t, i)

            return x

    if hasattr(model, "_ddpm_sampler") and model._ddpm_sampler is not None:
        return model._ddpm_sampler

    model._ddpm_sampler = _Diffusion(model.betas, w=0.0, device=device)
    return model._ddpm_sampler


def p_sample_parallel(model, cond_emb, x, iid_emb, device, diff_id):
    B = x.shape[0]
    cond_mask = torch.ones(B, device=device)

    sampler = _get_ddpm_sampler(model, device)
    total_T = model.num_steps
    sample_steps = model.sample_steps  # = 30

    use_hier = cond_emb.dim() == 3  # (L,B,D)

    # ----------------------------
    # ✅ hier_conds 미리 계산: (S,B,D)
    # ----------------------------
    if use_hier:
        # cond_emb: (L,B,D)
        L = cond_emb.shape[0]
        cond_cumsum = torch.cumsum(cond_emb, dim=0)  # (L,B,D)

        # sampler loop와 동일한 step index 만들기
        step_indices = torch.linspace(total_T - 1, 0, sample_steps, device=device).long()  # (S,)

        # timestep -> level id 매핑 (coarse->fine)
        progress = 1.0 - step_indices.float() / (total_T - 1)  # 0->1
        level_ids = torch.clamp((progress * (L - 1)).long(), 0, L - 1)  # (S,)

        hier_conds = cond_cumsum[level_ids]  # (S,B,D)
    else:
        hier_conds = cond_emb  # (B,D)

    # ----------------------------
    # ✅ sampler에 h로 넣고, model_forward는 h_를 그대로 cond로 사용
    # ----------------------------
    x0 = sampler.sample(
        model_forward=lambda x_, h_, t_: model.forward(x_, t_, h_, cond_mask, diff_id=diff_id),  # step마다 바뀌는 condition
        model_forward_uncon=lambda x_, t_: model.forward(x_, t_, torch.zeros_like(x_), torch.zeros(B, device=device), diff_id=diff_id),
        h=hier_conds,  # (S,B,D) or (B,D)
        x_t=x,
    )

    return x0.to(device), iid_emb


def p_sample_loop_parallel(model, start_emb, cond_emb, iid_input, device, diff_id):
    cur_x, iid_emb_out = p_sample_parallel(model=model, cond_emb=cond_emb, x=start_emb, iid_emb=iid_input, device=device, diff_id=diff_id)
    return cur_x, iid_emb_out
