import torch
import torch.nn as nn

import math

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from VBGE import singleVBGE

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

        # time, condition, noised emb -> reverse 하는 3FC diffusion solver
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, diff_dim),
                nn.Linear(diff_dim, diff_dim),
                nn.Linear(diff_dim, input_dim),
            ]
        )

        # time embedding
        self.step_emb_linear = nn.ModuleList(
            [
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [
                nn.Linear(input_dim, input_dim),
            ]
        )

        self.num_layers = 1

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)

    def forward(self, x, t, cond_emb, cond_mask):

        for idx in range(self.num_layers):

            t_embedding = get_timestep_embedding(t, self.diff_dim)  # sin파 기반의 position embedding 얻고
            t_embedding = self.step_emb_linear[idx](t_embedding)  # linear 통과 -> time embedding

            cond_embedding = self.cond_emb_linear[idx](cond_emb)  # condition(user emb from src) -> linear 통과

            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb  # 세 가지를 모두 더해줌.
            # x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)

            x = self.linears[0](x)  # reverse -- 3 FC를 통해 denosing.
            x = self.linears[1](x)
            x = self.linears[2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


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
        vbge_opt=None,
        use_vbge=True,
        parallel=None,
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

        # VBGE or simple aggregator cache
        self.use_vbge = use_vbge
        self.s_gnn = singleVBGE(vbge_opt)  # source graph encoder
        self.t_gnn = singleVBGE(vbge_opt)  # target graph encoder
        self.vbge_user_emb_src = None
        self.vbge_user_emb_tgt = None

        # Parallel setting
        self.parallel = parallel

        # time, condition, noised emb -> reverse 하는 3FC diffusion solver
        self.diff_models = nn.ModuleList(
            [
                nn.ModuleList(
                    [  # diff model 1 -- MF condition
                        nn.Linear(input_dim, diff_dim),
                        nn.Linear(diff_dim, diff_dim),
                        nn.Linear(diff_dim, input_dim),
                    ]
                ),
                nn.ModuleList(
                    [  # diff model 2 -- VBGE condition
                        nn.Linear(input_dim, diff_dim),
                        nn.Linear(diff_dim, diff_dim),
                        nn.Linear(diff_dim, input_dim),
                    ]
                ),
            ]
        )
        # time embedding
        self.step_emb_linear = nn.ModuleList(
            [
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [
                nn.Linear(input_dim, input_dim),
            ]
        )

        self.num_layers = 1

        # linear for alm
        self.al_linear = nn.Linear(input_dim, input_dim, False)
        # self.al_linear = nn.Linear(input_dim*2,input_dim,False)

    def forward(self, x, t, cond_emb, cond_mask, diff_id):

        for idx in range(self.num_layers):
            t_embedding = get_timestep_embedding(t, self.diff_dim)  # sin파 기반의 position embedding 얻고
            t_embedding = self.step_emb_linear[idx](t_embedding)  # linear 통과 -> time embedding

            cond_embedding = self.cond_emb_linear[idx](cond_emb)  # condition(user emb from src) -> linear 통과

            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb  # 세 가지를 모두 더해줌.
            # x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)

            x = self.diff_models[diff_id][0](x)  # reverse -- 3 FC를 통해 denosing.
            x = self.diff_models[diff_id][1](x)
            x = self.diff_models[diff_id][2](x)

        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


# ---------------------------------------------------------
# loss
import torch.nn.functional as F


def q_x_fn(model, x_0, t, device):  # forward
    # eq(4)
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise  # x0에 노이즈를 더함.


def diffusion_loss_fn(model, x_0, cond_emb, iid_emb, y_input, device, is_task):  # DIM(reconstruction) loss

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:  # DIM loss 먼저

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0.shape[0]
        # sample t, timestep t를 랜덤하게 추출.
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)
        # x_t 생성 -- x_0에 노이즈 e 추가
        x, e = q_x_fn(model, x_0, t, device)

        # random mask
        cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
        cond_mask = 1 - cond_mask.int()

        # pred noise
        output = model(x, t.squeeze(-1), cond_emb, cond_mask)  # x_t, condition -> FC3 -> 노이즈 예측

        return F.smooth_l1_loss(e, output)  # 예측 노이즈와 실제 노이즈 비교 L1 loss

    elif is_task:  # task loss ALM 수행
        final_output, iid_emb = p_sample_loop(model, cond_emb, iid_emb, device)  # 디노이징 된 user emb / item emb
        y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 곱해서 예측

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()
        # RMSE
        # task_loss =   (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]

        return F.smooth_l1_loss(x_0, final_output) + model.task_lambda * task_loss  # ALM 로스 + task loss


def diffusion_loss_fn_parallel(model, x_0_m, x_0_g, cond_emb1, cond_emb2, iid_emb, y_input, device, is_task):  # DIM(reconstruction) loss

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:  # DIM loss 먼저

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0_m.shape[0]
        # sample t, timestep t를 랜덤하게 추출.
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        # noised x_t 생성 - MF feat(x_m), VBGE feat (x_g)
        # TODO 두 디퓨전 모델의 x_0를 다르게 줄 것인지 or condition만 다르게 줄 것인지.

        # x_0도 다르게
        if model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
            x_m, e_m = q_x_fn(model, x_0_m, t, device)
            x_g, e_g = x_m, e_m
        elif model.parallel["set_init"] == 1:  # 각각 MF, Aggr
            x_m, e_m = q_x_fn(model, x_0_m, t, device)
            x_g, e_g = q_x_fn(model, x_0_g, t, device)
        elif model.parallel["set_init"] == 2:  # x_0 둘다 MF + Aggr로
            x_m, e_m = q_x_fn(model, (x_0_m + x_0_g) / 2, t, device)
            x_g, e_g = x_m, e_m
        elif model.parallel["set_init"] == 3:  # x_0 둘다 Aggr로
            x_m, e_m = q_x_fn(model, x_0_g, t, device)
            x_g, e_g = x_m, e_m

        # random mask
        cond_mask1 = 1 * (torch.rand(cond_emb1.shape[0], device=device) <= mask_rate)
        cond_mask1 = 1 - cond_mask1.int()

        cond_mask2 = 1 * (torch.rand(cond_emb2.shape[0], device=device) <= mask_rate)
        cond_mask2 = 1 - cond_mask2.int()

        # pred noise
        if model.parallel["set_aggr"] == "aggonly":
            output2 = model(x_g, t.squeeze(-1), cond_emb2, cond_mask2, diff_id=1)  # x_t, c2 -> noise
            return F.smooth_l1_loss(e_g, output2)  # 예측 노이즈와 실제 노이즈 비교 L1 loss

        output1 = model(x_m, t.squeeze(-1), cond_emb1, cond_mask1, diff_id=0)  # x_t, c1 -> noise
        output2 = model(x_g, t.squeeze(-1), cond_emb2, cond_mask2, diff_id=1)  # x_t, c2 -> noise

        return F.smooth_l1_loss(e_m, output1) + F.smooth_l1_loss(e_g, output2)  # 예측 노이즈와 실제 노이즈 비교 L1 loss

    elif is_task:  # task loss ALM 수행

        if model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
            final_output_m, iid_emb = p_sample_loop_parallel(
                model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0
            )  # 디노이징 된 user emb_m / item emb
            final_output_g, iid_emb = p_sample_loop_parallel(
                model, cond_emb1, cond_emb2, iid_emb, device, diff_id=1
            )  # 디노이징 된 user emb_g / item emb
        elif model.parallel["set_init"] == 1:  # 각각 MF, Aggr
            final_output_m, iid_emb = p_sample_loop_parallel(
                model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0
            )  # 디노이징 된 user emb_m / item emb
            final_output_g, iid_emb = p_sample_loop_parallel(
                model, cond_emb2, cond_emb2, iid_emb, device, diff_id=1
            )  # 디노이징 된 user emb_g / item emb
        elif model.parallel["set_init"] == 2:  # x_0 둘다 MF + Aggr로
            start = (cond_emb1 + cond_emb2) / 2
            final_output_m, iid_emb = p_sample_loop_parallel(model, start, cond_emb1, iid_emb, device, diff_id=0)  # 디노이징 된 user emb_m / item emb
            final_output_g, iid_emb = p_sample_loop_parallel(model, start, cond_emb2, iid_emb, device, diff_id=1)  # 디노이징 된 user emb_g / item emb
        elif model.parallel["set_init"] == 3:  # Aggr만 사용
            # final_output_m, iid_emb=p_sample_loop_parallel(model,start, cond_emb1,iid_emb,device, diff_id=0) # 디노이징 된 user emb_m / item emb
            final_output_g, iid_emb = p_sample_loop_parallel(
                model, cond_emb2, cond_emb2, iid_emb, device, diff_id=1
            )  # 디노이징 된 user emb_g / item emb

        # final output = x'_c + x'_g

        if model.parallel["set_aggr"] == "avg":
            final_output = (final_output_m + final_output_g) / 2
        elif model.parallel["set_aggr"] == "concat":
            final_output = torch.cat([final_output_m, final_output_g], dim=1)
        elif model.parallel["set_aggr"] == "aggonly":
            final_output = final_output_g

        if model.parallel["set_proj"] == 1:
            final_output = model.get_al_emb(final_output).to(device)
        # TODO item 임베딩은 MF 임베딩을 공유?
        y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 곱해서 예측

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()
        # RMSE
        # task_loss =   (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]

        if model.parallel["set_loss"] == 0:
            return F.smooth_l1_loss(x_0_m, final_output) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 1:
            return F.smooth_l1_loss(x_0_g, final_output) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 2:
            return F.smooth_l1_loss((x_0_m + x_0_g) / 2, final_output) + model.task_lambda * task_loss
        elif model.parallel["set_loss"] == 3:
            return (
                F.smooth_l1_loss(x_0_m, final_output_m) + F.smooth_l1_loss(x_0_g, final_output_g) + model.task_lambda * task_loss
            )  # ALM 로스 + task loss


# generation fun
def p_sample(model, cond_emb, x, iid_emb, device):  # ALM + task loss
    # wrap for dpm_solver
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    model_kwargs = {
        "cond_emb": cond_emb,
        "cond_mask": torch.zeros(cond_emb.size()[0], device=device),
    }

    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)  # 노이즈, 노이즈 임베딩으로부터 denoised feat 예측 모델. 내부에서 forward 호출

    sample = dpm_solver.sample(  #  x_t-1 예측
        x,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )

    return model.get_al_emb(sample).to(device), iid_emb  # FC(x_t-1), item emb


def p_sample_loop(model, cond_emb, iid_input, device):
    # source emb input
    cur_x = cond_emb
    # noise input
    # cur_x = torch.normal(0,1,size = cond_emb.size() ,device=device)

    # reversing
    cur_x, iid_emb_out = p_sample(model, cond_emb, cur_x, iid_input, device)  # denoised embedding, item emb

    return cur_x, iid_emb_out


def p_sample_parallel(model, cond_emb, x, iid_emb, device, diff_id):  # ALM + task loss
    # wrap for dpm_solver
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    model_kwargs = {
        "cond_emb": cond_emb,
        "cond_mask": torch.zeros(cond_emb.size()[0], device=device),
        "diff_id": diff_id,  # DiffParallel.forword 처리 위해 diff id 인자 추가
    }

    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)  # 노이즈, 노이즈 임베딩으로부터 denoised feat 예측 모델. 내부에서 forward 호출

    sample = dpm_solver.sample(  #  x_t-1 예측
        x,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )

    if model.parallel["set_proj"] == 0:
        return model.get_al_emb(sample).to(device), iid_emb  # FC(x_t-1), item emb
    elif model.parallel["set_proj"] == 1:
        return sample, iid_emb


def p_sample_loop_parallel(model, start_emb, cond_emb, iid_input, device, diff_id):
    # source emb input
    cur_x = start_emb
    # noise input
    # cur_x = torch.normal(0,1,size = cond_emb.size() ,device=device)

    # reversing
    cur_x, iid_emb_out = p_sample_parallel(model, cond_emb, cur_x, iid_input, device, diff_id)  # denoised embedding, item emb

    return cur_x, iid_emb_out
