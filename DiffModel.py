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
    def __init__(self,num_steps=200, diff_dim=32,input_dim =32,c_scale=0.1,diff_sample_steps=30,diff_task_lambda=0.1,diff_mask_rate=0.1 ):
        super(DiffCDR,self).__init__()

        #-------------------------------------------
        #define params
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4,0.02 ,num_steps)

        self.alphas = 1-self.betas
        self.alphas_prod = torch.cumprod(self.alphas,0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(),self.alphas_prod[:-1]],0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape==self.alphas_prod.shape==self.alphas_prod_p.shape==\
        self.alphas_bar_sqrt.shape==self.one_minus_alphas_bar_log.shape\
        ==self.one_minus_alphas_bar_sqrt.shape

        #-----------------------------------------------
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate
        #-----------------------------------------------
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim,diff_dim),    
                nn.Linear(diff_dim,diff_dim) ,     
                nn.Linear(diff_dim,input_dim),  
            ]
        )
        
        self.step_emb_linear = nn.ModuleList(
            [   
                nn.Linear(diff_dim,input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [   
                nn.Linear(input_dim,input_dim),
            ]
        ) 

        self.num_layers = 1

        self.attn_layer = AttentionLayer(in_dim=input_dim, out_dim=input_dim)

        self.linear_m = nn.Linear(input_dim, input_dim, False)
        self.ln_iid = nn.LayerNorm(input_dim)
        self.ln_m   = nn.LayerNorm(input_dim)

        self.style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.style_ln = nn.LayerNorm(input_dim)
        self.style_scale = nn.Parameter(torch.tensor(0.1))

        self.item_style_encoder = nn.Sequential(nn.Linear(9, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim))
        self.item_style_ln = nn.LayerNorm(input_dim)
        self.item_style_scale = nn.Parameter(torch.tensor(0.1))
        
        self.tgt_global_bias = nn.Parameter(torch.tensor(0.0))

        #linear for alm 
        self.al_linear = nn.Linear(input_dim,input_dim,False)

    def forward(self, x,t, cond_emb,cond_mask ):

        for idx in range( self.num_layers ):
        
            t_embedding = get_timestep_embedding( t , self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)
        
            cond_embedding = self.cond_emb_linear[idx](cond_emb)
        
            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb
            #x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)

            x = self.linears[0](x) 
            x = self.linears[1](x) 
            x = self.linears[2](x) 

        return x
        
    def get_al_emb(self,emb):
        return self.al_linear (emb)

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
        self.aggregation = parallel.get("aggregation", "aggregation") # 'aggregation', 'aggregation_ab1', 'aggregation_ab2'
        
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
            self.linear_m = nn.Linear(input_dim, input_dim, False)
            self.ln_m = nn.LayerNorm(input_dim)

        if self.aggregation in ["aggregation", "aggregation_ab2"]:
            self.diff_models.append(nn.ModuleList([nn.Linear(input_dim * 3, input_dim)]))
            self.cond_emb_linear.append(nn.Linear(input_dim, input_dim))
            self.linear_g = nn.Linear(input_dim, input_dim, False)
            self.ln_g = nn.LayerNorm(input_dim)

        self.num_layers = 1

        self.ln_iid = nn.LayerNorm(input_dim)
        self.attn_layer = AttentionLayer(in_dim=input_dim, out_dim=input_dim)

        if self.parallel["bias_mapping"] == 'user': 
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

        if self.rqvae["RQVAE"]:
            self.rq_mf = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])
            self.rq_aggr = ResidualQuantizer(code_dim=input_dim, num_levels=rqvae["codebook_num"], codebook_size=rqvae["codebook_size"])


    def forward(self, x, t, cond_emb, cond_mask, diff_id):

        for idx in range(self.num_layers):
            t_embedding = self.step_mlp(t)

            cond_embedding = self.cond_emb_linear[diff_id](cond_emb)

            x = torch.cat([t_embedding, cond_embedding * cond_mask.unsqueeze(-1), x], axis=1)  # * cond_mask.unsqueeze(-1)

            x = self.diff_models[diff_id][0](x)  # reverse -- 3 FC를 통해 denosing.

        return x

def q_x_fn(model, x_0, t, device):  # forward
    # eq(4)
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise  # x0에 노이즈를 더함.

def diffusion_loss_fn(model,x_0,cond_emb, iid_emb,y_input,
                        device,is_task,style_src=None, uid=None, iid=None):

    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:

        #------------------------
        #sampling
        #------------------------
        batch_size = x_0.shape[0]
        #sample t
        t = torch.randint(0,num_steps,size=(batch_size//2,),device=device)
        if batch_size%2 ==0:
            t = torch.cat([t,num_steps-1-t],dim=0)
        else:
            extra_t = torch.randint(0,num_steps,size=(1,),device=device)
            t = torch.cat([t,num_steps-1-t,extra_t],dim=0)
        t = t.unsqueeze(-1)

        x,e = q_x_fn(model,x_0,t,device)
        
        #random mask
        cond_mask = 1 * (torch.rand(cond_emb.shape[0],device=device) <= mask_rate  )
        cond_mask = 1 - cond_mask.int()

        #pred noise
        output = model(x, t.squeeze(-1),cond_emb,cond_mask )

        return F.smooth_l1_loss(e, output)

    elif is_task:
        final_output, iid_emb=p_sample_loop(model,cond_emb,iid_emb,device)

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

        tokens = torch.stack([iid_emb, final_output_m,  style_tok_u, item_style_tok], dim=1)
        out = model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
        final_output = out[:, 0, :]  # (B, D)

        y_pred = torch.sum(final_output * iid_emb, dim=1) 

        # domain bias 
        mu_t = model.tgt_global_bias
        y_pred = y_pred + mu_t
        
        #MSE
        task_loss =   (y_pred - y_input.squeeze().float()).square().mean()
        #RMSE
        #task_loss =   (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]

        return F.smooth_l1_loss(x_0, final_output) + model.task_lambda* task_loss

def diffusion_loss_fn_parallel(
    model, x_0_m, x_0_g, cond_emb1, cond_emb2, iid_emb, y_input, device, is_task, q_embs1=None, q_embs2=None, style_src=None, uid=None, iid=None, Q_emb1=None, Q_emb2=None
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
            return F.mse_loss(e_m, output1) + F.mse_loss(e_g, output2)
        elif model.aggregation == "aggregation_ab1":
            output1 = model(x_m, t.squeeze(-1), c1, cond_mask1, diff_id=0)  
            return F.mse_loss(e_m, output1)
        elif model.aggregation == "aggregation_ab2":
            output1 = model(x_g, t.squeeze(-1), c2, cond_mask2, diff_id=0)  
            return F.mse_loss(e_g, output1)

    elif is_task:  
        if model.rqvae["start_point"] == "src_u":
            start1, start2 = cond_emb1, cond_emb2 
        elif model.rqvae["start_point"] == "quant_u":
            start1, start2 = Q_emb1, Q_emb2
        elif model.rqvae["start_point"] == "noise":
            start1, start2 = torch.randn_like(cond_emb1), torch.randn_like(cond_emb2)

        if model.rqvae["RQVAE"] == True:
            cond1, cond2 = q_embs1, q_embs2
            p_sample = p_sample_loop_parallel 
        else: 
            cond1, cond2 = cond_emb1, cond_emb2 
            p_sample = p_sample_loop 

        iid_emb = model.ln_iid(iid_emb)

        if model.aggregation == "aggregation":
            final_output_m, iid_emb = p_sample(model, start1, cond1, iid_emb, device, diff_id=0)
            final_output_g, iid_emb = p_sample(model, start2, cond2, iid_emb, device, diff_id=1)
            final_output_m_proj = model.linear_m(final_output_m)
            final_output_g_proj = model.linear_g(final_output_g)
            final_output_m = model.ln_m(final_output_m_proj)
            final_output_g = model.ln_g(final_output_g_proj)
            base_tokens = torch.stack([final_output_m, final_output_g], dim=1)

        elif model.aggregation == "aggregation_ab1":
            final_output_m, iid_emb = p_sample(model, start1, cond1, iid_emb, device, diff_id=0)
            final_output_m_proj = model.linear_m(final_output_m)
            final_output_m = model.ln_m(final_output_m_proj)
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

            if model.parallel["bias_mapping"] == 'user':
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
            
            if model.parallel["bias_mapping"] == 'user':
                style_u = model.user_style_mapper(style_u)            
                mapping_loss = F.mse_loss(style_u, model.style_tgt_user[uid, :2])
                style_u = style_u.detach()

            style_tok = model.style_encoder(style_u)  # (B, D)
            style_tok = model.style_ln(style_tok)  # (B, D)
            style_tok_u = model.style_scale * style_tok  # (B, D)

            tokens = torch.cat([base_tokens, style_tok_u.unsqueeze(1)], dim=1)

        out = model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
        final_output = out[:, 0, :]  # (B, D)
        y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측

        # MSE
        task_loss = (y_pred - y_input.squeeze().float()).square().mean()

        if model.parallel["bias_mapping"] == "user":
            task_loss += (model.parallel["mapping_lambda"] * mapping_loss)

        if model.aggregation == "aggregation":
            return F.mse_loss(x_0_m, final_output_m) + F.mse_loss(x_0_g, final_output_g) + model.task_lambda * task_loss
        elif model.aggregation == "aggregation_ab1":
            return F.mse_loss(x_0_m, final_output_m) + model.task_lambda * task_loss
        elif model.aggregation == "aggregation_ab2":
            return F.mse_loss(x_0_g, final_output_g) + model.task_lambda * task_loss

# generation fun
def p_sample(model, cond_emb, x, iid_emb, device, diff_id):  # ALM + task loss
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

    return sample, iid_emb


def p_sample_loop(model, start_emb, cond_emb, iid_input, device, diff_id):
    cur_x, iid_emb_out = p_sample(model, cond_emb, start_emb, iid_input, device, diff_id)  # denoised embedding, item emb

    return cur_x, iid_emb_out


def p_sample_parallel(model, cond_emb, x, iid_emb, device, diff_id):  
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    B = cond_emb.shape[1]
    cond_mask = torch.zeros(B, device=device).int()  

    model_kwargs = {
        "cond_emb": cond_emb.to(device),
        "cond_mask": cond_mask,
        "diff_id": diff_id,  
    }

    model_fn = model_wrapper_hierarchical_cond(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)  

    sample = dpm_solver.sample(  
        x,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )

    return sample, iid_emb


def p_sample_loop_parallel(model, start_emb, cond_emb, iid_input, device, diff_id):
    cur_x, iid_emb_out = p_sample_parallel(model=model, cond_emb=cond_emb, x=start_emb, iid_emb=iid_input, device=device, diff_id=diff_id)
    return cur_x, iid_emb_out