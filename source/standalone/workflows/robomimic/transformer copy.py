import torch
import torch.nn as nn
from torch.nn import functional as F

# dim_act = 4
# dim_obs = 11
# history_size = 128
# seq_size = history_size

batch_size = 128
# dim_emb = n_head * head_size
dim_emb = 96
n_head = 6
n_layer = 6
dropout = 0.2
# hidden_size = 128

device = "cuda"


class SelfAttention(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, input_seq_size, output_seq_size):
        super().__init__()
        self.key = nn.Linear(dim_emb, head_size, bias=False)
        self.query = nn.Linear(dim_emb, head_size, bias=False)
        self.value = nn.Linear(dim_emb, head_size, bias=False)

        # # all_seq_size = 2 * input_seq_size
        # input_size = 2 * input_seq_size - 1
        # output_size = 2 * output_seq_size - 1
        # all_seq_size = input_size + output_size
        tril = torch.tril(torch.ones(input_seq_size, input_seq_size))
        # tril = torch.tril(torch.ones(all_seq_size, all_seq_size))
        # tril[input_size:, :input_size] = torch.tril(torch.ones(input_size, input_size))
        # tril[input_size:, input_size:] = torch.diag(torch.ones(input_size))
        # # all_seq_size = 2 * input_seq_size - 1 + 2 * output_seq_size - 1
        self.register_buffer('tril', tril)

        self.dropout = nn.Dropout(dropout)

        # self.wei = torch.empty((0, input_seq_size))

    def forward(self, x):
        # input of size (batch, history size, dim obs)
        # output of size (batch, history size, head size)
        B, H, D = x.shape
        q = self.query(x) # (B, H, hs)
        k = self.key(x)   # (B, H, hs)
        v = self.value(x) # (B, H, hs)
        # k = self.key(torch.cat([y, x], dim=1))
        # v = self.value(torch.cat([y, x], dim=1))
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, H, hs) @ (B, hs, H) -> (B, H, H)
        wei = wei.masked_fill(self.tril[:H, :H] == 0, float('-inf')) # (B, H, H)
        # wei = wei.masked_fill(self.tril[-H:] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, H, H)

        # # self.wei = torch.cat((self.wei, (torch.sum(wei, dim=-2) / H).cpu()))
        # # self.wei = torch.cat((self.wei, (torch.sum(wei * torch.arange(1, H + 1, device=device).reshape(1, -1, 1) / H, dim=-2) / H).cpu()))
        # self.wei = torch.cat((self.wei, wei[:, -1].cpu()))
        
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, H, H) @ (B, H, hs) -> (B, H, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, input_seq_size, output_seq_size):
        super().__init__()
        head_size = dim_emb // n_head
        self.heads = nn.ModuleList([SelfAttention(head_size, input_seq_size, output_seq_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, dim_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb, 4 * dim_emb),
            nn.ReLU(),
            nn.Linear(4 * dim_emb, dim_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, input_seq_size, output_seq_size):
        # dim_emb: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(input_seq_size, output_seq_size)
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(dim_emb)
        self.ln2 = nn.LayerNorm(dim_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        # x = self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    # def forward(self, inputs):
    #     x, y = inputs
    #     x = x + self.sa(self.ln1(x), self.ln1(y))
    #     # x = self.sa(self.ln1(x))
    #     x = x + self.ffwd(self.ln2(x))
    #     outputs = x, y
    #     return outputs

class ActorTransformer(nn.Module):

    def __init__(self, dim_goal, dim_obs, dim_act, history_size, horizon_size):
        super().__init__()
        self.input_seq_size = history_size
        # self.output_seq_size = history_size
        self.output_seq_size = horizon_size
        # each token directly reads off the outputs for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, dim_emb)
        # self.goal_vel_embedding_table = nn.Linear(1, dim_emb, bias=False)
        # self.goal_pos_embedding_table = nn.Linear(3, dim_emb, bias=False)
        # self.goal_embedding = nn.Linear(dim_goal, dim_emb)
        # self.obs_embedding = nn.Linear(dim_obs, dim_emb)
        self.goal_obs_embedding = nn.Linear(dim_goal + dim_obs, dim_emb)
        self.act_embedding = nn.Linear(dim_act, dim_emb)
        # self.noised_act_embedding = nn.Linear(dim_act, dim_emb)
        self.time_act_embedding = nn.Linear(1 + dim_act, dim_emb)
        self.time_obs_embedding = nn.Linear(1 + dim_obs, dim_emb)
        self.output_act_embedding = nn.Linear(dim_act, dim_emb)
        self.output_obs_embedding = nn.Linear(dim_obs, dim_emb)
        # self.position_embedding = nn.Embedding(input_seq_size + output_seq_size - 1, dim_emb)
        self.input_position_embedding = nn.Embedding(self.input_seq_size, dim_emb)
        self.output_position_embedding = nn.Embedding(self.output_seq_size, dim_emb)
        self.blocks = nn.Sequential(*[Block(self.input_seq_size, self.output_seq_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim_emb) # final layer norm
        # self.lm_head = nn.Linear(dim_emb, dim_act)
        self.obs_mean = nn.Linear(dim_emb, dim_obs)
        self.obs_log_std = nn.Linear(dim_emb, dim_obs)
        self.act_mean = nn.Linear(dim_emb, dim_act)
        self.act_log_std = nn.Linear(dim_emb, dim_act)
        self.f_obs_mean = nn.Linear(dim_emb, dim_obs)
        self.f_act_mean = nn.Linear(dim_emb, dim_act)
        # self.value_mean = nn.Linear(dim_emb * 2 * (2 * self.input_seq_size - 1), 1)
        # self.value_log_std = nn.Linear(dim_emb * 2 * (2 * self.input_seq_size - 1), 1)
        self.value_act_mean = nn.Linear(dim_emb, 1)
        self.value_obs_mean = nn.Linear(dim_emb, 1)
        self.value_log_std = nn.Linear(dim_emb, 1)
        # self.lm_head = nn.Linear(dim_emb * input_seq_size, dim_act)
        # self.lm_head = nn.Linear(dim_emb * input_seq_size, dim_act * act_seq_size)
        
        # self.fc_in = nn.Linear((dim_goal + dim_obs) * input_seq_size, hidden_size)
        # # self.fc_hid = nn.Linear(hidden_size, hidden_size)
        # self.fc_out = nn.Linear(hidden_size, dim_act)
        # hidden_size = 128
        # self.fc = nn.Sequential(
        #     nn.Linear((dim_goal + dim_obs) * history_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, dim_act * history_size)
        # )

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # def forward(self, goals, obss, acts, noised_acts, noised_obss, time_steps):
    def forward(self, goals, obss, acts=None, noised_acts=None, noised_obss=None):
        H_in = self.input_seq_size
        H_out = self.output_seq_size

        # if noised_acts is None:
        #     # noised_acts = acts[:, -1].unsqueeze(1).repeat(1, H_out, 1)
        #     noised_acts = acts[:, -1].unsqueeze(1).repeat(1, H_out - 1, 1)
        #     noised_obss = obss[:, -1].unsqueeze(1).repeat(1, H_out - 1, 1)

        if noised_acts is not None:
            # inputs are (B, H, D_obs) tensor
            # goal_vel_emb = self.goal_vel_embedding_table(goals[:, :, -1].unsqueeze(-1))
            # goal_pos_emb = self.goal_pos_embedding_table(goals[:, :, :3])
            goal_obs = torch.cat([goals.repeat(1, H_in, 1), obss], dim=-1)
            # time_noised_act = torch.cat([time_steps.repeat(1, H_out, 1), noised_acts], dim=2)
            # time_noised_obs = torch.cat([time_steps.repeat(1, H_out - 1, 1), noised_obss], dim=2)
            # goal_emb = self.goal_embedding(goals)
            # obs_emb = self.obs_embedding(obss)
            obs_emb = self.goal_obs_embedding(goal_obs) # (B, H, D)
            act_emb = self.act_embedding(acts)
            # noised_act_emb = self.noised_act_embedding(noised_acts)
            # noised_act_emb = self.time_act_embedding(time_noised_act)
            # noised_obs_emb = self.time_obs_embedding(time_noised_obs)
            noised_act_emb = self.output_act_embedding(noised_acts)
            noised_obs_emb = self.output_obs_embedding(noised_obss)
            # input_seq = torch.cat([goal_vel_emb, goal_pos_emb, obs_emb], dim=1)

            # pos_emb = self.position_embedding(torch.arange(H_in + H_out - 1, device=device))
            # obs_pos_emb = obs_emb + pos_emb[:H_in]
            # act_pos_emb = act_emb + pos_emb[:H_in - 1]
            # noised_act_pos_emb = noised_act_emb + pos_emb[- H_out:]
            # noised_obs_pos_emb = noised_obs_emb + pos_emb[- H_out + 1:]
            input_pos_emb = self.input_position_embedding(torch.arange(H_in, device=device)) # (H, D)
            output_pos_emb = self.output_position_embedding(torch.arange(H_out, device=device))
            obs_pos_emb = obs_emb + input_pos_emb # (B, H, D)
            act_pos_emb = act_emb + input_pos_emb[:-1]
            # noised_act_pos_emb = noised_act_emb + output_pos_emb
            noised_act_pos_emb = noised_act_emb + output_pos_emb[1:]
            noised_obs_pos_emb = noised_obs_emb + output_pos_emb[1:]
            # x = torch.cat([goal_emb, obs_pos_emb], dim=1)
            # x = torch.cat([obs_pos_emb, act_pos_emb], dim=1)
            # x[:, 0::2] = obs_pos_emb
            # x[:, 1::2] = act_pos_emb
            # x = obs_pos_emb
            # x = torch.cat([obs_pos_emb, noised_act_pos_emb], dim=1)

            # x = torch.cat([obs_pos_emb, act_pos_emb, noised_act_pos_emb, noised_obs_pos_emb], dim=1)
            input_size = 2 * H_in - 1
            # x[:, 0:input_size:2] = obs_pos_emb
            # x[:, 1:input_size:2] = act_pos_emb
            # x[:, input_size::2] = noised_act_pos_emb
            # x[:, input_size + 1::2] = noised_obs_pos_emb

            y = torch.cat([obs_pos_emb, act_pos_emb], dim=1)
            x = torch.cat([noised_act_pos_emb, noised_obs_pos_emb], dim=1)
            y[:, 0::2] = obs_pos_emb
            y[:, 1::2] = act_pos_emb
            x[:, 0::2] = noised_act_pos_emb
            x[:, 1::2] = noised_obs_pos_emb
            x = torch.cat([y, x], dim=1)

            x = self.blocks(x) # (B, H, D)
            # x = self.blocks(y)
            # x, y = self.blocks((x, y))
            x = self.ln_f(x)
            # x = self.ln_f(x[:, input_size:]) # (B, H, D)
            # x = self.ln_f(x[:, :input_size])
            # x = x[:, 1:]
            # x = x.reshape(x.shape[0], -1)
            # outputs = self.lm_head(x) # (B, H, D_act)

            # act_mean = self.act_mean(x[:, 0::2])
            # act_log_std = self.act_log_std(x[:, 0::2])
            # obs_mean = self.obs_mean(x[:, 1::2])
            # obs_log_std = self.obs_log_std(x[:, 1::2])
            # value_mean = self.value_mean(x)
            # value_log_std = self.value_log_std(x)
            # value_act_mean = self.value_act_mean(x[:, 0::2])
            # value_obs_mean = self.value_obs_mean(x[:, 1::2])
            # value_act_mean = self.act_mean(x[:, 0::2])
            # value_obs_mean = self.obs_mean(x[:, 1::2])
            act_mean = self.act_mean(x[:, 0:input_size:2])
            obs_mean = self.obs_mean(x[:, 1:input_size:2])
            f_act_mean = self.f_act_mean(x[:, input_size + 1::2])
            f_obs_mean = self.f_obs_mean(x[:, input_size::2])

            # value_mean = torch.cat([value_act_mean, value_obs_mean], dim=1)
            # value_mean[:, 0::2] = value_act_mean
            # value_mean[:, 1::2] = value_obs_mean

            # obs_mean = self.obs_mean(x[:, :-1])
            # obs_log_std = self.obs_log_std(x[:, :-1])
            # act_mean = self.act_mean(x)
            # act_log_std = self.act_log_std(x)
            # outputs = outputs[:, -1]
            # outputs = outputs[:, :4]
            # x = self.lm_head_1(x)
            # outputs = outputs.reshape(outputs.shape[0], 4, -1)

            # x = self.fc_in(goal_obs.reshape(goal_obs.shape[0], -1))
            # # x = self.fc_hid(x)
            # outputs = self.fc_out(x)

            # return outputs
            # return act_mean, act_log_std, obs_mean, obs_log_std
            # return value_mean, value_log_std
            # return value_act_mean, value_obs_mean
            return act_mean, obs_mean, f_act_mean, f_obs_mean
        
        elif acts is not None:
            goal_obs = torch.cat([goals.repeat(1, H_in, 1), obss], dim=-1)
            obs_emb = self.goal_obs_embedding(goal_obs)
            act_emb = self.act_embedding(acts)
            input_pos_emb = self.input_position_embedding(torch.arange(H_in, device=device)) # (H, D)
            obs_pos_emb = obs_emb + input_pos_emb
            act_pos_emb = act_emb + input_pos_emb[:-1]
            x = torch.cat([obs_pos_emb, act_pos_emb], dim=1)
            x[:, 0::2] = obs_pos_emb
            x[:, 1::2] = act_pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            act_mean = self.act_mean(x[:, 0::2])
            # act_log_std = self.act_log_std(x[:, 0::2])
            obs_mean = self.obs_mean(x[:, 1::2])
            # obs_log_std = self.obs_log_std(x[:, 1::2])
            # return act_mean, act_log_std, obs_mean, obs_log_std
            return act_mean, obs_mean
        
        else:
            goal_obs = torch.cat([goals.repeat(1, H_in, 1), obss], dim=-1)
            obs_emb = self.goal_obs_embedding(goal_obs)
            input_pos_emb = self.input_position_embedding(torch.arange(H_in, device=device)) # (H, D)
            obs_pos_emb = obs_emb + input_pos_emb
            x = obs_pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            act_mean = self.act_mean(x)
            # act_mean = self.fc(goal_obs.reshape(goal_obs.shape[0], -1)).reshape(goal_obs.shape[0], H_in, -1)
            return act_mean

    # def sample(self, goals, obss, acts, noised_acts, time_steps):
    #         B, H, D = obss.shape

    #         # inputs are (B, H, D_obs) tensor
    #         # goal_vel_emb = self.goal_vel_embedding_table(goals[:, :, -1].unsqueeze(-1))
    #         # goal_pos_emb = self.goal_pos_embedding_table(goals[:, :, :3])
    #         goal_obs = torch.cat([goals.repeat(1, H, 1), obss], dim=2)
    #         time_noised_act = torch.cat([time_steps, noised_acts], dim=2)
    #         # goal_emb = self.goal_embedding(goals)
    #         # obs_emb = self.obs_embedding(obss)
    #         obs_emb = self.goal_obs_embedding(goal_obs) # (B, H, D)
    #         act_emb = self.act_embedding(acts)
    #         # noised_act_emb = self.noised_act_embedding(noised_acts)
    #         noised_act_emb = self.time_act_embedding(time_noised_act)
    #         # input_seq = torch.cat([goal_vel_emb, goal_pos_emb, obs_emb], dim=1)

    #         pos_emb = self.position_embedding(torch.arange(H, device=device)) # (H, D)
    #         obs_pos_emb = obs_emb + pos_emb # (B, H, D)
    #         act_pos_emb = act_emb + pos_emb[:-1]
    #         noised_act_pos_emb = noised_act_emb + pos_emb[-1]
    #         # x = torch.cat([goal_emb, obs_pos_emb], dim=1)
    #         # x = torch.cat([obs_pos_emb, act_pos_emb], dim=1)
    #         # x[:, 0::2] = obs_pos_emb
    #         # x[:, 1::2] = act_pos_emb
    #         # x = obs_pos_emb
    #         x = torch.cat([obs_pos_emb, noised_act_pos_emb], dim=1)
    #         x = self.blocks(x) # (B, H, D)
    #         # x = self.ln_f(x)
    #         x = self.ln_f(x[:, H:]) # (B, H, D)
    #         # x = x[:, 1:]
    #         # x = x.reshape(x.shape[0], -1)
    #         # outputs = self.lm_head(x) # (B, H, D_act)
    #         # obs_mean = self.obs_mean(x[:, 1::2])
    #         # obs_log_std = self.obs_log_std(x[:, 1::2])
    #         # act_mean = self.act_mean(x[:, 0::2])
    #         # act_log_std = self.act_log_std(x[:, 0::2])
    #         # obs_mean = self.obs_mean(x[:, :-1])
    #         # obs_log_std = self.obs_log_std(x[:, :-1])
    #         act_mean = self.act_mean(x)
    #         # act_log_std = self.act_log_std(x)
    #         # outputs = outputs[:, -1]
    #         # outputs = outputs[:, :4]
    #         # x = self.lm_head_1(x)
    #         # outputs = outputs.reshape(outputs.shape[0], 4, -1)

    #         # x = self.fc_in(goal_obs.reshape(goal_obs.shape[0], -1))
    #         # # x = self.fc_hid(x)
    #         # outputs = self.fc_out(x)

    #         # return outputs
    #         # return obs_mean, obs_log_std, act_mean, act_log_std
    #         return act_mean

    # def policy(self, history_obs):
    #     # history_obs is (B, H, D_obs) array of indices in the current context
    #     # crop history_obs to the last history_size tokens
    #     idx_cond = history_obs[:, -history_size:]
    #     # get the predictions
    #     history_act = self(idx_cond)
    #     # focus only on the last time step
    #     act = history_act[:, -1, :] # becomes (B, D_act)
    #     return act
