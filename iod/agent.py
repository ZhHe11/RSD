import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torch


class AgentWrapper(object):
    """Wrapper for communicating the agent weights with the sampler."""

    def __init__(self, policies):
        for k, v in policies.items():
            setattr(self, k, v)
        if 'target_traj_encoder' not in policies:
            self.target_traj_encoder = copy.deepcopy(policies['traj_encoder'])

    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
        

    def get_torch_concat_obs(self, obs, option, dim=1):
        concat_obs = torch.cat([obs] + [option], dim=dim)
        return concat_obs
        
        
    def Psi(self, phi_x, phi_x0=None):
        if 'Projection' in self.method['phi']:   
            return torch.tanh(2/self.max_path_length * (phi_x))
        else:
            return phi_x
        
        
    @torch.no_grad()  
    def _get_concat_obs(self, obs, option):
        x = self.get_torch_concat_obs(obs, option)
        psi_s = self.Psi(self.target_traj_encoder(obs).mean.detach())
        return self.get_torch_concat_obs(x, psi_s)
        

    @torch.no_grad()
    def gen_z_phi_g(self, phi_g, obs, device='cpu', ret_emb: bool = False):
        traj_encoder = self.target_traj_encoder.to(device)
        goal_z = phi_g
        target_cur_z = traj_encoder(obs.unsqueeze(0)).mean.squeeze(0)

        z = self.vec_norm(goal_z - target_cur_z)
        if ret_emb:
            return z.numpy(), target_cur_z.numpy(), goal_z.numpy()
        return z
    
    @torch.no_grad()
    def gen_phi_s(self, obs, device='cpu', ret_emb: bool = False):
        traj_encoder = self.target_traj_encoder.to(device)
        target_cur_z = traj_encoder(obs.unsqueeze(0)).mean.squeeze(0)
        return target_cur_z.numpy()
        
        
    def get_param_values(self):
        param_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                param_dict[k] = v.state_dict() if hasattr(v, "state_dict") else v.get_param_values()

        return param_dict

    def set_param_values(self, state_dict):
        for k, v in state_dict.items():
            net = getattr(self, k)
            net.load_state_dict(v)

    def eval(self):
        for v in self.__dict__.values():
            if isinstance(v, torch.nn.Module): 
                v.eval()

    def train(self):
        for v in self.__dict__.values():
            if isinstance(v, torch.nn.Module): 
                v.train()

    def reset(self):
        self.default_policy.reset()



def copy_init_policy(policy, qf1, qf2):
    policy = copy.deepcopy(policy)
    qf1 = copy.deepcopy(qf1)
    qf2 = copy.deepcopy(qf2)
    target_qf1 = copy.deepcopy(qf1)
    target_qf2 = copy.deepcopy(qf2)

    return policy, qf1, qf2, target_qf1, target_qf2
    
    
    
    
     
    
    
    

















