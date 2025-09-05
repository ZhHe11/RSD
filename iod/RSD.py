import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

import wandb
from iod.agent import AgentWrapper

from iod.viz_utils import PlotMazeTraj

import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange, tqdm
from iod.GradCLipper import GradClipper
import matplotlib.pyplot as plt
import torch.distributions as dist
from iod.viz_utils import PlotMazeTrajDist, PlotMazeTrajWindowDist, viz_dist_circle, PlotGMM
from functools import partial


class RSD(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            pixel_shape=None,
            
            init_obs=None,
            
            phi_type="baseline",
            policy_type="baseline",
            explore_type="baseline",
        
            SampleZPolicy=None,
            
            SZN_w2=3,
            SZN_w3=3,
            SZN_window_size=10,
            SZN_repeat_time=5,

            Repr_temperature=0.5,
            Repr_max_step=5,
            
            z_unit = 0,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.dual_slack2 = dual_slack
        
        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None
        
        # added 
        self.last_w = None
        self.epoch_data = None
        
        self.method = {
            "eval": 'random',
            "phi": phi_type,
            "policy": policy_type,
            "explore": explore_type,
        }
        
        '''
        wrapper for agent for online interaction.
        '''
        policy_for_agent = {
            "default_policy": self.option_policy,
            "traj_encoder": self.traj_encoder,
            "InjectPhi": 0, 
            "method": self.method,
            "max_path_length": self.max_path_length,
        }
        self.policy_for_agent = AgentWrapper(policies=policy_for_agent) 
    
        ### new alternative:
        self.init_obs = torch.tensor(init_obs).unsqueeze(0).expand(self.num_random_trajectories, -1).to(self.device)
        self.s0 = torch.tensor(init_obs).unsqueeze(0).to(self.device)
        self.input_token = torch.zeros(self.num_random_trajectories, self.num_random_trajectories).float().to(self.device)  
        self.buffer_ready = 0
        
        self.last_z = None
        self.SampleZPolicy = SampleZPolicy.to(self.device)
        self.ResetSZPolicy = copy.deepcopy(self.SampleZPolicy)
        self.SampleZPolicy_optim = optim.Adam(self.SampleZPolicy.parameters(), lr=1e-4)
        self.grad_clip = GradClipper(clip_type='clip_norm', threshold=3, norm_type=2)

        self.last_policy = copy.deepcopy(self.option_policy)
        self.last_qf1 = copy.deepcopy(self.qf1)
        self.last_qf2 = copy.deepcopy(self.qf2)
        self.last_alpha = copy.deepcopy(self.log_alpha)
        self.copyed = 0
        with torch.no_grad():
            self.DistWindow = [self.SampleZPolicy(self.input_token)]
        
        self.NumSampleTimes = 0
        self.last_trial = []
        self.new_trial = []

        self.SZN_w2 = SZN_w2
        self.SZN_w3 = SZN_w3
        self.SZN_window_size = SZN_window_size
        self.SZN_repeat_time = SZN_repeat_time
        
        self.Repr_max_step = Repr_max_step
        self.SfReprBuffer = []
        
        self.z_unit = z_unit
    
        self.train_policy = False
        self.train_phi = False
        self.save_debug = False
    
    
    def Psi(self, phi_x, phi_x0=None):
        if 'Projection' in self.method['phi']:   
            return torch.tanh(2/self.max_path_length * (phi_x))
        else:
            return phi_x
    
    def norm(self, x, keepdim=False):
        return torch.norm(x, p=2, dim=-1, keepdim=keepdim)        
        
    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

    @property
    def policy(self):
        return {
            'option_policy': self.policy_for_agent,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)


    @torch.no_grad()
    def EstimateValue(self, policy, alpha, qf1, qf2, option, state, num_samples=3):
        batch = option.shape[0]     # [s0, z]
        processed_cat_obs = self._get_concat_obs(policy.process_observations(state), option.float())     # [b,dim_s+dim_z]
        dist, info = policy(processed_cat_obs)    # [b, dim]
        actions = dist.sample((num_samples,))          # [n, b, dim]
        log_probs = dist.log_prob(actions).squeeze(-1)  # [n, b]
        processed_cat_obs_flatten = processed_cat_obs.repeat(num_samples, 1, 1).view(batch * num_samples, -1)      # [n*b, dim_s+z]
        actions_flatten = actions.view(batch * num_samples, -1)     # [n*b, dim_a]
        q_values = torch.min(qf1(processed_cat_obs_flatten, actions_flatten), qf2(processed_cat_obs_flatten, actions_flatten))      # [n*b, dim_1]
        alpha = alpha.param.exp()
        values = q_values - alpha * log_probs.view(batch*num_samples, -1)      # [n*b, 1]
        values = values.view(num_samples, batch, -1)        # [n, b, 1]
        E_V = values.mean(dim=0)        # [b, 1]

        return E_V.squeeze(-1)
    

    def copy_params(self, ori_model, target_model):
        for t_param, param in zip(target_model.parameters(), ori_model.parameters()):
            t_param.data.copy_(param.data)


    def cal_regeret(self, z, state):
        '''
        z: [batch_sample, dim_option]
        '''
        V_z = self.EstimateValue(policy=self.option_policy, alpha=self.log_alpha, qf1=self.qf1, qf2=self.qf2, option=z, state=state)                    
        if self.copyed:
            V_z_last_iter = self.EstimateValue(policy=self.last_policy, alpha=self.last_alpha, qf1=self.last_qf1, qf2=self.last_qf2, option=z, state=state)
        else:
            V_z_last_iter = 0
            
        return V_z - V_z_last_iter, V_z


    def get_confidence_mix(self, buffer : list, dist_z, num_dist):
        sf_repr_buffer_tensor = torch.tensor(np.array(buffer)).to(self.device)
        x = sf_repr_buffer_tensor.unsqueeze(0).repeat(num_dist,1,1)
        p_sf = torch.zeros((num_dist,1)).to(self.device)

        for i in range(x.shape[1]):
            x_i = x[:,i]
            if i == 0:
                p_sf = dist_z.log_prob(x_i)
            else:
                p_sf = torch.maximum(p_sf, dist_z.log_prob(x_i))
        confidence = p_sf
        return confidence
    
    def get_confidence(self, buffer: list, dist_z, num_dist):
        sf_repr_buffer_tensor = torch.tensor(np.array(buffer), device=self.device)        
        if sf_repr_buffer_tensor.shape[0] % num_dist != 0:
            CorrectLen = sf_repr_buffer_tensor.shape[0] - sf_repr_buffer_tensor.shape[0] % num_dist
            sf_repr_buffer_tensor = sf_repr_buffer_tensor[:CorrectLen]
        
        x = sf_repr_buffer_tensor.reshape(-1, num_dist, self.dim_option) 

        log_probs = dist_z.log_prob(x)

        log_probs = log_probs.view(num_dist, -1)

        confidence = torch.max(log_probs, dim=1)[0]

        return confidence

    def UpdateGMM(self, dists, GMM=None, mix_dist_prob=None, device='cuda'):
        if GMM is None:
            component_distribution = dist.Independent(
                dist.Normal(
                    loc=torch.stack([g.mean[0] for g in dists]),
                    scale=torch.stack([g.stddev[0] for g in dists])
                ),
                reinterpreted_batch_ndims=1
            )
            if mix_dist_prob is None:
                mixture_distribution = dist.Categorical(
                    probs=(torch.ones(len(dists)) / len(dists)).to(device)
                )
            else: 
                mixture_distribution = dist.Categorical(
                    probs=mix_dist_prob
                )
            window_dist = dist.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution
            )
            return window_dist
    
        else:
            component_distribution = GMM.component_distribution
            mixture_distribution = mixture_distribution

            window_dist = dist.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution
            )

            return window_dist


    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            if self.method['explore'] == 'SZN' and self.buffer_ready:
                if self.NumSampleTimes == self.SZN_repeat_time:
                    self.NumSampleTimes = 0
                    self.copy_params(self.ResetSZPolicy, self.SampleZPolicy)
                    self.SampleZPolicy_optim = optim.Adam(self.SampleZPolicy.parameters(), lr=3e-2)
                    # training loop
                    for t in trange(100):
                        # Reset the SZN:
                        z_values = self.SampleZPolicy(self.input_token).mean
                        probabilities = F.softmax(z_values, dim=-1)
                        z_index = torch.multinomial(probabilities, 1).squeeze(-1)
                        z_onehot = F.one_hot(z_index, num_classes=self.dim_option).float()
                        p_z = (probabilities * z_onehot).sum(dim=-1)
                        z_logp = torch.log(p_z)
                        V_szn, V_z = self.cal_regeret(z_onehot, self.init_obs)
                        V_z = (V_z - V_z.mean()) / (V_z.std() + 1e-6)  
                        V_szn = (V_szn - V_szn.mean()) / (V_szn.std() + 1e-6)

                        self.SampleZPolicy_optim.zero_grad()    

                        loss_SZP = (-z_logp * (V_szn.detach() + V_z.detach())).mean()

                        loss_SZP.backward()
                        self.grad_clip.apply(self.SampleZPolicy.parameters())
                        self.SampleZPolicy_optim.step()
                        if wandb.run is not None:
                            wandb.log({
                                "SZN/loss_SZP": loss_SZP,
                                "SZN/logp": z_logp.mean(),
                                "epoch": runner.step_itr,
                            })
                    # save k-1 policy and qf
                    # Attention this part should process after all other things
                    self.copy_params(self.option_policy, self.last_policy)
                    self.copy_params(self.log_alpha, self.last_alpha)
                    self.copy_params(self.qf1, self.last_qf1)
                    self.copy_params(self.qf2, self.last_qf2)
                    self.copyed = 1
                    # Visualization
                    if wandb.run is not None:
                        probabilities = probabilities.detach().cpu().numpy()
                        path = wandb.run.dir + '/E' + str(runner.step_itr)
                        fig = plt.figure(figsize=(8, 5), facecolor='w')
                        plt.bar(range(len(probabilities[0])), probabilities[0], tick_label=[f"z{i}" for i in range(len(probabilities[0]))])
                        plt.xlabel("z values")
                        plt.ylabel("Probabilities")
                        plt.title("Distribution of Probabilities")
                        plt.savefig(path + '-Regret' + '.png')
                        print('save at: ' + path + '-Regret' + '.png')
                        plt.close()
                
                ## GMM samples
                self.NumSampleTimes += 1
                z_values = self.SampleZPolicy(self.input_token).mean
                probabilities = F.softmax(z_values, dim=-1)
                z_index = torch.multinomial(probabilities, 1).squeeze(-1)
                z_onehot = F.one_hot(z_index, num_classes=self.dim_option).float().detach().cpu().numpy()
                extras = self._generate_option_extras(z_onehot, psi_g=z_onehot)   
            
            else: 
                extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
                
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

            if self.method['explore'] == 'SZN' and self.buffer_ready: 
                if self.NumSampleTimes == self.SZN_repeat_time * len(self.DistWindow):
                    # window pool operation: PopDist   
                    # Method 2. pop the dist whose Regret less than 0;
                    def PopDistDeque(window_size=5, pop_min=True):
                        if len(self.DistWindow) >= window_size:
                            if pop_min:
                                All_Regrets = torch.tensor([(self.cal_regeret(dist_i.sample(), self.init_obs)[0]).mean() for dist_i in self.DistWindow])
                                min_index = torch.argmin(All_Regrets)
                                self.DistWindow.pop(min_index)
                            else:
                                self.DistWindow.pop(0)
                        return self.DistWindow
                    
                    self.NumSampleTimes = 0
                    self.copy_params(self.ResetSZPolicy, self.SampleZPolicy)
                    self.SampleZPolicy_optim = optim.Adam(self.SampleZPolicy.parameters(), lr=3e-2)
                    with torch.no_grad():
                        self.DistWindow = PopDistDeque(self.SZN_window_size, pop_min=True)
                    window_dist = self.UpdateGMM(self.DistWindow, device=self.device)
                    
                    for t in trange(100):
                        # Reset the SZN:
                        dist_z = self.SampleZPolicy(self.input_token)
                        z = dist_z.rsample()
                        z_logp = dist_z.log_prob(z.detach())
                        if self.z_unit:
                            z = self.vec_norm(z)
                        V_szn, V_z = self.cal_regeret(z, self.init_obs)
                        Regret = V_szn
                        V_z_value = V_z
                        V_szn = (V_szn - V_szn.mean()) / (V_szn.std() + 1e-6)
                        self.SampleZPolicy_optim.zero_grad()    
                        # weight of GMM KL
                        log_pz = window_dist.log_prob(z)
                        pz = torch.exp(log_pz)
                        log_qz = z_logp
                        kl_window = pz * (log_pz - log_qz)
                        # weight of Confidence Factor
                        confidence = self.get_confidence(self.SfReprBuffer, dist_z, num_dist=self.num_random_trajectories)  
                        confidence = torch.clamp(confidence, max=0)
                        # total loss
                        loss_SZP = (-z_logp * (V_szn.detach()) - self.SZN_w2 * kl_window).mean() - self.SZN_w3 * confidence.mean()
                        
                        loss_SZP.backward()
                        self.grad_clip.apply(self.SampleZPolicy.parameters())
                        self.SampleZPolicy_optim.step()
                        if wandb.run is not None:
                            wandb.log({
                                "SZN/loss_SZP": loss_SZP,
                                "SZN/logp": z_logp.mean(),
                                "SZN/entropy": dist_z.entropy().mean(),
                                "SZN/kl_window": kl_window.mean(),
                                "SZN/confidence": confidence.mean(),
                                "SZN/V_z": V_z_value.mean(),
                                "SZN/Regret": Regret.mean(),
                                "epoch": runner.step_itr,
                            })
                    # window queue operation    
                    with torch.no_grad():
                        dist = self.SampleZPolicy(self.input_token)    
                        is_different = 1
                        for j in range(len(self.DistWindow)):
                            dist_j = self.DistWindow[j]
                            if (self.norm(dist.mean- dist_j.mean)).mean() < 0.1:
                                is_different = 0
                                break
                        if is_different == 1:
                            self.DistWindow.append(dist)
                                
                    # save k-1 policy and qf
                    self.copy_params(self.option_policy, self.last_policy)
                    self.copy_params(self.log_alpha, self.last_alpha)
                    self.copy_params(self.qf1, self.last_qf1)
                    self.copy_params(self.qf2, self.last_qf2)
                    self.copyed = 1
                    self.SfReprBuffer = []
                
                window_dist_raw = self.UpdateGMM(self.DistWindow, device=self.device).component_distribution
                window_len = len(self.DistWindow)
                mix_dist_prob = F.softmax(self.get_confidence_mix(self.new_trial, window_dist_raw, num_dist=window_len) - self.get_confidence_mix(self.last_trial, window_dist_raw, num_dist=window_len))
                min_prob = 1 / self.SZN_window_size * 0.1
                adjusted_probs = torch.maximum(mix_dist_prob, torch.tensor(min_prob))
                adjusted_probs = adjusted_probs / torch.sum(adjusted_probs)
                print(f"mix_dist_prob: {adjusted_probs.detach()}")
                window_dist = self.UpdateGMM(self.DistWindow, mix_dist_prob=adjusted_probs, device=self.device)
                self.last_z = window_dist.sample((self.num_random_trajectories,))
                if self.z_unit:
                    self.last_z = self.vec_norm(self.last_z)
                else:
                    self.last_z = torch.clamp(self.last_z, min=-1, max=1)

                self.NumSampleTimes += 1
                if len(self.SfReprBuffer) == 0:
                    self.last_trial = []
                
                np_z = self.last_z.cpu().numpy()
                extras = self._generate_option_extras(np_z, psi_g=np_z)   
                

            elif self.method['explore'] == 'uniform' and self.buffer_ready:
                random_options = np.random.uniform(-1,1, (runner._train_args.batch_size, self.dim_option))
                extras = self._generate_option_extras(random_options, psi_g=random_options)
            
            else: 
                self.last_z = torch.tensor(random_options, dtype=torch.float32).to(self.device)
                extras = self._generate_option_extras(random_options, psi_g=random_options)   
                

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data):
        self.last_trial.extend(self.new_trial)
        self.new_trial = []
        if self.replay_buffer is not None:
            sfs = []
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                
                self.replay_buffer.add_path(path)
                sfs.append(path['obs'][-1])

            sfs = np.stack(sfs, axis=0)
            with torch.no_grad():
                SfRepr = self.Psi(self.traj_encoder(torch.tensor(sfs).to(self.device)).mean)
            self.SfReprBuffer.extend(SfRepr.cpu().numpy())
            self.new_trial.extend(SfRepr.cpu().numpy())

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        Batch_data = []
        for key, value in samples.items():
            if key in ['obs', 'next_obs', 's_0']: 
                Batch_data.append(value)
        Batch_data = torch.stack(Batch_data).to(self.device)
        
        for key, value in samples.items():
            if key in ['rewards', 'returns', 'ori_obs', 'next_ori_obs', 'pre_tanh_values', 'log_probs']:
                continue
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            if key == 'obs':
                data[key] = Batch_data[0]
            elif key == 'next_obs':
                data[key] = Batch_data[1]
            elif key == 's_0':
                data[key] = Batch_data[2]
            else:
                data[key] = value.float().to(self.device)

        return data
    
    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)
        epoch_data = self._flatten_data(path_data)
        tensors = self._train_components(epoch_data)
        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}
        self.buffer_ready = 1
        for _ in trange(self._trans_optimization_epochs):
            tensors = {}
            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()
            self._optimize_te(tensors, v)
            with torch.no_grad():
                self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        if self.NumSampleTimes == 1:
            self.save_debug = True
        else:
            self.save_debug = False
            
        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
            params=self.traj_encoder.parameters(),
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
                params=[self.dual_lam.param],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                    params=self.dist_predictor.parameters(),
                )


    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
            params=list(self.qf1.parameters()) + list(self.qf2.parameters()),
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
            params=self.option_policy.parameters(),
        )

        self._update_loss_alpha(tensors, internal_vars) 
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
            params=[self.log_alpha.param],
        )
        
        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        if self.method['phi'] == 'Projection':
            self._update_rewards_C(tensors, v)
        else:
            obs = v['obs']
            next_obs = v['next_obs']
            
            if self.inner:
                cur_z = self.traj_encoder(obs).mean
                next_z = self.traj_encoder(next_obs).mean
                target_z = next_z - cur_z

                if self.discrete:
                    masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                    rewards = (target_z * masks).sum(dim=1)
                else:
                    ## baseline
                    inner = (target_z * v['options']).sum(dim=1)
                    rewards = inner
                # For dual objectives
                v.update({
                    'cur_z': cur_z,
                    'next_z': next_z,
                })
            else:
                target_dists = self.traj_encoder(next_obs)

                if self.discrete:
                    logits = target_dists.mean
                    rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
                else:
                    rewards = target_dists.log_prob(v['options'])

            tensors.update({
                'PureRewardMean': rewards.mean(),
                'PureRewardStd': rewards.std(),
            })

            v['rewards'] = rewards
    
    
    def _update_rewards_C(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']
        phi_s = self.traj_encoder(obs).mean
        phi_s_next = self.traj_encoder(next_obs).mean
        psi_g = v['options']
        phi_s_0 = self.traj_encoder(v['s_0']).mean
        psi_s = self.Psi(phi_s)
        psi_s_next = self.Psi(phi_s_next)
        psi_s_0 = self.Psi(phi_s_0)
        updated_option = psi_g
        updated_next_option = psi_g
        k = self.Repr_max_step
        d = 1 / self.max_path_length
        
        # 1. Similarity Reward
        delta_norm = self.norm((psi_s_next - psi_s))
        direction_sim = (1 * (psi_s_next - psi_s) * self.vec_norm(psi_g - psi_s.detach())).sum(dim=-1)        
        phi_obj = direction_sim

        # 2. Goal Arrival Reward
        reward_g_distance = 1/d * torch.clamp(self.norm(psi_g - psi_s) - self.norm(psi_g - psi_s_next), min=-k*d, max=k*d)
        policy_rewards = 1 * reward_g_distance
        
        v.update({
            'cur_z': phi_s,
            'next_z': phi_s_next,
            'rewards': phi_obj,
            'policy_rewards': policy_rewards,
            'psi_s_0': psi_s_0,
            'psi_s': psi_s,
            'psi_s_next': psi_s_next,
            'updated_option': updated_option,
            "updated_next_option": updated_next_option,
        })
        
        tensors.update({
            'phi_obj': phi_obj.mean(),
            'direction_sim': direction_sim.mean(),
            'reward_g_distance': reward_g_distance.mean(),
            'delta_norm': delta_norm.mean(),
            'direction_sim': direction_sim.mean(),
            "policy_rewards": policy_rewards.mean(),
        })
    

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError

            if 'psi_s' in v.keys():
                ##  using norm
                cst_penalty_1 = 1 / self.max_path_length - self.norm(v['psi_s'] - v['psi_s_next'])
                cst_penalty_2 = -self.norm(v['psi_s_0'])
                cst_penalty = torch.clamp(cst_penalty_1, max=self.dual_slack * 1 / self.max_path_length)
                
                te_obj = rewards + dual_lam.detach() * cst_penalty + cst_penalty_2
                v.update({
                    'cst_penalty': cst_penalty
                })
                tensors.update({
                    'cst_penalty_2': cst_penalty_2.mean(),
                    'cst_penalty_1': cst_penalty_1.mean(),
                })
            
            else:
                cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
                cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
                te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'].detach())
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'].detach())

        if self.method['phi'] == 'Projection':
            policy_rewards = v['policy_rewards']
        else:
            policy_rewards = v['rewards']
        
        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards= policy_rewards * self._reward_scale_factor,
            policy=self.option_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            alpha=self.log_alpha,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            loss_type='',
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
            qf1=self.qf1,
            qf2=self.qf2,
            alpha=self.log_alpha,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            loss_type='',   
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v, alpha=self.log_alpha, loss_type='', 
        )


    '''
    Evaluation
    '''
    @torch.no_grad()
    def _evaluate_policy(self, runner, env_name):
        if env_name in ['lm', 'ant_maze', 'ant_maze_large']:  
            if wandb.run is not None:
                path = wandb.run.dir + '/E' + str(runner.step_itr) + '-'
            else:
                path = '.'
                
            FD, AR, eval_metrics = PlotMazeTraj(runner._env, self.traj_encoder, self.option_policy, self.device, Psi=partial(self.Psi), dim_option=self.dim_option, max_path_length=self.max_path_length, path=path, option_type=self.method['eval'])
    
            wandb.log(  
                {
                    "epoch": runner.step_itr,
                    "SampleSteps": runner.step_itr * self.max_path_length * self.num_random_trajectories,
                    "CoordsCover": eval_metrics['MjNumUniqueCoords'], 
                    "Maze_traj": wandb.Image(path + "-Maze_traj.png"),
                },
            )
        
        elif env_name == 'kitchen':
            self.eval_metra(runner)
            
        else:
            self.eval_metra(runner)
            
    def _save_pt(self, epoch):
        if wandb.run is not None:
            path = wandb.run.dir
        else:
            path = '.'
        file_name = path + 'option_policy-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'qf1': self.qf1,
            'qf2': self.qf2,
            'alpha': self.log_alpha,
            'policy': self.option_policy,
            's0': self.s0,
        }, file_name)
        file_name = path + 'traj_encoder-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'traj_encoder': self.traj_encoder,
        }, file_name)
        file_name = path + 'SampleZPolicy-' + str(epoch) + '.pt'
        torch.save({
            'discrete': self.discrete,
            'dim_option': self.dim_option,
            'input_token': self.input_token,
            'goal_sample_network': self.SampleZPolicy,
            'window': self.DistWindow,
        }, file_name)
     
    def eval_kitchen_metra(self, runner):
        random_options = np.eye(self.dim_option)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=True,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )
        eval_option_metrics = {}
        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        
        record_video(runner, 'Video_RandomZ', random_trajectories, skip_frames=self.video_skip_frames)
        
        if wandb.run is not None:
            eval_option_metrics.update({'epoch': runner.step_itr})
            wandb.log(eval_option_metrics)
            
    def eval_metra(self, runner):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            eval_num = 8
            random_options = np.random.randn(eval_num, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        if wandb.run is not None:
            eval_option_metrics.update({'epoch': runner.step_itr,
                                        'interaction_steps': runner.step_itr * self.num_random_trajectories * self.max_path_length,
                                        })
            wandb.log(eval_option_metrics)

    
    # viz the Regert Map
    def viz_Regert_in_Psi(self, state, device='cpu', path='./', ax=None):
        if self.dim_option > 2:
            return
        density = 100
        x = np.linspace(-1, 1, density)
        y = np.linspace(-1, 1, density)
        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        pos = torch.tensor(pos).to(device)
        pos_flatten = pos.view(-1,2)
        option = pos_flatten
        state_batch = state.repeat(option.shape[0], 1)
        Regret = self.cal_regeret(option, state_batch)[0].view(pos.shape[0], pos.shape[1])
        if ax is None:
            fig = plt.figure(figsize=(18, 12), facecolor='w')
            ax = fig.add_subplot(111, projection='3d')
            
        ax.plot_surface(X, Y, Regret.cpu().numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        ax.view_init(60, 270+20)
        ax.set_xlabel('X')          
        ax.set_ylabel('Y')
        ax.set_zlabel('Regret')
        if ax is None:
            plt.savefig(path + '-Regret' + '.png')
            print('save at: ' + path + '-Regret' + '.png')
            plt.close()


