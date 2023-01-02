import argparse
import gym
from gym.spaces import space
from gym import spaces
import torch

import rlkit.torch.pytorch_util as ptu
import yaml
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm
import numpy as np
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from ego_attention import models

def experiment(variant):
    dummy_env = make_env(variant['env'],False)
    #print(dummy_env.observation_space.low.size)
    '''Box([[-inf -inf -inf -inf -inf]
        [-inf -inf -inf -inf -inf]
        [-inf -inf -inf -inf -inf]
        [-inf -inf -inf -inf -inf]
        [-inf -inf -inf -inf -inf]], [[inf inf inf inf inf]
        [inf inf inf inf inf]
        [inf inf inf inf inf]
        [inf inf inf inf inf]
        [inf inf inf inf inf]], (5, 5), float32)
        
    action space Box(-1.0, 1.0, (), float32)'''
    
    obs_dim = np.prod(dummy_env.observation_space.shape or dummy_env.observation_space.n) 
    action_dim = np.prod(dummy_env.action_space.shape or dummy_env.action_space.n)
    #action_dim = 1
    # obs_dim = dummy_env.observation_space.low.size
    
    # action_dim = dummy_env.action_space.low.size
    print('观察维度：{} 动作维度： {}'.format(obs_dim,action_dim))
    #action_dim = spaces.Box(dummy_env.action_space.low.size,dummy_env.action_space.high.size)

    #print(obs_dim,action_dim)
    expl_env = VectorEnv([lambda: make_env(variant['env']) for _ in range(variant['expl_env_num'])])
    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: make_env(variant['env']) for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])
    use_ego_attention = variant['use_ego_attention']
    M = variant['layer_size']

    config = {'type': 'EgoAttentionNetwork', 'layers': [128, 128], 
                        'embedding_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False, 'in': 7}, 
                        'others_embedding_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False, 'in': 7}, 
                        'self_attention_layer': None, 
                        'attention_layer': {'type': 'EgoAttention', 'feature_size': 64, 'heads': 2}, 
                        'output_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False}, 'in': 105, 'out': 5, 
                        'presence_feature_idx': 0}

    if use_ego_attention:
              
        qf1 = models.EgoAttentionNetwork(config).cuda()
        qf2 = models.EgoAttentionNetwork(config).cuda()
        target_qf1 = models.EgoAttentionNetwork(config).cuda()
        target_qf2 = models.EgoAttentionNetwork(config).cuda()
    
    else:

        qf1 = FlattenMlp(
            input_size=obs_dim ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim  ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim  ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
    policy = TanhGaussianPolicy(use_ego_attention=False,config=config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    trainer = SACTrainer(
        True,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


def test_ep(variant):
    dummy_env = make_env(variant['env'])  
    # self.observation_dim = np.prod(env.observation_space.shape or env.observation_space.n)
    # self.action_dim = np.prod(env.action_space.shape or env.action_space.n)
    obs_dim = np.prod(dummy_env.observation_space.shape or dummy_env.observation_space.n) 
    action_dim = np.prod(dummy_env.action_space.shape or dummy_env.action_space.n)
    print('观察维度：{} 动作维度： {}'.format(obs_dim,action_dim))
    expl_env = VectorEnv([lambda: make_env(variant['env']) for _ in range(variant['view_env_num'])])
    expl_env.seed(variant["seed"])
    expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: make_env(variant['env']) for _ in range(variant['view_env_num'])])
    eval_env.seed(variant["seed"])

    use_ego_attention = variant['use_ego_attention']
    M = variant['layer_size']

    config = {'type': 'EgoAttentionNetwork', 'layers': [128, 128], 
                        'embedding_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False, 'in': 7}, 
                        'others_embedding_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False, 'in': 7}, 
                        'self_attention_layer': None, 
                        'attention_layer': {'type': 'EgoAttention', 'feature_size': 64, 'heads': 2}, 
                        'output_layer': {'type': 'MultiLayerPerceptron', 'layers': [64, 64], 'reshape': False}, 'in': 105, 'out': 5, 
                        'presence_feature_idx': 0}
    if use_ego_attention:
        
        
        qf1 = models.EgoAttentionNetwork(config).cuda()
        qf2 = models.EgoAttentionNetwork(config).cuda()
        target_qf1 = models.EgoAttentionNetwork(config).cuda()
        target_qf2 = models.EgoAttentionNetwork(config).cuda()

    else:
    
        qf1 = FlattenMlp(
            input_size=obs_dim ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim  ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim  ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim ,
            output_size=action_dim,
            hidden_sizes=[M, M],
        )
    
    get_parameter_number(qf1)
    
    policy = TanhGaussianPolicy(use_ego_attention=False,config=config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    
    #params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/data/sac-roundabout-normal-sac/sac_roundabout_normal-sac_2022_03_04_16_41_57_0000--s-0/params.pkl')
    # params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/data/sac-roundabout-3.26-sac-roundabout-all/sac_roundabout_3.26-sac-roundabout-all_2022_03_26_16_30_03_0000--s-1/params.pkl')
    
    # params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/模型/sac/normal/params.pkl')
    params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/模型/sac/ego/params.pkl')
    #params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/模型/sac/liner/params.pkl')
    # params = torch.load('/home/linux/wzr_program_manager/RL/RL_lib/distributional-sac-master/模型/sac/all/params.pkl')
    qf1.load_state_dict(params['trainer/qf1'])
    qf2.load_state_dict(params['trainer/qf2'])
    target_qf1.load_state_dict(params['trainer/target_qf1'])
    target_qf2.load_state_dict(params['trainer/target_qf2'])
    policy.load_state_dict(params['trainer/policy'])
    
    

    
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )
    trainer = SACTrainer(
        False,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )


    algorithm.to(ptu.device)
    algorithm._test(n_epoch=100, render=0.5)

def get_parameter_number(net):
    total_num = sum(p.numel()for p in net.parameters())
    #train_num = sum(p.numel()for p in net.parameters() if p.requires_grad)
    print({'Total':total_num})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Soft Actor Critic')
    parser.add_argument('--config', type=str, default="configs/sac-normal/highway_roundabout.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = args.seed
    log_prefix = "_".join(["sac", variant["env"][:-3].lower(), str(variant["version"])])
    
    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)
    set_seed(args.seed)
    
    
    test = 1
    if test:
        test_ep(variant)
    else:
        setup_logger(log_prefix, variant=variant, seed=args.seed)
        experiment(variant)
        



