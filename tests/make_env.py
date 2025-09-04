from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset

import sys
import os


def make_env(args, max_path_length):
    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)
    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100)
    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  # Only support pixel-based environments
        if args.env == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env == 'kitchen':
        local_lexa_path = os.path.abspath('./lexa')
        sys.path.insert(0, local_lexa_path)
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)
    
    elif args.env == 'ant_maze':
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)

    elif args.env == 'ant_maze_large':
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("antmaze-large-diverse-v0", random_init=False)


    elif args.env == 'lm':
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("maze2d-large-v1", random_init=False)
    
    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    return env



def make_env_wo_args(**kwargs):
    from argparse import Namespace
    args = Namespace(**kwargs)
    
    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=args.max_path_length,
            action_range=0.2,
        )
    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)
    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100)
    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  # Only support pixel-based environments
        if args.env == 'dmc_cheetah':
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_quadruped':
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_humanoid':
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env == 'kitchen':
        local_lexa_path = os.path.abspath('./lexa')
        sys.path.insert(0, local_lexa_path)
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder 
        env = MyKitchenEnv(log_per_goal=True)
    
    elif args.env == 'ant_maze':
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)

    elif args.env == 'lm':
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        env = MazeWrapper("maze2d-large-v1", random_init=False)
    
    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    return env


