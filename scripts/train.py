import argparse
import os

import ray
from ray import tune

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

from ray.rllib.utils.framework import try_import_tf, try_import_torch

import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np
import random

from dc_environment.envs.packing_env import PackingEnv, ParametricActionWrapper, ParametricActionsModel

import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from gnn import *

tf1, tf, tfv = try_import_tf()
tf.get_logger().setLevel('INFO')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--stop-reward", type=float, default=10)
    parser.add_argument("--as-test", action="store_true")
    parser.add_argument("--training_iterations", type=int, default=100)
    parser.add_argument("--checkpoint_frequency", type=int, default=1)

    parser.add_argument('--topology', nargs='?', type=str, default='small')

    parser.add_argument('--dataset', nargs='?', type=str, default='uniform')
    parser.add_argument('--episode_length', nargs='?', default='32',type=int)
    parser.add_argument('--save_dir', nargs='?', type=str, default='../../mlfs_github_tests')

    parser.add_argument("--number_of_trains", type=int, default=1)
    args = parser.parse_args()

    request_type = 'SingleResourceRequest'
    restore = None

    args = parser.parse_args()
    ray.shutdown()
    ray.init(temp_dir='/tmp/uceezs0_ray_tmp')

    register_env("pa_network", lambda config: ParametricActionWrapper(config))
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    cfg = {}

    FEATURES = {'node':['node','mem'],
                'link':['ports']}
    NUM_FEATURES = 2

    config = dict({
            "env": "pa_network",
            "env_config": {
                'restore':False,
                'new_save':False,
                'sanity_check':False,
                'num_init_requests':args.episode_length,
                'network_dir':os.path.abspath('../topologies/{}'.format(args.topology)),
                'features':FEATURES,
                'request_type':request_type,
                'rnd_seed':0,
                'seed_on_reset':False,
                'lb_route_weighting':False,
            },
            "model": {
                "custom_model": "pa_model",
                "custom_model_config": {
                    'features':FEATURES,
                    'graph_dir':os.path.abspath('../topologies/{}'.format(args.topology)),
                    'use_gnn':True,
                    'agg_type':'MeanPool2',
                    'agg_dim':16,
                    'num_mp_stages':3,
                    'obs_emb_dim':8,
                    'num_features':NUM_FEATURES,
                    'embedding_save_dir':None,
                    'top_k':None,
                },
                "fcnet_hiddens": [8],
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "2")),
            "num_workers": 1,
            "framework":"tfe",
            "eager_tracing":True,
            "vf_share_layers": False,
            "lr": tune.grid_search([5e-3]*args.number_of_trains),
            "horizon":None,
            "rollout_fragment_length": 2048,      
            "train_batch_size":tune.grid_search([2048]),
            "sgd_minibatch_size":tune.grid_search([256]),
            "log_level":"ERROR",
            "batch_mode": "complete_episodes",
        },
        **cfg)

    stop = {
        "training_iteration": args.training_iterations,
    }
    print('args: {}'.format(args))
    results = tune.run(args.run, 
                        local_dir='{}/train_sanity_check'.format(args.save_dir),
                        checkpoint_freq=args.checkpoint_frequency,
                        stop=stop,
                        config=config,
                        verbose=2,
                        restore=restore)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()