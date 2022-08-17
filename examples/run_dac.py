import argparse

import tensorflow as tf

from rllab.envs.normalized_env import normalize

from dac.algos import DAC
from dac.envs import (
    GymEnv,
    GymEnvDelayed,
)

from dac.misc.instrument import run_sac_experiment
from dac.misc.utils import timestamp, unflatten
from dac.policies import GaussianPolicy, UniformPolicy
from dac.misc.sampler import SimpleSampler
from dac.replay_buffers import SimpleReplayBuffer
from dac.value_functions import NNQFunction, NNVFunction, NNRFunction, NNAFunction
from examples.variants import parse_domain_and_task, get_variants, get_variants_sparse

DELAY_CONST = 20
ENVIRONMENTS = {
    'ant': {
        'default': lambda: GymEnv('Ant-v1'),
        'delayed': lambda: GymEnvDelayed('Ant-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseAnt-v1'),
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1'),
        'delayed': lambda: GymEnvDelayed('Hopper-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseHopper-v1'),
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1'),
        'delayed': lambda: GymEnvDelayed('HalfCheetah-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseHalfCheetah-v1'),
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1'),
        'delayed': lambda: GymEnvDelayed('Walker2d-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseWalker2d-v1'),
    },
    'humanoid-standup-gym': {
        'default': lambda: GymEnv('HumanoidStandup-v1'),
    },
}

AVAIlABLE_ENVS=['half-cheetah','hopper','ant','walker','humanoid-standup-gym']
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='sparse')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
                        default='gaussian')
    parser.add_argument('--env', type=str, default='half-cheetah')
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--alpha_adapt', type=int, default=1)
    parser.add_argument('--fixed_alpha', type=float, default=0.5)
    parser.add_argument('--ctrl_coef', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args

def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    env = normalize(ENVIRONMENTS[domain][task](**env_params))
    eval_env = None

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(domain=domain, task=task, **sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    rf = NNRFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='rf')
    if algorithm_params['alpha_adapt']:
        af = NNAFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='af')
    else:
        af = None
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    policy = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M,M),
        reparameterize=True,
        reg=1e-3,
    )

    algorithm = DAC(
        base_kwargs=base_kwargs,
        env=env,
        eval_env=eval_env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        rf=rf,
        af=af,
        lr=algorithm_params['lr'],
        scale_reward=algorithm_params['scale_reward'], # reward scale = 1/beta
        ctrl_coef=algorithm_params['ctrl_coef'],
        fixed_alpha=algorithm_params['fixed_alpha'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']
        algo_params = variant['algorithm_params']
        variant['algorithm_params']['ctrl_coef'] = args.ctrl_coef
        variant['algorithm_params']['alpha_adapt'] = args.alpha_adapt
        variant['algorithm_params']['fixed_alpha'] = args.fixed_alpha

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl'],
        )


def main():
    args = parse_args()

    domain, task = args.env, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)
    if args.task == 'sparse':
        variant_generator = get_variants_sparse(domain=domain, task=task, policy=args.policy, seed = args.seed, gamma = args.gamma)
    else:
        variant_generator = get_variants(domain=domain, task=task, policy=args.policy, seed = args.seed, gamma = args.gamma)
    launch_experiments(variant_generator, args)

if __name__ == '__main__':
    main()
