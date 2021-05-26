from gym.envs.registration import register
from .gym_env import GymEnv, GymEnvDelayed

register(
    id='SparseHalfCheetah-v1',
    entry_point='dac.envs.sparse_halfcheetah:SparseHalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SparseHopper-v1',
    entry_point='dac.envs.sparse_hopper:SparseHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SparseWalker2d-v1',
    max_episode_steps=1000,
    entry_point='dac.envs.sparse_walker2d:SparseWalker2dEnv',
)

register(
    id='SparseAnt-v1',
    entry_point='dac.envs.sparse_ant:SparseAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

