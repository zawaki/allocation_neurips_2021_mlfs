from gym.envs.registration import register

register(
    id='dc-environment-v0',
    entry_point='dc_environment.envs:PackingEnv',
)