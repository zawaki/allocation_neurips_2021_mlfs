from gym.envs.registration import register

register(
    id='dcn-env-v0',
    entry_point='dcn_env.envs:PackingEnv',
)