import gym
import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.dates as mpdates

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('./data/MSFT.csv')
df = df.sort_values('Date')


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo_stock_trading")

obs = env.reset()
# for i in range(2000):
for i in range(len(df['Date'])):    
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render(title='MSFT')
    env.render(mode='live')
