"""
Final Project: Stock Trading -Reinforcement Learning
Author: Nisha Mohan Devadiga
        Akanksha Rawat
        Karishma Kuria
"""
import argparse
import sys
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, DDPG , SAC

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.monitor   import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
from env.stock_trading_env import StockTradingEnv

TOTAL_TIMESTEPS = 200000

if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Stock Trade Inference')
    parser.add_argument('-a', "--algorithm", type=str, required=True,
                        default="PPO", help="mention algorithm  - PPO / A2C / DDPG / SAC")
    args = parser.parse_args()
    algo = args.algorithm
    config = {
         "policy_type": "MlpPolicy",
         "total_timesteps": TOTAL_TIMESTEPS,
         "learning_rate": 0.01,
         "momentum" : 0.2,
         "env_name": "StockTradingEnv",
        }
    
    # Initialize wandb
    run = wandb.init(
          project="StockTrading",
          config=config,
          sync_tensorboard=True,  # auto-upload StockTrading's tensorboard metrics
          monitor_gym=True,  # auto-upload the videos of agents playing the game
          save_code=True,  # optional
        )

    # Load data
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: Monitor(StockTradingEnv(df)), n_envs=10])
    
    # Instantiate the agent model
    if algo == "PPO":
        model = PPO("MlpPolicy", env, tensorboard_log=f"runs", verbose=1)
    elif algo == "A2C":
        model = A2C("MlpPolicy", env, tensorboard_log=f"runs", verbose=1)
    elif algo == "DDPG":
        model = DDPG("MlpPolicy", env, tensorboard_log=f"runs", verbose=1)
    elif algo == "SAC":
        model  = SAC("MlpPolicy", env, tensorboard_log=f"runs", verbose=1)
    else:
        print("Program Terminated. Please enter valid algorithm - PPO / A2C / DDPG / SAC")
        sys.exit()

    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path='./model_checkpoints/')
    eval_callback = EvalCallback(env,
                                 best_model_save_path=f"./model_checkpoints/best_model/",
                                 log_path="./logs/results",
                                 eval_freq=1000,verbose=1)
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
                    # gradient_save_freq=1000
        verbose=1,
        )
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback,wandb_callback])
    
    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS,callback=callback)
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
    obs = env.reset()

    for i in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='file')
    env.render(mode='static')
    run.finish()