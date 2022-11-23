
"""
Final Project: Stock Trading -Reinforcement Learning
Author: Nisha Mohan Devadiga
        Akanksha Rawat
        Karishma Kuria.
"""
import argparse
import sys
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C , SAC

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
from env.stock_trading_env import StockTradingEnv


TOTAL_TIMESTEPS = 100000 # 1000000 for PPO, 100 for A2C , 100 for SAC 

class Stocktrade:
    
    def stocktrade(algo,timesteps,hparam ):
    
        config = {
             "policy_type": "MlpPolicy",
             "total_timesteps": timesteps,
             "learning_rate": 0.01,
              "momentum" : 0.2,
              "env_name": "stock-v0",
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
        env = DummyVecEnv([lambda: StockTradingEnv(df)])
         
         # env = make_vec_env([lambda: StockTradingEnv(df)],n_envs=10)
         # print("Check env: ",check_env(StockTradingEnv(df)))
         
         # Instantiate the agent model
        if algo == "PPO":
          if hparam == "T":
               model = PPO(config["policy_type"],env,gamma=0.80,learning_rate=0.000010,ent_coef=0.3, verbose=1, tensorboard_log=f"runs/{run.id}")
          else: 
               model = PPO(config["policy_type"],env, verbose=1, tensorboard_log=f"runs/{run.id}")
     
        elif algo == "A2C":
          if hparam == 'T':
             model = A2C(config["policy_type"],env,gamma=0.80 , learning_rate=0.000010 , ent_coef=0.3,verbose=1, tensorboard_log=f"runs/{run.id}")
          else:
             model = A2C(config["policy_type"],env, verbose=1, tensorboard_log=f"runs/{run.id}")
                
        elif algo == "SAC":
          policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))

          if hparam == 'T':
             model = SAC(config["policy_type"], env, gamma=0.80 , learning_rate=0.000010 , ent_coef=0.3, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log=f"runs/{run.id}" )
          else:
             model = SAC(config["policy_type"], env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log=f"runs/{run.id}" )

        else:
             print("Program Terminated. Please enter valid algorithm - PPO / A2C / SAC")
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
        model.learn(
             total_timesteps=config["total_timesteps"],
             callback=callback,
             )
         # print rewards
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
     
        obs = env.reset()
     
        for i in range(200):
             action, _states = model.predict(obs)
             obs, rewards, done, info = env.step(action)
             env.render(mode='file')
        env.render(mode='static')
        run.finish()


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Stock Trade Inference')
    parser.add_argument('-a', "--algorithm", type=str, required=True,
                        default="PPO", help="mention algorithm  - PPO / A2C / SAC")
    parser.add_argument('-t', "--timesteps", type=int, required=True,
                        default="PPO", help="enter timesteps")
    parser.add_argument('-h', "--hparam", type=str, required=True,
                        default="F", help="enter T/F")
    args = parser.parse_args()
    algo = args.algorithm
    timesteps = args.timesteps
    hparam = args.hparam
    print("hparam:", hparam)
    Stocktrade.stocktrade(algo,timesteps,hparam)
