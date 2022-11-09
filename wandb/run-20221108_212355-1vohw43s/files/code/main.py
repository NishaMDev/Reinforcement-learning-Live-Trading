import gym
import json
import datetime as dt
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C, DDPG , SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import matplotlib.dates as mpdates
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback , StopTrainingOnNoModelImprovement
import wandb
from wandb.integration.sb3 import WandbCallback
from env.stock_trading_env import StockTradingEnv
import pandas as pd
import argparse

TOTAL_TIMESTEPS = 200000


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Stock Trade Inference')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to input image or video file. Skip this argument to pick default file.')
    parser.add_argument('-a', "--algorithm", type=str, required=True,
                        default="PPO", help="mention algorithm needs to be performed - PPO / A2C / DDPG / SAC")
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
          sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
          monitor_gym=True,  # auto-upload the videos of agents playing the game
          save_code=True,  # optional
        )
    
    # artifact = wandb.Artifact(name='Stocks', type='dataset')
    # artifact.add_file(local_path='path/file.format')
        
    # Load data 
    df = pd.read_csv('./data/MSFT.csv')
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    env = Monitor(env) 

    # Instantiate the agent
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
        exit()
                      
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path='./model_checkpoints/{}/'.format(algo))
    
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path="./model_checkpoints/best_model/{}/".format(algo),
                                 log_path="./logs/results", callback_after_eval=stop_train_callback, eval_freq=1000,verbose=1)

    wandb_callback = WandbCallback(
            model_save_path=f"models/{algo}/{run.id}",
            gradient_save_freq=1000,
            verbose=1,
        )

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback,wandb_callback])

    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS,callback=callback)

# model.save("ppo_stock_trading")
# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_stock_trading",print_system_info=True))

    obs = env.reset()
    #   for i in range(2000):
    for i in range(len(df['Date'])):    
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # env.render(title='MSFT')
        env.render(mode='live')
        
    run.finish()   


