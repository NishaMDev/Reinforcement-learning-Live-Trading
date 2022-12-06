
# Reinforcement-learning-Live-Trading

# Project Summary - 

Trading has become an important sector for investors to invest in. 

In order to earn profit from stocks or cryptocurrency,  the investor has to make sure to dedicate enough time by checking the market rates for these stocks. investors cannot dedicate 24 hours to monitoring the market. This reduces their chances of doing a profitable transaction at any given point of time. 

Various aspects that restraining the efficacy of humans in trading in several ways. 
The reaction time of the investors. 
Price fluctuation


# Dataset

Dataset is downloaded for each stocks - AAPL, GME. 
 Link - https://www.marketwatch.com/investing/stock/aapl/download-data?startDate=11/16/2021&endDate=11/16/2022

# Algorithm

<li>PPO - The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).</li>
<li>A2C - A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C). It uses multiple workers to avoid the use of a replay buffer.</li>
<li>SAC - Soft Actor Critic (SAC) Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.</li>

# Stock-Trading-Environment

A custom OpenAI gym environment for simulating stock trades on historical price data.

Steps to run -

      For PLOT - python stocktrade.py -a 'SAC' -t 100
      For LIVE - python stock_live.py -a 'PPO' -t 100000

you can provide any algo [-a]: PPO / A2C / SAC & [-t]: timesteps

# Colab Link 

https://colab.research.google.com/drive/1OdU08AEvQBDxIRr2YsxV1j1CPWrCBbR2?usp=sharing


References:-

Medium article: https://medium.com/@adamjking3/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

