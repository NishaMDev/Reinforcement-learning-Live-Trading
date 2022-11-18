"""
name - stock_trading_Env.py
"""

import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from enum import Enum

plt.style.use('fivethirtyeight')

from render.stock_trading_graph import StockTradingGraph

MAX_ACCOUNT_BALANCE = 1000000
MAX_NUM_SHARES = 200000
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 2000
INITIAL_ACCOUNT_BALANCE = 10000
LOOKBACK_WINDOW_SIZE = 40

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none','static']}
    visualization = None

    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
                # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.trades = []
        self.trades_new = []


        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)
            # low=0, high=1, shape=(1O, LOOKBACK_WINDOW_SIZE + 1), dtype=np.float16)

    
    def _next_observation(self):
        frame = np.zeros((5, LOOKBACK_WINDOW_SIZE + 1))

        # Get the stock data points for the last 5 days and scale to between 0-1
        np.put(frame, [0, 4], [
            self.stock_data.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Open'].values / MAX_SHARE_PRICE,
            self.stock_data.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'High'].values / MAX_SHARE_PRICE,
            self.stock_data.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Low'].values / MAX_SHARE_PRICE,
            self.stock_data.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Close'].values / MAX_SHARE_PRICE,
            self.stock_data.loc[self.current_step: self.current_step +
                        LOOKBACK_WINDOW_SIZE, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            [self.balance / MAX_ACCOUNT_BALANCE],
            [self.max_net_worth / MAX_ACCOUNT_BALANCE],
            [self.shares_held / MAX_NUM_SHARES],
            [self.cost_basis / MAX_SHARE_PRICE],
            [self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)],
        ], axis=1)

        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.stock_data.loc[self.current_step, "Open"],
            self.stock_data.loc[self.current_step, "Close"])
        

        action_type = action[0]
        amount = action[1]
        

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)            
            shares_bought = int(total_possible * amount)            
            prev_cost = self.cost_basis * self.shares_held            
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost            
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)        
            self.shares_held += shares_bought


            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"}) 

        elif action_type < 2:
            # Sell amount % of shares held

            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold            
            self.total_shares_sold += shares_sold            
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})

        self.net_worth = self.balance + self.shares_held * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
            

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier + self.current_step
        done = self.net_worth <= 0 or self.current_step >= len(
            self.stock_data.loc[:, 'Open'].values)

        obs = self._next_observation()

        return obs, reward, done, {}

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.trades = []
        self.trades_new = []
        return self._next_observation()

    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        self.trades_new.append({'Step_nbr': self.current_step,
                            'Balance': self.balance, 
                            'Shares_held': self.shares_held,
                            'Total_shares_sold': self.total_shares_sold,
                            'Avg_cost': self.cost_basis,
                            'Total_sales_value': self.total_sales_value,
                            'Net_worth': self.net_worth,
                            'Max_net_worth': self.max_net_worth,
                            'Profit': profit})
        

        
        file = open(filename, 'a')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(f'Avg cost for held shares: {self.cost_basis} \
            (Total sales value: {self.total_sales_value})\n')
        file.write(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()

    def _render_to_screen(self):
        trades_df = pd.DataFrame(self.trades_new)

        avg_profit = trades_df.Profit.mean(axis=0)
        max_profit = trades_df.Profit.max(axis=0)
        
        # plot the portfolio value over time
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        fig.tight_layout(pad=5)
             
        # Plot share hold and sold over each step.
        axes[0][0].set(ylabel = 'Shares over next days')
        axes[0][0].set(xlabel = 'Day number')
        axes[0][0].plot(trades_df['Step_nbr'], trades_df['Shares_held'], 'b-', label='Shares Held')
        axes[0][0].plot(trades_df['Step_nbr'], trades_df['Total_shares_sold'], 'g-', label='Total Shares Sold')
        axes[0][0].legend(loc="best", shadow=True, fancybox=True, framealpha =0.5)
            
        # Plot Profit over each step.
        axes[0][1].set(ylabel = 'Profit over next days')
        axes[0][1].set(xlabel = 'Day number')
        axes[0][1].plot(trades_df['Step_nbr'], trades_df['Profit'], 'r-', label='Profit')
        axes[0][1].axhline(y=avg_profit, ls='--',color='black', label='Average')
        axes[0][1].axhline(y=max_profit, ls='-',color='black', label='Max')
        axes[0][1].legend(loc="best", shadow=True, fancybox=True, framealpha =0.5)

        # Plot Net worth over each step.
        axes[1][0].set(ylabel = 'Net Worth next days')
        axes[1][0].set(xlabel = 'Day number')
        axes[1][0].plot(trades_df['Step_nbr'], trades_df['Net_worth'], 'b-', label='Net Worth')
        axes[1][0].plot(trades_df['Step_nbr'], trades_df['Max_net_worth'], 'g-', label='Max Net Worth')
        axes[1][0].legend(loc="best", shadow=True, fancybox=True, framealpha =0.5)
        
        # Plot Profit over each step.
        axes[1][1].set(ylabel = 'Total Sales value')
        axes[1][1].set(xlabel = 'Day number')
        axes[1][1].plot(trades_df['Step_nbr'], trades_df['Total_sales_value'], 'g-', label='Total Sales Value')
        axes[1][1].plot(trades_df['Step_nbr'], trades_df['Balance'], 'b-', label='Balance')
        axes[1][1].legend(loc="best", shadow=True, fancybox=True, framealpha =0.5)
        
        plt.legend()
        plt.savefig('result.png')
        plt.show()

    
    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization is None:
                self.visualization = StockTradingGraph(
                    self.stock_data, kwargs.get('title', None))

            if self.current_step > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.current_step,
                    self.net_worth,
                    self.trades,
                    window_size=LOOKBACK_WINDOW_SIZE)
                
        elif mode == 'static':
            self._render_to_screen()

    def close(self):
     
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None
