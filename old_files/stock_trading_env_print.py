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

from render.stock_trading_graph import StockTradingGraph

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 2000
INITIAL_ACCOUNT_BALANCE = 10000
LOOKBACK_WINDOW_SIZE = 100

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
        
        print("self.action_space:",self.action_space)
        print("self.action_space.shape:",self.action_space.shape)
        

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)
        print("self.observation_space:",self.observation_space)
        print("self.observation_space.shape:",self.observation_space.shape)
        

    
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
        
        file1 = open('logger.txt', 'a')
        file1.write(f'Step: {self.current_step}\n')
        file1.write(f'current_price: {current_price}\n')
        action_type = action[0]
        amount = action[1]
        
        file1.write(f'amount: {amount}\n')
        file1.write(f'action type: {action_type}\n')
        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            file1.write('----------BUY----------\n')
            file1.write('total_possible = int(self.balance / current_price\n')
            file1.write(f'balance: {self.balance}\n')
            
            file1.write(f'current_price: {current_price}\n')
            file1.write(f'###total_possible: {total_possible}\n')
            
            shares_bought = int(total_possible * amount)
            file1.write('shares_bought = int(total_possible * amount\n')
            file1.write(f'amount: {amount}\n')
            file1.write(f'###shares_bought: {shares_bought}\n')
            
            prev_cost = self.cost_basis * self.shares_held
            file1.write(f'self.cost_basis: {self.cost_basis}\n')
            file1.write(f'self.shares_held:  {self.shares_held}\n')
            file1.write(f'###prev_cost: {prev_cost}\n')
            
            additional_cost = shares_bought * current_price
            file1.write('additional_cost = shares_bought * current_price\n')
            file1.write(f'###additional_cost: {additional_cost}\n')

            self.balance -= additional_cost
            file1.write('self.balance -= additional_cost\n')
            file1.write(f'###self.balance: {self.balance}\n')
            
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            file1.write('cost_basis = prev_cost + additional_cost) / (self.shares_held + shares_bought)\n')
            file1.write(f'###self.cost_basis: {self.cost_basis}\n')
            
            self.shares_held += shares_bought
            file1.write('self.shares_held += shares_bought\n')
            file1.write(f'###self.shares_held: {self.shares_held}\n')

            if shares_bought > 0:
                self.trades.append({'Step_nbr': self.current_step,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"})
            else:
                self.trades.append({'Step_nbr': self.current_step,
                                    'shares': 0, 'total': 0,
                                    'type': "no-trade"})    

        elif action_type < 2:
            # Sell amount % of shares held
            file1.write('----------SOLD----------\n')
            shares_sold = int(self.shares_held * amount)
            file1.write('shares_sold = int(self.shares_held * amount\n')
            file1.write(f'amount: {amount}\n')
            file1.write(f'shared_held: {self.shares_held}\n')
            file1.write(f'###shares_sold: {shares_sold}\n')
                
            self.balance += shares_sold * current_price
            file1.write('self.balance += shares_sold * current_price\n')
            file1.write(f'current_price: {current_price}\n')
            file1.write(f'###self.balance: {self.balance}\n')
            
            self.shares_held -= shares_sold
            file1.write('self.shares_held -= shares_sold\n')
            file1.write(f'####self.shares_held: {self.shares_held}\n')
            
            self.total_shares_sold += shares_sold
            file1.write('self.total_shares_sold += shares_sold\n')
            file1.write(f'###self.total_shares_sold: {self.total_shares_sold}\n')
            
            self.total_sales_value += shares_sold * current_price
            file1.write('self.total_sales_value += shares_sold * current_price\n')
            file1.write(f'###self.total_sales_value: {self.total_sales_value}\n')
            

            if shares_sold > 0:
                self.trades.append({'Step_nbr': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})
            else:
                    self.trades.append({'Step_nbr': self.current_step,
                                    'shares': 0, 'total': 0,
                                    'type': "no-trade"})
                    
        file1.write('----------END of action logic----------\n')
        self.net_worth = self.balance + self.shares_held * current_price
        file1.write('self.net_worth = self.balance + self.shares_held * current_price\n')
        file1.write(f'###self.net_worth: {self.net_worth}\n')    

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            file1.write('if self.net_worth > self.max_net_worth then self.max_net_worth = self.net_worth \
            ---> self.max_net_worth: ", self.max_net_worth\n')

        if self.shares_held == 0:
            self.cost_basis = 0
            file1.write('if self.shares_held == 0: self.cost_basis = 0 ---> self.cost_basis: ", self.cost_basis\n')
            
        file1.write('----------END----------\n')

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
        
        print("self.current_step: ", self.current_step)
        print("self.balance: ", self.balance)
        print("profit: ", profit)
        
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
        print(self.trades_new)
        print("length - ",len(self.trades_new))
        trades_df = pd.DataFrame(self.trades_new)
        print(trades_df.head())
        
        avg_profit = trades_df.Profit.mean(axis=0)
        print("avg_profit: ", avg_profit)
        
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
        # axes[0][1].plot(trades_df['Step_nbr'], trades_df['Avg_cost'], 'b-', label='Avg_cost')
        axes[0][1].axhline(y=avg_profit, ls='--',color='black', label='average')
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
