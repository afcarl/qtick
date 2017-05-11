import pandas as pd
import numpy as np
from copy import deepcopy

import argparse
import os

import qlearn
import state
import utils

class episodes(object):
    def __init__(self, df, action_num):
        self.df = df
        self.index = 0
        self.action_num = action_num
        self.prev_date = df.index[0]

    def start_over(self):
        self.index = 0
        self.prev_date = self.df.index[0]

    def reset(self):
        if self.index == len(self.df):
            return np.empty(0)

        r = self.df.iloc[self.index]
        obs = r.values
        self.index += 1
        return obs

    def step(self, action):
        if action < 0 or action >= self.action_num:
            print "invalid action: %d, must be in [0, %d)" % (action, self.action_num)
            exit(-1)

        if self.index == len(self.df):
            print "episode has been completed"
            exit(-1)

        done = False
        if self.index == len(self.df) - 1:
            done = True

        d = self.df.index[self.index]
        r = self.df.iloc[self.index]
        obs = r.values
        reward = 1
        self.index += 1

        if not done:
            d = self.df.index[self.index]
            if self.prev_date.day != d.day:
                done = True

        self.prev_date = d

        return obs, reward, done, None

    def current_time(self):
        index = self.index
        if self.index > len(self.df) - 1:
            index = -1

        return self.df.index[index]

    def current_date_str(self):
        d = self.current_time()
        return "%02d.%02d.%d" % (d.day, d.month, d.year)

class qtick(object):
    ACTION_BUY = 0
    ACTION_HOLD = 1
    ACTION_SELL = 2
    #COMISSION = 0.1 / 100.
    COMISSION = 0
    MONEY = 100000
    MAX_DEBT = -0.0

    def __init__(self, output_path, observation_num=1, checkpoint_path=None):
        self.observation_num = observation_num
        self.checkpoint_path = checkpoint_path

        # equity, currency, index prices normalized by the first value
        self.observation_shape = 1 + 1 + 1
        
        # buy, hold, sell
        self.action_num = 3

        self.current_state = state.state(self.observation_shape, self.observation_num)

        self.q = qlearn.qlearn((self.observation_shape*self.observation_num,), self.action_num, output_path)

    def save(self):
        self.q.save(self.checkpoint_path)
    def load(self, path):
        self.q.load(path)

    def new_state(self, obs):
        self.current_state.push_array(obs)
        self.current_state.complete()
        return deepcopy(self.current_state)

    def read_csv(self, path, sep=';'):
        dtypes = {'Date': str, 'Time': str}
        df = pd.read_csv(path, sep=sep, header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype=dtypes)
        dtime = df.Date + ' ' + df.Time
        df.index = pd.to_datetime(dtime)
        return df

    def run_episode(self):
        print "%s-%s: %d" % (self.current_df.index[0], self.current_df.index[-1], len(self.current_df))

    def a2s(self, action):
        if action == self.ACTION_BUY:
            return "BUY"
        if action == self.ACTION_HOLD:
            return "HOLD"
        if action == self.ACTION_SELL:
            return "SELL"

    def get_action_limit(self, s, comission_price, equity_price, money):
        a = self.ACTION_HOLD

        while True:
            a = self.q.get_action(s)
            if a == self.ACTION_BUY and comission_price > money:
                a = self.ACTION_HOLD
                continue

            if a == self.ACTION_SELL and equity_price <= self.MAX_DEBT:
                a = self.ACTION_HOLD
                continue

            break

        return a

    def get_last_price(self, s):
        return s.read()[-self.observation_shape]

    def train(self, currency_path, index_path, equity_path, sep=';'):
        cdf = self.read_csv(currency_path, sep=sep)
        idf = self.read_csv(index_path, sep=sep)
        edf = self.read_csv(equity_path, sep=sep)

        df = pd.DataFrame(index=edf.index)
        df.ix[:, 'Equity'] = edf.ix[:, 'Close']
        df.ix[:, 'Currency'] = cdf.ix[:, 'Close']
        df.ix[:, 'Index'] = idf.ix[:, 'Close']

        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(1.0)
        df /= df.iloc[0, :]

        e = episodes(df, self.action_num)

        num_episodes = 0

        num = 1000
        equity = 0
        money = self.MONEY

        episode_set_start = None
        episode_set_end = None

        while True:
            obs = e.reset()
            if obs.size == 0:
                episode_set_end = e.current_date_str()
                print "Saving and starting over, episodes: %d, period: %s-%s" % (num_episodes, episode_set_start, episode_set_end)
                self.save()

                e.start_over()
                continue

            episode_start = e.current_date_str()
            if not episode_set_start:
                episode_set_start = episode_start

            s = self.new_state(obs)
            actions = {}
            while True:
                price = self.get_last_price(s)
                cost = price * num
                comission_price = cost * (1. + self.COMISSION)
                equity_price = price * equity

                a = self.get_action_limit(s, comission_price, equity_price, money)

                prev = money + equity_price

                if a == self.ACTION_BUY:
                    money -= comission_price
                    equity += num
                if a == self.ACTION_SELL:
                    money += (1. - self.COMISSION) * equity_price
                    equity = 0
                    #money += (1. - self.COMISSION) * cost
                    #equity -= num
                if a == self.ACTION_HOLD:
                    pass

                obs, r, done, _ = e.step(a)
                sn = self.new_state(obs)
                
                new_equity_price = self.get_last_price(sn) * equity
                reward = money + new_equity_price - prev

                au = actions.get(a, 0)
                au += 1
                actions[a] = au

                if a != self.ACTION_HOLD:
                    #print "s: %s, obs: %s, action: %s, new equity: %d/%.2f, money: %.2f, total: %.2f->%.2f, reward: %.2f" % (
                    #    s, obs, self.a2s(a), equity, new_equity_price, money, prev, new_equity_price + money, reward)
                    pass

                self.q.history.append((s, a, reward, sn, done), 1)
                self.q.learn()

                s = sn

                if done:
                    break

            episode_end = e.current_date_str()

            print "Episode %d %s-%s completed: portfolio: %.2f, money: %.2f, equity: %d, actions: %s" % (
                    num_episodes, episode_start, episode_end, money + equity * self.get_last_price(s), money, equity, actions)
            num_episodes += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning ticks util')
    parser.add_argument('--equity', action='store', required=True)
    parser.add_argument('--currency', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--tf_output_path')
    parser.add_argument('--observation_num', action=utils.store_long, default=1)
    parser.add_argument('--checkpoint')

    args = parser.parse_args()

    q = qtick(args.tf_output_path, args.observation_num, args.checkpoint)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        q.load(args.checkpoint)


    q.train(args.currency, args.index, args.equity)
