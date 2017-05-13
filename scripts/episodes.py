import pandas as pd
import numpy as np

class episodes(object):
    def __init__(self, currency_path, index_path, equity_path, sep=';'):
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

        self.df = df
        self.index = 0
        self.prev_date = df.index[0]

    def read_csv(self, path, sep=';'):
        dtypes = {'Date': str, 'Time': str}
        df = pd.read_csv(path, sep=sep, header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], dtype=dtypes)
        dtime = df.Date + ' ' + df.Time
        df.index = pd.to_datetime(dtime)
        return df

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

    def next(self):
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

