import re
import os
import pandas as pd
import numpy as np
import warnings

DATA_PATH = './data/'
INTERVALS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
IMG_SIZE = 12
DROP_ROWS = 48

symbol_dict = {'cell': 'Celltrion',
               'hmotor': 'HyundaiMotor',
               'naver': 'NAVER',
               'kakao': 'Kakao',
               'lgchem': 'LGChemical',
               'lghnh': 'LGH&H',
               'bio': 'SamsungBiologics',
               'samsung1': 'SamsungElectronics',
               'samsung2': 'SamsungElectronics2',
               'sdi': 'SamsungSDI',
               'sk': 'SKhynix',
               'kospi': 'KOSPI'}


def symbol_to_path_exist(symbol, base_dir=DATA_PATH):
    return os.path.join(base_dir, '{}.csv'.format(str(symbol_dict[symbol])+'_preprocessed'))


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_width, img_height))
    return x_temp


def min_max_scaling(df):
    summary = df.describe()
    min_max = []
    for col in summary.columns:
        min_col = summary[col]['min']
        max_col = summary[col]['max']
        min_max.append((min_col, max_col))
        df[col] = df[col].apply(lambda x: (x - min_col)/(max_col - min_col))

    return min_max


def symbol_to_path(symbol, base_dir="data/"):
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))


def EMA_helper(series, period):
    w = 2 / float(period + 1)
    ema_top = series[period - 1]
    ema_bottom = 1
    for idx in range(1, period):
        ema_top += ((1-w)**idx) * series[period - 1 - idx]
        ema_bottom += (1 - w)**idx
    ema = ema_top / ema_bottom
    return ema


def return_EMA(series, period):
    emas = [EMA_helper(series[idx - period + 1 : idx + 1], period) for idx in range(period -1 , len(series))]
    emas = fill_for_noncomputable_vals(series, emas)
    return emas


def return_SMA(series, period):
    sma = [np.mean(series[idx-(period-1):idx+1]) for idx in range(0, len(series))]
    sma = fill_for_noncomputable_vals(series, sma)
    return sma


def typical_price(close, high, low):
    tp = [(high[idx] + low[idx]+ close[idx])/ 3 for idx in range(0, len(close))]
    return np.array(tp)


def money_flow(close, high, low, volume):
    mf = volume * typical_price(close, high, low)
    return mf


def fill_for_noncomputable_vals(input_data, result_data):
    non_computable_values = np.repeat(
        np.nan, len(input_data) - len(result_data)
    )
    filled_result_data = np.append(non_computable_values, result_data)
    return filled_result_data


def make_features_cnn(trade_symbols, start_date, end_date, is_training):
    """
    There are some modifications. However, They might not violate the rules
    We use ith day's data to select action for i+1th day's action
    Minmax scaling is conducted using the data except ones for test
    We use the minimum and maximum from train set to conduct scaling for test data

    :param trade_symbols: list of symbols that we're going to trade
    :return:
        For training :list of open prices and close prices of the companies and features in
        [:-test_days], [:-test_days], [:-test_days] respectively
        For test : list of open prices and close prices of the companies and features in
        [-test_days:], [-test_days:], [-test_days-1:-1] respectively
    """
    open_prices_train = []
    open_prices_test = []
    close_prices_train = []
    close_prices_test = []
    features_train = []
    features_test = []
    # DO NOT CHANGE
    test_days = 10

    for company in trade_symbols:
        if os.path.exists(symbol_to_path_exist(company)):
            print('preprocessed data for {} is already exists'.format(str(symbol_dict[company])))
            table = pd.read_csv(symbol_to_path_exist(company), index_col=0, parse_dates=True)
        else:
            trade_company = FeatureGenerator(symbol_dict[company], start_date, end_date)
            trade_company.generate_data(INTERVALS)
            table = trade_company.get_data()
            table = table.iloc[DROP_ROWS:]
            table.to_csv(symbol_to_path_exist(company))
        open_price_train = table['open'][:-test_days]
        open_price_test = table['open'][-test_days:]
        close_price_train = table['close'][:-test_days]
        close_price_test = table['close'][-test_days:]

        open_prices_train.append(open_price_train)
        open_prices_test.append(open_price_test)
        close_prices_train.append(close_price_train)
        close_prices_test.append(close_price_test)

        feature_test = table[table.columns[5:]].iloc[-test_days-1:-1]
        feature_train = table[table.columns[5:]].iloc[:-test_days]  # except data for test before scaling
        min_max = min_max_scaling(feature_train)
        feature_columns = list(feature_test.columns)
        for i in range(len(min_max)):
            feature_test[feature_columns[i]] = feature_test[feature_columns[i]].apply(
                lambda x: (x-min_max[i][0])/(min_max[i][1] - min_max[i][0]))
        feature_train = np.asarray(feature_train)
        feature_test = np.asarray(feature_test)

        feature_train = reshape_as_image(feature_train, IMG_SIZE, IMG_SIZE)
        feature_test = reshape_as_image(feature_test, IMG_SIZE, IMG_SIZE)
        features_train.append(feature_train)
        features_test.append(feature_test)

    # DO NOT CHANGE
    open_prices_train = np.asarray(open_prices_train)
    open_prices_test = np.asarray(open_prices_test)
    close_prices_train = np.asarray(close_prices_train)
    close_prices_test = np.asarray(close_prices_test)

    if not is_training:
        return open_prices_test, close_prices_test, features_test
    else:
        return open_prices_train, close_prices_train, features_train


def make_features_lstm(trade_symbols, start_date, end_date, is_training):
    """
    There are some modifications. However, They might not violate the rules
    We use ith day's data to select action for i+1th day's action
    Minmax scaling is conducted using the data except ones for test
    We use the minimum and maximum from train set to conduct scaling for test data

    :param trade_symbols: list of symbols that we're going to trade
    :return:
        For training :list of open prices and close prices of the companies and features in
        [:-test_days], [:-test_days], [:-test_days] respectively
        For test : list of open prices and close prices of the companies and features in
        [-test_days:], [-test_days:], [-test_days-1:-1] respectively
    """
    open_prices_train = []
    open_prices_test = []
    close_prices_train = []
    close_prices_test = []
    features_train = []
    features_test = []
    # DO NOT CHANGE
    test_days = 10
    needed_index = [0, 1, 2, 3, 4, 5, 25, 33, 45, 57, 75, 81, 100, 107, 124, 136, 141]

    for company in trade_symbols:
        if os.path.exists(symbol_to_path_exist(company)):
            print('preprocessed data for {} is already exists'.format(str(symbol_dict[company])))
            table = pd.read_csv(symbol_to_path_exist(company), index_col=0, parse_dates=True)
        else:
            trade_company = FeatureGenerator(symbol_dict[company], start_date, end_date)
            trade_company.generate_data(INTERVALS)
            table = trade_company.get_data()
            table = table.iloc[DROP_ROWS:]
            table.to_csv(symbol_to_path_exist(company))
        open_price_train = table['open'][:-test_days]
        open_price_test = table['open'][-test_days:]
        close_price_train = table['close'][:-test_days]
        close_price_test = table['close'][-test_days:]

        open_prices_train.append(open_price_train)
        open_prices_test.append(open_price_test)
        close_prices_train.append(close_price_train)
        close_prices_test.append(close  _price_test)

        feature_test = table[table.columns[needed_index]].iloc[-test_days-21:-1]
        feature_train = table[table.columns[needed_index]].iloc[:-test_days]  # except data for test before scaling
        min_max = min_max_scaling(feature_train)
        feature_columns = list(feature_test.columns)
        for i in range(len(min_max)):
            feature_test[feature_columns[i]] = feature_test[feature_columns[i]].apply(
                lambda x: (x-min_max[i][0])/(min_max[i][1] - min_max[i][0]))
        feature_train = np.asarray(feature_train)
        feature_test = np.asarray(feature_test)

        features_train.append(feature_train)
        features_test.append(feature_test)

    # DO NOT CHANGE
    open_prices_train = np.asarray(open_prices_train)
    open_prices_test = np.asarray(open_prices_test)
    close_prices_train = np.asarray(close_prices_train)
    close_prices_test = np.asarray(close_prices_test)

    if not is_training:
        return open_prices_test, close_prices_test, features_test
    else:
        return open_prices_train, close_prices_train, features_train


'''
* Example of data generation
* kakao = FeatureGenerator(kakao, start_date, end_date)
* kakao.generate_data(intervals)
* data_frame = kakao.get_data()
'''

'''
Functions for technical indicators 
Reference : DarkKnight1991's github, pyti
* Momentum indicator : RSI_smooth, williamR, WMA, EMA, SMA, HMA, MFI, CMO, ROC, TEMA
* Trend indicator : CCI, DPO
'''
class FeatureGenerator:

    def __init__(self, symbol, start_date, end_date):
        dates = pd.date_range(start_date, end_date)
        self.df = pd.DataFrame(index=dates)
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', parse_dates=True)
        self.df = self.df.join(df_temp)
        self.df = self.df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                                          'Volume': 'volume'})
        self.symbol = symbol

    # TODO : cleaning or filling missing value
        self.df = self.df.dropna()

    def generate_data(self, intervals):
        self.get_RSI_smooth('close', intervals)
        self.get_williamR('close', intervals)
        self.get_WMA('close', intervals)
        self.get_EMA('close', intervals)
        self.get_SMA('close', intervals)
        self.get_HMA('close', intervals)
        self.get_MFI(intervals)
        self.get_CMO('close', intervals)
        self.get_ROC('close', intervals)
        self.get_TEMA('close', intervals)
        self.get_CCI(intervals)
        self.get_DPO('close', intervals)

    def get_data(self):
        return self.df

    def get_RSI_smooth(self, col_name, intervals):
        """
        Momentum indicator
        """
        prev_avg_gain = np.inf
        prev_avg_loss = np.inf
        rolling_count = 0

        def calculate_RSI(series, period):
            nonlocal prev_avg_gain
            nonlocal prev_avg_loss
            nonlocal rolling_count

            curr_gains = series.where(series >=0, 0)
            curr_losses = np.abs(series.where(series < 0, 0))
            avg_gain = curr_gains.sum() / period
            avg_loss = curr_losses.sum() / period
            rsi = -1

            if rolling_count == 0:
                # first RSI calculation
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
            else:
                rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                         (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))
            rolling_count += 1
            prev_avg_gain = avg_gain
            prev_avg_loss = avg_loss
            return rsi

        diff = self.df[col_name].diff()[1:]
        for period in intervals:
            self.df['rsi_' + str(period)] = np.nan
            rolling_count = 0
            res = diff.rolling(period).apply(calculate_RSI, args=(period,), raw=False)
            self.df['rsi_' + str(period)][1:] = res

        print("Calculation of RSI_smooth is Done")

    def get_williamR(self, col_name, intervals):
        """
        Momentum indicator
        """
        def wr(high, low, close, period):
            top = high.rolling(period).max()
            bottom = low.rolling(period).min()
            will = (close - top)*100/(top-bottom)

            return will

        for period in intervals:
            self.df["wr_" + str(period)] = wr(self.df['high'], self.df['low'], self.df['close'], period)

        print("Calculation of WilliamR is Done")

    def get_WMA(self, col_name, intervals, hma_step=0):
        """
        Momentum indicator
        """
        def wavg(rolling_prices, period):
            weights = pd.Series(range(1, period + 1))
            return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

        temp_col_count_dict = {}
        for i in intervals:
            res = self.df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
            if hma_step == 0:
                self.df['wma_' + str(i)] = res
            elif hma_step == 1:
                if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                    temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
                else:
                    temp_col_count_dict['hma_wma_' + str(i)] = 0
                self.df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
            elif hma_step == 3:
                expr = r"^hma_[0-9]{1}"
                columns = list(self.df.columns)
                self.df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

        if hma_step == 0:
            print("Calculation of WMA is Done")

    def get_EMA(self, col_name, intervals):
        """
        Momentum indicator
        """
        series = self.df[col_name]
        for period in intervals:
            emas = [EMA_helper(series[idx - period + 1 : idx + 1], period) for idx in range(period - 1, len(series))]
            emas = fill_for_noncomputable_vals(series, emas)
            self.df["ema_" + str(period)] = emas

        print("Calculation of EMA is Done")

    def get_SMA(self, col_name, intervals):
        """
        Momentum indicator
        """
        series = self.df[col_name]
        for period in intervals:
            sma = [np.mean(series[idx-(period-1):idx+1]) for idx in range(0, len(series))]
            sma = fill_for_noncomputable_vals(series, sma)
            self.df["sma_" + str(period)] = sma

        print("Calculation of SMA is Done")

    def get_MFI(self, intervals):
        """
        Momentum indicator
        """
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']
        for period in intervals:
            mf = money_flow(close, high, low, volume)
            tp = typical_price(close, high, low)
            flow = [tp[idx] > tp[idx-1] for idx in range(1, len(tp))]
            pf = [mf[idx] if flow[idx] else 0 for idx in range(0, len(flow))]
            nf = [mf[idx] if not flow[idx] else 0 for idx in range(0, len(flow))]

            pmf = [sum(pf[idx+1-period:idx+1]) for idx in range(period-1, len(pf))]
            nmf = [sum(nf[idx+1-period:idx+1]) for idx in range(period-1, len(nf))]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                money_ratio = np.array(pmf) / np.array(nmf)

            mfi = 100 - (100 / (1 + money_ratio))
            mfi = fill_for_noncomputable_vals(close, mfi)
            self.df["mfi_" + str(period)] = mfi

        print("Calculating MFI is Done")

    def get_HMA(self, col_name, intervals):
        """
        Momentum indicator
        """
        expr = r"^wma_.*"

        if len(list(filter(re.compile(expr).search, list(self.df.columns)))) > 0:
            print("WMA calculated already. Proceed with HMA")
        else:
            print("Need WMA first...")
            self.get_WMA(col_name, intervals)

        intervals_half = np.round([i / 2 for i in intervals]).astype(int)

        self.get_WMA(col_name, intervals_half, 1)

        columns = list(self.df.columns)
        expr = r"^hma_wma.*"
        hma_wma_cols = list(filter(re.compile(expr).search, columns))
        rest_cols = [x for x in columns if x not in hma_wma_cols]
        expr = r"^wma.*"
        wma_cols = list(filter(re.compile(expr).search, rest_cols))

        self.df[hma_wma_cols] = self.df[hma_wma_cols].sub(self.df[wma_cols].values, fill_value=0)

        intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
        for i, col in enumerate(hma_wma_cols):
            self.get_WMA(col, [intervals_sqrt[i]], 3)
        self.df.drop(columns=hma_wma_cols, inplace=True)
        print("Calculation of HMA is Done")

    def get_CMO(self, col_name, intervals):
        """
        Momentum indicator
        """

        def calculate_CMO(series, period):
            # num_gains = (series >= 0).sum()
            # num_losses = (series < 0).sum()
            sum_gains = series[series >= 0].sum()
            sum_losses = np.abs(series[series < 0].sum())
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            return np.round(cmo, 3)

        diff = self.df[col_name].diff()[1:]  # skip na
        for period in intervals:
            self.df['cmo_' + str(period)] = np.nan
            res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
            self.df['cmo_' + str(period)][1:] = res

        print("Calculation of CMO is Done")

    def get_ROC(self, col_name, intervals):
        """
        Momentum indicator
        """

        def calculate_roc(series, period):
            return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

        for period in intervals:
            self.df['roc_' + str(period)] = np.nan
            # for 12 day period, 13th day price - 1st day price
            res = self.df['close'].rolling(period + 1).apply(calculate_roc, args=(period,), raw=False)
            # print(len(df), len(df[period:]), len(res))
            self.df['roc_' + str(period)] = res

        print("Calculation of ROC is Done")

    def get_TEMA(self, col_name, intervals):
        """
        Momentum indicator
        """
        series = self.df[col_name]
        for period in intervals:
            tema = ((3 * return_EMA(series, period) - (3 * return_EMA(return_EMA(series, period), period))) +
                    return_EMA(return_EMA(return_EMA(series, period), period), period)
                    )
            self.df["tema_" + str(period)] = tema

        print("Calculation of TEMA is Done")

    def get_CCI(self, intervals):
        """
        Trend indicator
        """
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        for period in intervals:
            tp = typical_price(close, high, low)
            cci = ((tp - return_SMA(tp, period)) /
                   (0.015 * np.mean(np.absolute(tp - np.mean(tp)))))
            self.df['cci_'+str(period)] = cci

        print("Calculation of CCI is Done")

    def get_DPO(self, col_name, intervals):
        """
        Trend indicator
        """
        series = self.df[col_name]
        for period in intervals:
            dpo = [series[idx] - np.mean(series[idx + 1 - (int(period / 2) + 1):idx + 1]) for idx in
                   range(period - 1, len(series))]
            dpo = fill_for_noncomputable_vals(series, dpo)
            self.df['dpo_'+str(period)] = dpo

        print("Calculation of DPO is Done")


if __name__ == "__main__":
    trade_company_list = ['samsung1', 'cell', 'kakao']
    open, close, feature = make_features_cnn(trade_company_list, '2010-01-01', '2020-05-19', True)
    print(open[0], '\n')
    print(close[0], '\n')
    for i in feature:
        print(i.shape)
    print(feature[0][0:20])