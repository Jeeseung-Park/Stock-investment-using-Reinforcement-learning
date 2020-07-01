from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import deque
import warnings

register_matplotlib_converters()

BATCH_SIZE = 20
COPY_STEP = 20
REPLAY_MEMORY_SIZE = 500000
TRAIN_SIZE = 2500


def do_action(action, budget, num_stocks, stock_price):
    if action == "Buy" and budget >= stock_price:
        n_buy = min(np.ceil(10. ** 7 / stock_price), budget // stock_price)
        num_stocks += n_buy
        budget -= stock_price * n_buy
    elif action == "Sell":
        n_sell = num_stocks
        num_stocks = 0
        budget += stock_price * n_sell
    else:
        num_stocks = num_stocks
        budget = budget
    return budget, num_stocks, action


def sample_memories(batch_size, replay_memory):
    indices=np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], []]
    for idx in indices:
        memory=replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols=[np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3]


def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features_cnn, features_lstm, epoch, replay_memory):
    action_count = [0] * len(policy.actions)
    action_seq = list()
    step = epoch * (len(open_prices) - 21)
    budget = initial_budget
    num_stocks = initial_num_stocks
    for t in range(20, len(open_prices) - 1):
        current_state_cnn = features_cnn[t]
        current_state_lstm = features_lstm[t-20 : t]
        current_state = [current_state_cnn, current_state_lstm]
        stock_price = float(open_prices[t])
        current_portfolio = budget + num_stocks * stock_price
        action = policy.select_action(current_state, step, t)[0]

        budget, num_stocks, action = do_action(action, budget, num_stocks, stock_price)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1

        stock_price = float(close_prices[t])
        new_portfolio = budget + num_stocks * stock_price

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reward = (new_portfolio - current_portfolio)/current_portfolio

        next_state_cnn = features_cnn[t + 1]
        next_state_lstm=features_lstm[t-19: t+1]
        next_state=[next_state_cnn, next_state_lstm]
        replay_memory.append((current_state, action, reward, next_state))

        step = step + 1
        if step < 5000:
            continue

        current_state_val, action_val, reward_val, next_state_val = (sample_memories(BATCH_SIZE, replay_memory))
        policy.update_q(current_state_val, action_val, reward_val, next_state_val)

        if t % COPY_STEP == 0:
            policy.copy_online_target()


    portfolio = budget + num_stocks * stock_price
    print(
        'budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks, stock_price, portfolio))

    return portfolio, action_count, np.asarray(action_seq)


def run_simulations(policy, budget, num_stocks, open_prices, close_prices, features_cnn, features_lstm, num_epoch, company):
    final_portfolios = list()
    replay_memory = deque([], maxlen=REPLAY_MEMORY_SIZE)

    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        final_portfolio, action_count, action_seq = run_simulation(policy, budget, num_stocks, open_prices,
                                                                   close_prices, features_cnn, features_lstm, epoch, replay_memory)
        final_portfolios.append(final_portfolio)
        print('actions : ', *zip(policy.actions, action_count))

        if (epoch + 1) % num_epoch == 0:
            plt.figure(figsize=(40, 20))
            plt.title('Epoch {}'.format(epoch + 1))
            plt.plot(open_prices[0: len(action_seq)], 'grey')
            plt.plot(pd.DataFrame(open_prices[: len(action_seq)])[action_seq == 'Sell'], 'ro', markersize=1)  # sell
            plt.plot(pd.DataFrame(open_prices[: len(action_seq)])[action_seq == 'Buy'], 'bo', markersize=1)  # buy
            plt.show()
    #policy.save_model("LFD_project4_team04/saved/{}".format(company))
    print(final_portfolios[-1])
