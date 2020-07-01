import numpy as np
import DataGenerator as DataGenerator
from decision_ql import QLearningDecisionPolicy
import tensorflow as tf


def do_test_action(actions, assures, prices, budget, num_stocks):

    buy = []
    sell = []
    buy_assure = []
    sell_assure = []
    decision = [0 for i in actions]
    for i in range(len(actions)):
        if actions[i] == "Buy":
            buy.append(i)
            buy_assure.append(assures[i])
        else:
            sell.append(i)

    if len(buy) != 0:  # there is something to buy

        for i in range(len(sell)):
            stock_price = prices[sell[i]]
            n_sell = num_stocks[sell[i]]
            num_stocks[sell[i]] = 0
            budget += stock_price * n_sell
            decision[sell[i]] = 'Sell'

        total_assure = sum(buy_assure)
        price_dist = (buy_assure / total_assure) * min((10. ** 7), budget)

        for i in range(len(buy)):
            stock_price = prices[buy[i]]
            n_buy = min(np.ceil(price_dist[i] / stock_price), budget // stock_price)
            num_stocks[buy[i]] += n_buy
            budget -= stock_price * n_buy
            decision[buy[i]] = 'Buy'
    else:  # there is nothing to buy
        idx = -1
        max_assure = -1
        for i, j in enumerate(sell_assure):
            if j > max_assure:
                idx = i
                max_assure = j

        stock_to_buy = sell[idx]

        for i in range(len(sell)):
            stock_price = prices[sell[i]]
            n_sell = num_stocks[sell[i]]
            num_stocks[sell[i]] = 0
            budget += stock_price * n_sell
            decision[sell[i]] = 'Sell'

        stock_price = prices[stock_to_buy]
        n_buy = min(np.ceil(10. ** 7 / stock_price), budget // stock_price)
        num_stocks[stock_to_buy] += n_buy
        budget -= stock_price * n_buy

        decision[stock_to_buy] = 'Sell then Buy'

    return budget, num_stocks, decision


def run(action_list, assure_list , initial_budget, initial_num_stock, open_prices, close_prices):

    budget = initial_budget
    num_stocks = initial_num_stock
    history = []
    for i in range(len(open_prices[0])):
        cur_actions=np.array(action_list)[:,i]
        cur_assures=np.array(assure_list)[:,i]
        cur_open_prices=np.array(open_prices)[:,i]
        cur_close_prices=np.array(close_prices)[:,i]
        budget, num_stocks, decision = do_test_action(cur_actions, cur_assures, cur_open_prices, budget, num_stocks)

        stock_value_open = 0
        for k in range(len(num_stocks)):
            stock_value_open += num_stocks[k]*cur_open_prices[k]

        stock_value_close = 0
        for k in range(len(num_stocks)):
            stock_value_close += num_stocks[k] * cur_close_prices[k]

        print('Day {}'.format(i + 1))
        print('action {} / budget {} / shares {}'.format(decision, budget, num_stocks))
        print('portfolio with  open price : {}'.format(budget + stock_value_open))
        print('portfolio with close price : {}\n'.format(budget + stock_value_close))
        history.append(budget+stock_value_close)

    stock_value = 0
    for k in range(len(num_stocks)):
        stock_value += num_stocks[k] * close_prices[k][-1]

    portfolio = budget + stock_value

    final_close_prices = []
    for k in range(len(num_stocks)):
        final_close_prices.append(close_prices[k][-1])

    print('Finally, you have')
    print('budget: %.2f won' % budget)
    print('Shares: {}'.format(num_stocks))
    print('Share value: {} won'.format(final_close_prices))
    print()

    with open('your_file.txt', 'w') as f:
        for item in history:
            f.write("%s\n" % item)

    return portfolio


if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'
    company_list = ['cell', 'hmotor', 'naver', 'kakao', 'lgchem', 'lghnh', 'samsung1' ,'samsung2', 'sdi', 'sk']
    portfolio_size = len(company_list)
    actions = ["Buy", "Sell"]
    #########################################
    open_prices, close_prices, features_cnn = DataGenerator.make_features_cnn(company_list, start, end, is_training=False)
    features_lstm = DataGenerator.make_features_lstm(company_list, start, end, is_training=False)[2]

    budget = 10. ** 8
    num_stocks = [0 for i in range(portfolio_size)]
    input_dim_cnn = 12
    input_dim_lstm = 17
    action_list = []
    assure_list = []

    for i, company in enumerate(company_list):
        tf.reset_default_graph()
        policy = QLearningDecisionPolicy(0, 1, 0, actions=actions, input_dim_cnn=input_dim_cnn, input_dim_lstm = input_dim_lstm, model_dir="")
        policy.restore_model("LFD_project4_team04/saved/{}".format(company))
        action_company = []
        assure_company = []

        for j in range(len(open_prices[0])):
            current_state_cnn = features_cnn[i][j]
            current_state_lstm = features_lstm[i][j: j+20]
            current_state = [current_state_cnn, current_state_lstm]
            curr_action_company, curr_assure_company=policy.select_action(current_state, 0, is_training=False)
            action_company.append(curr_action_company)
            assure_company.append(curr_assure_company)

        action_list.append(action_company)
        assure_list.append(assure_company)

    final_portfolio = run(action_list, assure_list, budget, num_stocks, open_prices, close_prices)
    print("Final portfolio: %.2f won" % final_portfolio)