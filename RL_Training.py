from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import simulation as simulation
import tensorflow as tf
tf.compat.v1.reset_default_graph()

if __name__ == '__main__':
    start, end = '2010-01-01', '2020-05-19'

    # TODO: Choose companies for trading
    company_list = ['samsung1']

    # TODO: define action
    actions = ["Buy", "Sell"]

    # TODO: tuning model hyperparameters
    epsilon = 1.0
    gamma = 0.95
    lr = 0.001
    num_epoch = 20
    budget = 10. ** 8
    num_stocks = 0
    input_dim_cnn = 12
    input_dim_lstm = 17
    train_size = 2500
    #########################################
    for iteration, company in enumerate(company_list):
        tf.compat.v1.reset_default_graph()
        open_prices, close_prices, features_cnn = DataGenerator.make_features_cnn([company], start, end, is_training=True)
        features_lstm = DataGenerator.make_features_lstm([company], start, end, is_training=True)[2]
        policy = QLearningDecisionPolicy(epsilon=epsilon, gamma=gamma, lr=lr, actions=actions, input_dim_cnn=input_dim_cnn, input_dim_lstm=input_dim_lstm,
                                         model_dir="")
        policy.set_eps_decay(train_size*num_epoch)
        simulation.run_simulations(policy=policy, budget=budget, num_stocks=num_stocks,
                                   open_prices=open_prices[0], close_prices=close_prices[0], features_cnn=features_cnn[0], features_lstm=features_lstm[0], num_epoch=num_epoch,
                                   company=company)
    # TODO: fix checkpoint directory name