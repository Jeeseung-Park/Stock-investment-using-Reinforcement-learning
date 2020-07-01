from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class QLearningDecisionPolicy:

    def __init__(self, epsilon, gamma, lr, actions, input_dim_cnn, input_dim_lstm, model_dir):
        self.gamma = gamma
        self.lr = lr
        self.actions = actions
        self.eps_decay_steps = 1000
        self.epsilon = epsilon
        output_dim = 2
        tf.reset_default_graph()

        # CNN Architecture
        channels = 1

        conv1_fmaps = 30
        conv1_ksize = 1
        conv1_stride = 1
        conv1_pad = "SAME"

        conv2_fmaps = 10
        conv2_ksize = 2
        conv2_stride = 2
        conv2_pad = "VALID"

        flat_dropout_rate = 0.01

        n_fc = 100
        fc_dropout_rate = 0.07

        we_init = tf.keras.initializers.he_normal()
        he_init = tf.variance_scaling_initializer()

        # lstm Architecture
        n_steps = 20
        n_neurons = 50
        n_layers = 3

        self.training = tf.placeholder_with_default(False, shape=[], name="training")
        self.X_online_cnn = tf.placeholder(tf.float32, shape=[None, input_dim_cnn, input_dim_cnn])
        self.X_online_reshaped_cnn = tf.reshape(self.X_online_cnn, shape=[-1, input_dim_cnn, input_dim_cnn, channels])
        self.X_target_cnn = tf.placeholder(tf.float32, shape=[None, input_dim_cnn, input_dim_cnn])
        self.X_target_reshaped_cnn = tf.reshape(self.X_target_cnn, shape=[-1, input_dim_cnn, input_dim_cnn, channels])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.X_online_lstm = tf.placeholder(tf.float32, shape=[None, n_steps, input_dim_lstm])
        self.X_target_lstm = tf.placeholder(tf.float32, shape=[None, n_steps, input_dim_lstm])

        # Dueling DQN : Online DQN
        with tf.name_scope("Q_networks/online"):
            conv1_online = tf.layers.conv2d(self.X_online_reshaped_cnn, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                            strides=conv1_stride, padding=conv1_pad, kernel_initializer=we_init,
                                            name="conv1_online", activation=tf.nn.relu)
            bn1_online = tf.layers.batch_normalization(conv1_online, training=self.training, momentum=0.9)
            #bn1_act_online = tf.nn.relu(bn1_online)

            conv2_online = tf.layers.conv2d(bn1_online, filters=conv2_fmaps, kernel_size=conv2_ksize,
                                            strides=conv2_stride, padding=conv2_pad, kernel_initializer=we_init,
                                            name="conv2_online",activation=tf.nn.relu)
            bn2_online = tf.layers.batch_normalization(conv2_online, training=self.training, momentum=0.9)
            #bn2_act_online = tf.nn.relu(bn2_online)

            flat_online = tf.reshape(bn2_online, shape=[-1, (conv2_fmaps * input_dim_cnn * input_dim_cnn) // 4])
            flat_drop_online = tf.layers.dropout(flat_online, flat_dropout_rate, training=self.training)

            fc_online = tf.layers.dense(flat_drop_online, n_fc, activation=tf.nn.relu, kernel_initializer=he_init,
                                        name="fc_online")
            fc_online_drop = tf.layers.dropout(fc_online, fc_dropout_rate, training=self.training)

            lstm_cells_online = [
                tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, name="lstm_cell_online_{}".format(layer))
                for layer in range(n_layers)]
            multi_cell_online = tf.contrib.rnn.MultiRNNCell(lstm_cells_online)
            outputs_online, states_online = tf.nn.dynamic_rnn(multi_cell_online, self.X_online_lstm, dtype=tf.float32)
            top_layer_h_state_online = states_online[-1][1]
            fusion_layer_online=tf.concat( [fc_online_drop, top_layer_h_state_online], 1, name="concat_online")
            self.q_online = tf.layers.dense(fusion_layer_online, output_dim, name="output_online")

        # Dueling DQN : Target DQN
        with tf.name_scope("Q_networks/target"):
            conv1_target = tf.layers.conv2d(self.X_target_reshaped_cnn, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                            strides=conv1_stride, padding=conv1_pad, kernel_initializer=we_init,
                                            name="conv1_target",activation=tf.nn.relu)
            bn1_target = tf.layers.batch_normalization(conv1_target, training=self.training, momentum=0.9)
            #bn1_act_target = tf.nn.relu(bn1_target)

            conv2_target = tf.layers.conv2d(bn1_target, filters=conv2_fmaps, kernel_size=conv2_ksize,
                                            strides=conv2_stride, padding=conv2_pad, kernel_initializer=we_init,
                                            name="conv2_target",activation=tf.nn.relu)
            bn2_target = tf.layers.batch_normalization(conv2_target, training=self.training, momentum=0.9)
            #bn2_act_target = tf.nn.relu(bn2_target)

            flat_target = tf.reshape(bn2_target, shape=[-1, (conv2_fmaps * input_dim_cnn * input_dim_cnn) // 4])
            flat_drop_target = tf.layers.dropout(flat_target, flat_dropout_rate, training=self.training)

            fc_target = tf.layers.dense(flat_drop_target, n_fc, activation=tf.nn.relu, kernel_initializer=he_init,
                                        name="fc_target")
            fc_target_drop = tf.layers.dropout(fc_target, fc_dropout_rate, training=self.training)

            lstm_cells_target = [
                tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, name="lstm_cell_target_{}".format(layer))
                for layer in range(n_layers)]
            multi_cell_target = tf.contrib.rnn.MultiRNNCell(lstm_cells_target)
            outputs_target, states_target = tf.nn.dynamic_rnn(multi_cell_target, self.X_target_lstm, dtype=tf.float32)
            top_layer_h_state_target = states_target[-1][1]

            fusion_layer_target = tf.concat([fc_target_drop, top_layer_h_state_target], 1, name="concat_target")
            self.q_target = tf.layers.dense(fusion_layer_target, output_dim, name="output_target")

        # Copy Online DQN to Target DQN
        online_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Q_network/online")
        target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Q_network/target")
        target_vars = {var.name[len("Q_networks/target"):]: var for var in target_variables}
        copy_list = [target_var.assign(online_variables[var_name]) for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_list)

        self.X_action = tf.placeholder(tf.int32, shape=[None])
        self.q_value = tf.reduce_sum(self.q_online * tf.one_hot(self.X_action, output_dim), axis=1, keepdims=True)
        error = tf.abs(self.y - self.q_value)
        truncated_error = tf.clip_by_value(error, 0.0, 1.0)
        loss = tf.reduce_mean(tf.square(truncated_error) + 2 * (error - truncated_error))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        #######################################
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        #######################################

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.run(self.copy_online_to_target)

        self.saver = tf.train.Saver()
        self.restore_saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("load model: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def set_eps_decay(self, decay):
        self.eps_decay_steps = decay

    def select_action(self, current_state, step, is_training=True):
        epsilon_revised = max(0.1, self.epsilon - (self.epsilon - 0.1) * step / self.eps_decay_steps)

        if random.random() >= epsilon_revised or not is_training:
            action_q_vals = self.sess.run(self.q_online, feed_dict={self.X_online_cnn: [current_state[0]], self.X_online_lstm: [current_state[1]]})
            action_q_vals = action_q_vals[0]
            action_q_vals_softmax = softmax(action_q_vals)
            assure = (action_q_vals_softmax[0] - action_q_vals_softmax[1]) / (
                    action_q_vals_softmax[0] + action_q_vals_softmax[1])
            if assure > 0:
                action = self.actions[0]
            else:
                action = self.actions[1]
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
            assure = 0

        return action, assure

    def copy_online_target(self):
        self.sess.run(self.copy_online_to_target)

    def update_q(self, current_state, action, reward, next_state):
        next_action_q_vals = self.sess.run(self.q_target, feed_dict={self.X_target_cnn: list(map(lambda x: x[0], next_state)),
                                                                     self.X_target_lstm : list(map(lambda x: x[1], next_state)), self.training: True})
        max_next_action_q_vals = np.max(next_action_q_vals, axis=1, keepdims=True)
        y_val = reward + self.gamma * max_next_action_q_vals
        action_index = []
        for value in action:
            if value == "Buy":
                action_index.append(0)
            else:
                action_index.append(1)

        self.sess.run(self.train_op, feed_dict={self.X_online_cnn: list(map(lambda x: x[0], current_state)) , self.X_online_lstm: list(map(lambda x: x[1], current_state)),
                                                self.y: y_val, self.training: True, self.X_action: action_index})

    def save_model(self, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        checkpoint_path = output_dir + '/model.ckpt'
        self.saver.save(self.sess, checkpoint_path)

    def restore_model(self, input_dir):
        checkpoint_path = input_dir + '/model.ckpt'
        self.restore_saver.restore(self.sess, checkpoint_path)