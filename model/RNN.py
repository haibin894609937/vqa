import pdb

import tensorflow as tf
rnn_cell = tf.nn.rnn_cell

class RNN_Model():
    def __init__(self):
        print("Initializing RNN model")

    def bi_gru_question(self, batch_size, pre_word_embedding, inputs, time_step, layer_num, hidden_size, keep_prob=1, input_embedding_size=300):
        """build the bi-LSTMs network. Return the y_pred"""
        # ** 1.GRU
        print("11111111111")
        gru_fw_cell = rnn_cell.GRUCell(hidden_size)
        gru_bw_cell = rnn_cell.GRUCell(hidden_size)
        # ** 2.dropout
        gru_fw_cell = rnn_cell.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        gru_bw_cell = rnn_cell.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # ** 3.MultiLayer
        cell_fw = rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
        cell_bw = rnn_cell.MultiRNNCell([gru_bw_cell] * layer_num, state_is_tuple=True)
        # ** 4.initial state
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # ** 5. bi-lstm
        with tf.variable_scope('bidirectional_rnn'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope("fw"):
                for i in range(time_step):
                    if i == 0:
                        ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                    else:
                        tf.get_variable_scope().reuse_variables()
                        ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                    (output_fw, state_fw) = cell_fw(ques_emb_linear, state_fw)
                    outputs_fw.append(output_fw)
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope("bw"):
                inputs = tf.reverse(inputs, [1])
                for i in range(time_step):
                    if i == 0:
                        ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                    else:
                        tf.get_variable_scope().reuse_variables()
                        ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                    (output_bw, state_bw) = cell_bw(ques_emb_linear, state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1, 0, 2])
            states = tf.reshape(tf.concat([state_fw, state_bw], 2), shape=[batch_size, -1])
        return output, states
    def bi_gru_semantic(self, batch_size, pre_word_embedding, inputs, time_step, layer_num, hidden_size, keep_prob=1, input_embedding_size=300):
        """build the bi-LSTMs network. Return the y_pred"""
        # ** 1.GRU
        print("Initial Semantic RNN")
        gru_fw_cell = rnn_cell.GRUCell(hidden_size)
        gru_bw_cell = rnn_cell.GRUCell(hidden_size)
        # ** 2.dropout
        gru_fw_cell = rnn_cell.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        gru_bw_cell = rnn_cell.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # ** 3.MultiLayer
        cell_fw = rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
        cell_bw = rnn_cell.MultiRNNCell([gru_bw_cell] * layer_num, state_is_tuple=True)
        # ** 4.initial state
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # ** 5. bi-lstm
        with tf.variable_scope('semantic_rnn'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope("semantic_fw"):
                for i in range(time_step):
                    if i>0:
                        tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i])
                    (output_fw, state_fw) = cell_fw(ques_emb_linear, state_fw)
                    outputs_fw.append(output_fw)
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope("semantic_bw"):
                inputs = tf.reverse(inputs, [1])
                for i in range(time_step):
                    if i>0:
                        tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i])
                    (output_bw, state_bw) = cell_bw(ques_emb_linear, state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1, 0, 2])
            states = tf.reshape(tf.concat([state_fw, state_bw], 2), shape=[batch_size, -1])
        return output, states
    def bi_gru_image(self, batch_size, inputs, time_step, layer_num, hidden_size, keep_prob=1):
        """build the bi-LSTMs network. Return the y_pred"""
        # ** 1.GRU
        gru_fw_cell = rnn_cell.GRUCell(hidden_size)
        gru_bw_cell = rnn_cell.GRUCell(hidden_size)
        # ** 2.dropout
        gru_fw_cell = rnn_cell.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        gru_bw_cell = rnn_cell.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # ** 3.MultiLayer
        cell_fw = rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
        cell_bw = rnn_cell.MultiRNNCell([gru_bw_cell] * layer_num, state_is_tuple=True)
        # ** 4.initial state
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # ** 5. bi-lstm
        with tf.variable_scope('bidirectional_rnn_image'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw_image'):
                for timestep in range(time_step):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw_image'):
                for timestep in range(time_step):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1, 0, 2])
        return output
    def bi_gru_question_embedding(self, batch_size, inputs, time_step, layer_num, hidden_size, keep_prob=1):
        """build the bi-LSTMs network. Return the y_pred"""
        # ** 1.GRU
        gru_fw_cell = rnn_cell.GRUCell(hidden_size)
        gru_bw_cell = rnn_cell.GRUCell(hidden_size)
        # ** 2.dropout
        gru_fw_cell = rnn_cell.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        gru_bw_cell = rnn_cell.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # ** 3.MultiLayer
        cell_fw = rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
        cell_bw = rnn_cell.MultiRNNCell([gru_bw_cell] * layer_num, state_is_tuple=True)
        # ** 4.initial state
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # ** 5. bi-lstm
        with tf.variable_scope('bidirectional_rnn_image'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw_image'):
                for timestep in range(time_step):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw_image'):
                for timestep in range(time_step):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1, 0, 2])
        return output
    def bi_lstm_question(self, batch_size, pre_word_embedding, inputs, time_step, layer_num, hidden_size, keep_prob=1, input_embedding_size=300):
        """build the bi-LSTMs network. Return the y_pred"""
        # ** 1.GRU
        print("11111111111")
        gru_fw_cell = rnn_cell.LSTMCell(hidden_size)
        gru_bw_cell = rnn_cell.LSTMCell(hidden_size)
        # ** 2.dropout
        gru_fw_cell = rnn_cell.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        gru_bw_cell = rnn_cell.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # ** 3.MultiLayer
        cell_fw = rnn_cell.MultiRNNCell([gru_fw_cell] * layer_num, state_is_tuple=True)
        cell_bw = rnn_cell.MultiRNNCell([gru_bw_cell] * layer_num, state_is_tuple=True)
        # ** 4.initial state
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # ** 5. bi-lstm
        with tf.variable_scope('bidirectional_rnn'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope("fw"):
                for i in range(time_step):
                    if i == 0:
                        ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                    else:
                        tf.get_variable_scope().reuse_variables()
                        ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                    (output_fw, state_fw) = cell_fw(ques_emb_linear, state_fw)
                    outputs_fw.append(output_fw)
            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope("bw"):
                inputs = tf.reverse(inputs, [1])
                for i in range(time_step):
                    if i == 0:
                        ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                    else:
                        tf.get_variable_scope().reuse_variables()
                        ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                    (output_bw, state_bw) = cell_bw(ques_emb_linear, state_bw)
                    outputs_bw.append(output_bw)
            outputs_bw = tf.reverse(outputs_bw, [0])
            output = tf.concat([outputs_fw, outputs_bw], 2)
            output = tf.transpose(output, perm=[1, 0, 2])
            states = tf.reshape(tf.concat([state_fw, state_bw], 2), shape=[batch_size, -1])
        return output, states

    def multi_lstm_qustion(self, batch_size, pre_word_embedding, inputs, time_step, layer_num, hidden_size, keep_prob=1, input_embedding_size=300):
        # encoder: RNN body
        lstm_1 = rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
        lstm_dropout_1 = rnn_cell.DropoutWrapper(lstm_1, output_keep_prob=keep_prob)
        lstm_2 = rnn_cell.LSTMCell(hidden_size, use_peepholes=True)
        lstm_dropout_2 = rnn_cell.DropoutWrapper(lstm_2, output_keep_prob=keep_prob)
        stacked_lstm = rnn_cell.MultiRNNCell([lstm_dropout_1, lstm_dropout_2])

        states_feat = list()
        state = stacked_lstm.zero_state(batch_size, tf.float32)
        question_emb = None
        with tf.variable_scope("embed"):
            for i in range(time_step):
                if i == 0:
                    ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                # pdb.set_trace()
                # LSTM based question model
                # ques_emb_drop = tf.nn.dropout(ques_emb_linear, keep_prob)
                # ques_emb = tf.tanh(ques_emb_drop)
                (output, state) = stacked_lstm(ques_emb_linear, state)
                # question_emb = tf.concat([state[0].h, state[1].h], 1)
                # a = tf.shape(state)
                # print(tf.shape(state))
                # question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])
                states_feat.append(output)
        # multimodal (fusing question & image)
        # states_feat = tf.stack(states_feat)
        output = tf.transpose(states_feat, perm=[1, 0, 2])
        # state_emb = tf.transpose(states_feat, perm=[1, 0, 2])  # b*26*1024
        states = tf.reshape(state, shape=[batch_size, -1])
        return output,states

    def cnn_encode_question(self, batch_size, pre_word_embedding, inputs, time_step, input_embedding_size=300):
        states_feat = list()
        with tf.variable_scope("embed"):
            for i in range(time_step):
                if i == 0:
                    ques_emb_linear = tf.zeros([batch_size, input_embedding_size])
                else:
                    tf.get_variable_scope().reuse_variables()
                    ques_emb_linear = tf.nn.embedding_lookup(pre_word_embedding, inputs[:, i - 1])
                states_feat.append(ques_emb_linear)
        # multimodal (fusing question & image)
        question_feat = tf.stack(states_feat)
        question_feat = tf.transpose(question_feat, [1, 2, 0])
        question_feat = tf.expand_dims(question_feat, 3)

        with tf.variable_scope("conv1"):
            tanh1 = self.conv_tanh(question_feat, [input_embedding_size, 2, 1, 128], [128])
            tanh1 = tf.reduce_max(tanh1, 2)
        with tf.variable_scope("conv2"):
            tanh2 = self.conv_tanh(question_feat, [input_embedding_size, 3, 1, 256], [256])
            tanh2 = tf.reduce_max(tanh2, 2)
        with tf.variable_scope("conv3"):
            tanh3 = self.conv_tanh(question_feat, [input_embedding_size, 4, 1, 256], [256])
            tanh3 = tf.reduce_max(tanh3, 2)
        question_emb = tf.concat([tanh1, tanh2, tanh3], 2)  # b x 1 x d
        question_emb = tf.reduce_max(question_emb, 1)
        return question_emb

    def conv_tanh(self, input, kernel_shape, bias_shape):
        weights = tf.get_variable("cnn_weights", kernel_shape,
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable("cnn_biases", bias_shape,
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights,
                            strides=[1, 1, 1, 1], padding='VALID')

        return tf.tanh(conv + biases)

if __name__ == "__main__":
    print(10)
    rnn_model = RNN_Model()
    pre_word_embedding = tf.Variable(tf.random_uniform([1000, 300], -0.08, 0.08),
                                         name='Wr1')
    inputs = tf.Variable(tf.random_uniform([10, 26], 0, 100),
                                         name='Wr1')
    inputs = tf.cast(inputs, dtype=tf.int32)

    q_emb = rnn_model.bi_lstm_question(10, pre_word_embedding=pre_word_embedding, inputs=inputs, time_step=26,
                                      layer_num=1, hidden_size=512)
    print(q_emb)