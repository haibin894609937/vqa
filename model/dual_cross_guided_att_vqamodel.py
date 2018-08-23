import csv
import pdb

import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import sys
import RNN

csv.field_size_limit(sys.maxsize)

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer,batch_size, input_embedding_size,dim_image,dim_hidden,dim_attention
                       ,max_words_q,vocabulary_size, drop_out_rate, num_output, pre_word_embedding):
        print("Initializing dual cross-guided two-layer vqa model.........")
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.dim_att = dim_attention
        self.dim_q = self.rnn_size*2
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size
        self.drop_out_rate = drop_out_rate
        self.num_output= num_output
        self.K = 36
        self.hid = dim_attention
        self.rnn_model = RNN.RNN_Model()
        self.lamb = 10e-8
        # question-embedding
        # self.embed_question = tf.Variable(
        #     tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_question')
        self.pre_word_embedding = pre_word_embedding

        # image-embedding

        self.embed_image_W = self.get_weights(w_shape=[self.dim_image[2], self.dim_hidden], name="embed_image_W", lamb= self.lamb)
        self.embed_image_b =self.get_bias(b_shape=[self.dim_hidden], name="embed_image_b")

        self.embed_ques_W = self.get_weights(w_shape=[self.dim_q, self.dim_hidden], name="embed_ques_W", lamb=self.lamb)
        self.embed_ques_b = self.get_bias(b_shape=[self.dim_hidden], name="embed_ques_b")

        self.img_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="img_att_W", lamb=self.lamb)
        self.img_att_b = self.get_bias(b_shape=[1], name="img_att_b")

        self.ques_att_W = self.get_weights(w_shape=[self.dim_hidden, 1], name="ques_att_W", lamb=self.lamb)
        self.ques_att_b = self.get_bias(b_shape=[1], name="ques_att_b")

        self.qa_W_question = self.get_weights(w_shape=[self.dim_q, self.hid], name="qa_W_question", lamb=self.lamb)
        self.qa_b_question = self.get_bias(b_shape=[self.hid], name="qa_b_question")

        self.qa_W_prime_question = self.get_weights(w_shape=[self.dim_q, self.hid], name="qa_W_prime_question", lamb=self.lamb)
        self.qa_b_prime_question = self.get_bias(b_shape=[self.hid], name="qa_b_prime_question")

        self.qa_W_img = self.get_weights(w_shape=[self.dim_image[2], self.hid], name="qa_W_img", lamb=self.lamb)
        self.qa_b_img = self.get_bias(b_shape=[self.hid], name="qa_b_img")

        self.qa_W_prime_img = self.get_weights(w_shape=[self.dim_image[2], self.hid], name="qa_W_prime_img", lamb=self.lamb)
        self.qa_b_prime_img = self.get_bias(b_shape=[self.hid], name="qa_b_prime_img")

        self.qa_W_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_clf", lamb=self.lamb)
        self.qa_b_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_img")

        self.qa_W_prime_clf = self.get_weights(w_shape=[self.dim_hidden, self.dim_hidden], name="qa_W_prime_clf", lamb=self.lamb)
        self.qa_b_prime_clf = self.get_bias(b_shape=[self.dim_hidden], name="qa_b_prime_clf")
        # score-embedding
        self.embed_scor_W = self.get_weights(w_shape=[self.dim_hidden, self.num_output], name="embed_scor_W", lamb=self.lamb)
        self.embed_scor_b = self.get_bias(b_shape=[self.num_output], name="embed_scor_b")

    def get_weights(self, name, w_shape, lamb):
        weight = tf.Variable(tf.random_uniform(w_shape, -0.08, 0.08),name=name)
        weight_decay = tf.multiply(tf.nn.l2_loss(weight), lamb)
        tf.add_to_collection("losses", weight_decay)
        return weight

    def get_bias(self, name, b_shape):
        bias = tf.Variable(tf.random_uniform(b_shape, -0.08, 0.08),name=name)
        return bias

    def model(self):

        image = tf.placeholder(tf.float32, [self.batch_size, self.K, self.dim_image[2]]) # b*36*2048
        question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q]) # b*26
        label = tf.placeholder(tf.int64, [self.batch_size, ]) # b
        # drop_out = tf.placeholder(tf.float32)
        question_feat, _ = self.rnn_model.bi_gru_question(self.batch_size, pre_word_embedding=self.pre_word_embedding,
                                                  inputs=question, time_step=self.max_words_q,
                                                  layer_num=1, hidden_size=self.rnn_size)   # b*26*1024
        image_feat = tf.nn.l2_normalize(image, -1)
        # embedding
        image_emb = tf.reshape(image_feat, [-1, self.dim_image[2]])  # (b x m) x d
        image_emb = tf.nn.xw_plus_b(image_emb,self.embed_image_W, self.embed_image_b)
        image_emb = tf.nn.dropout(image_emb, keep_prob=self.drop_out_rate)
        image_emb = tf.tanh(tf.reshape(image_emb, shape=[self.batch_size, self.K, self.dim_hidden])) # (b*14*14)*1024

        ques_emb = tf.reshape(question_feat, [-1, self.dim_q])
        ques_emb = tf.nn.xw_plus_b(ques_emb, self.embed_ques_W, self.embed_ques_b)
        ques_emb = tf.nn.dropout(ques_emb, keep_prob=self.drop_out_rate)
        ques_emb = tf.tanh(tf.reshape(ques_emb, shape=[self.batch_size,-1, self.dim_hidden]))  # (b*26)*1024

        # first layer attention
        image_emb_att = tf.nn.xw_plus_b(tf.reshape(image_emb, shape=[-1,self.dim_hidden]), self.img_att_W, self.img_att_b)
        self.image_emb_prob = tf.nn.softmax(tf.reshape(image_emb_att, shape=[self.batch_size, -1]))
        image_emb_prob = self.image_emb_prob
        img_memory = tf.reduce_sum(tf.expand_dims(image_emb_prob,2)*image_emb, axis=1)
        # reduce
        ques_emb_att = tf.nn.xw_plus_b(tf.reshape(ques_emb, shape=[-1,self.dim_hidden]), self.ques_att_W, self.ques_att_b)
        ques_emb_prob = tf.nn.softmax(tf.reshape(ques_emb_att, shape=[self.batch_size, -1]))
        ques_memory = tf.reduce_sum(tf.expand_dims(ques_emb_prob, 2) * ques_emb, axis=1)
        # ques * image
        memory = img_memory*ques_memory  # b*1024
        # # attention models
        with tf.variable_scope("att1"):
            # vis_comb1 512   ques_comb1 1024
            self.vis_att_prob1, vis_comb1 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb)
            self.ques_att_prob1, ques_comb1 =self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb)
            img_memory = img_memory + vis_comb1
            ques_memory = ques_memory + ques_comb1
            memory = memory + img_memory*ques_memory
        with tf.variable_scope("att2"):
            self.vis_att_prob2, vis_comb2 = self.tanh_vis_attention(question_emb=ques_memory, image_emb=image_emb)
            self.ques_att_prob2, ques_comb2 = self.tanh_ques_attention(image_emb=img_memory, question_emb=ques_emb)
            img_memory = img_memory + vis_comb2
            ques_memory = ques_memory + ques_comb2
            memory = memory + img_memory * ques_memory

        s_head = self.gated_tanh(memory, self.qa_W_clf, self.qa_b_clf, self.qa_W_prime_clf, self.qa_b_prime_clf)
        s_head = tf.nn.dropout(s_head, keep_prob=self.drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(s_head, self.embed_scor_W, self.embed_scor_b)

        print("classification nums")
        print(scores_emb)
        return image, question, label, scores_emb

    def trainer(self):
        image, question, label, scores_emb = self.model()
        # Calculate cross entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=scores_emb)
        # Calculate loss
        loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', loss)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return image, question, label, loss

    def solver(self):
        image, question, label, scores_emb = self.model()
        answer_prob = tf.nn.softmax(scores_emb)
        return image, question, answer_prob, self.image_emb_prob, self.vis_att_prob1, self.vis_att_prob2, self.ques_att_prob1, self.ques_att_prob2

    def gated_tanh(self, concated, w1, b1, w2, b2):
        y_tilde = tf.tanh(tf.nn.xw_plus_b(concated, w1, b1))
        g = tf.sigmoid(tf.nn.xw_plus_b(concated, w2, b2))
        y = tf.multiply(y_tilde, g)
        return y


    def tanh_vis_attention(self, question_emb, image_emb):
        # Attention weight
        # question_emb b*1024  image_emb b*K*2048
        # question-attention
        # probability-attention
        gt_W_img_att = self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_img_att", lamb=self.lamb)
        gt_b_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_img_att")
        gt_W_prime_img_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_img_att", lamb=self.lamb)
        gt_b_prime_img_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_img_att")
        prob_image_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_image_att_W", lamb=self.lamb)
        prob_image_att_b = self.get_bias(b_shape=[1], name="prob_image_att_b")
        # pdb.set_trace()
        qenc_reshape = tf.tile(tf.expand_dims(question_emb, 1), multiples=[1, self.K, 1])  # b * k * 1024
        concated = tf.concat([image_emb, qenc_reshape], axis=2)  # b * m * (image_dim + ques_dim)
        concated = tf.reshape(concated, shape=[self.batch_size * self.K, -1])
        concated = self.gated_tanh(concated, gt_W_img_att, gt_b_img_att, gt_W_prime_img_att, gt_b_prime_img_att)  # (b * m) * hid
        att_map = tf.nn.xw_plus_b(concated, prob_image_att_W, prob_image_att_b)  # b*m*1
        att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.K]))
        v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * image_emb, axis=1)
        return att_prob, v_head

    def tanh_ques_attention(self, question_emb, image_emb):
        # Attention weight
        gt_W_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_ques_att", lamb=self.lamb)
        gt_b_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_ques_att")

        gt_W_prime_ques_att =self.get_weights(w_shape=[self.dim_hidden*2, self.hid], name="gt_W_prime_ques_att", lamb=self.lamb)
        gt_b_prime_ques_att = self.get_bias(b_shape=[self.hid], name="gt_b_prime_ques_att")

        prob_ques_att_W = self.get_weights(w_shape=[ self.hid,1], name="prob_ques_att_W", lamb=self.lamb)
        prob_ques_att_b = self.get_bias(b_shape=[1], name="prob_ques_att_b")

        img_reshape = tf.tile(tf.expand_dims(image_emb, 1), multiples=[1, self.max_words_q, 1])  # b * 26 * 1024
        concated = tf.concat([question_emb, img_reshape], axis=2)  # b * 26 * (image_dim + ques_dim)
        concated = tf.reshape(concated, shape=[self.batch_size * self.max_words_q, -1])
        concated = self.gated_tanh(concated, gt_W_ques_att, gt_b_ques_att, gt_W_prime_ques_att, gt_b_prime_ques_att)  # (b * m) * hid
        att_map = tf.nn.xw_plus_b(concated, prob_ques_att_W, prob_ques_att_b)  # b*m*1
        att_prob = tf.nn.softmax(tf.reshape(att_map, shape=[-1, self.max_words_q]))
        v_head = tf.reduce_sum(tf.expand_dims(att_prob, axis=2) * question_emb, axis=1)
        return att_prob, v_head

    def vis_attention(self, question_emb, image_emb):
        # Attention weight
        # question_emb b*1024  image_emb (b*14*14)*512
        # question-attention
        # probability-attention
        prob_att_W = tf.get_variable('prob_att_W1', [self.dim_hidden, 1],
                                     initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b1', [1],
                                     initializer=tf.random_uniform_initializer(-0.08, 0.08))
        question_att = tf.expand_dims(question_emb, 1)  # b x 1 x d
        question_att = tf.tile(question_att, tf.constant([1, self.K, 1]))  # b x m x d
        question_att = tf.reshape(question_att, [-1, self.dim_q])  # (b x m) x d
        output_att = image_emb * question_att  # (b x m) x k
        prob_att = tf.nn.xw_plus_b(output_att, prob_att_W, prob_att_b)  # (b x m) x 1
        prob_att = tf.reshape(prob_att, [self.batch_size, self.K])  # b x m
        prob_att = tf.nn.softmax(prob_att)
        image_emb = tf.reshape(image_emb, shape=[self.batch_size,-1, self.dim_hidden])
        image_att_sum = tf.reduce_sum(image_emb * tf.expand_dims(prob_att, dim=2), 1)
        return prob_att, image_att_sum

    def ques_attention(self, question_emb, image_emb):
        # Attention weight
        prob_att_W = tf.get_variable('prob_att_W2', [self.dim_hidden, 1],
                                     initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b2', [1],
                                     initializer=tf.random_uniform_initializer(-0.08, 0.08))
        image_att = tf.expand_dims(image_emb, 1)  # b x 1 x d
        image_att = tf.tile(image_att, tf.constant([1, self.max_words_q, 1]))  # b x m x d
        image_att = tf.reshape(image_att, shape=[-1, self.dim_hidden])
        output_att = tf.tanh(image_att * question_emb)  # (b x m) x k
        output_att = tf.nn.dropout(output_att, 1 - self.drop_out_rate)
        prob_att = tf.nn.xw_plus_b(output_att, prob_att_W, prob_att_b)  # (b x m) x 1
        prob_att = tf.reshape(prob_att, [self.batch_size, self.max_words_q])  # b x m
        prob_att = tf.nn.softmax(prob_att)  # b*26
        question_emb = tf.reshape(question_emb, shape=[self.batch_size, -1, self.dim_hidden])
        ques_att_sum = tf.reduce_sum(question_emb * tf.expand_dims(prob_att, dim=2), 1)
        return prob_att, ques_att_sum

if __name__ == "__main__":
    ac= 10


