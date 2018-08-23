import tensorflow as tf, numpy as np
import os
import time

import numpy as np
import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import argparse
import data_loader
from model import dual_cross_guided_att_vqamodel

def train(args):
    ####################
    loader = data_loader.Data_loader()
    # pdb.set_trace()
    print("Loading data...............")
    dataset, train_data = loader.get_qa_data(args.data_dir, data_set="train")
    print("Loading image features ...")
    image_features = loader.get_image_feature(train=True)
    print("Done !")
    # print("Image data length is %d " %len(image_features))
    word_num = len(dataset["ix_to_word"])
    print("Vocab size is %d "%word_num)
    print("Answer num is %d "%len(dataset["ix_to_ans"]))
    print("Loading pre-word-embedding ......")
    word_embed = loader.get_pre_embedding("./data")
    word_embed = np.float32(word_embed)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    # pdb.set_trace()
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    model = dual_cross_guided_att_vqamodel.Answer_Generator(
        rnn_size=args.rnn_size,
        rnn_layer=1,
        batch_size=args.batch_size,
        input_embedding_size=args.input_embedding_size,
        dim_image=[6,6,2048],
        dim_hidden=1200,
        dim_attention=512,
        max_words_q=args.questoin_max_length,
        vocabulary_size=word_num,
        drop_out_rate=0.5,
        num_output=args.num_answers,
        pre_word_embedding=word_embed)
    # model = mcbAtt_vqamodel.MCB_with_Attention(batch_size=args.batch_size,
    #                                     feature_dim=[2048,6,6],
    #                                     proj_dim=16000,
    #                                     word_num=word_num,
    #                                     embed_dim=300,
    #                                     ans_candi_num=3000,
    #                                     n_lstm_steps=26,
    #                                     pre_word_emb=word_embed)
    tf_image, tf_question, tf_label, tf_loss = model.trainer()
    saver = tf.train.Saver(max_to_keep = 1000)
    global_step = tf.Variable(0)
    sample_size = len(train_data["questions"])
    # learning rate decay
    lr = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps=sample_size/args.batch_size*30, decay_rate=0.9,
                                               staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    gvs = optimizer.compute_gradients(tf_loss, tvars)
    clipped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs if not grad is None]
    train_op = optimizer.apply_gradients(clipped_gvs, global_step=global_step)
    sess.run(tf.global_variables_initializer())
    model_file = "/home/hbliu/vqa/vqa-models/save/dual_cross_guided_att_vqamodel/model-390"
    print(model_file)
    saver.restore(sess, model_file)
    # Logging
    train_step = 0
    # pdb.set_trace()
    for itr in range(391, 1000):
        batch_no = 0
        # batch_no = 6834
        train_length = len(train_data["questions"])
        all_batch_num = train_length / args.batch_size
        sum_step = 0
        tot_loss = 0.0
        tStart = time.time()
        while (batch_no * args.batch_size) < train_length:
            curr_image_feat, curr_question, curr_answer = loader.get_next_batch(batch_no=batch_no,
                                                                        batch_size=args.batch_size,
                                                                        max_question_length=args.questoin_max_length,
                                                                        qa_data=train_data,
                                                                        image_features=image_features)
            # if train_length - batch_no * batch_size < batch_size:
            #     break
            # pdb.set_trace()
            _, loss = sess.run([train_op, tf_loss], feed_dict={tf_image: curr_image_feat,
                                                               tf_question: curr_question,
                                                               tf_label: curr_answer})

            # sum_writer.add_summary(summary, sum_step)
            sum_step += 1
            tot_loss += loss
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print("Loss : %f  batch : %d/%d   epoch : %d/%d"%(loss,batch_no,all_batch_num,itr,1000))
            lrate = sess.run(lr)
            print("learning rate :%s" % str(lrate))
            batch_no += 1
            train_step += 1
        tStop = time.time()
        if np.mod(itr, 10) == 0:
            print("Iteration: ", itr, " Loss: ", tot_loss, " Learning Rate: ", lr.eval(session=sess))
            print ("Time Cost:", round(tStop - tStart,2), "s")
        if np.mod(itr, 30) == 0:
            print("Iteration ", itr, " is done. Saving the model ...")
            saver.save(sess, os.path.join(args.save_path, 'model'), global_step=itr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch Size')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='rnn size')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Expochs')
    parser.add_argument('--input_embedding_size', type=int, default=300,
                        help='word embedding size')
    parser.add_argument('--version', type=int, default=1,
                        help='VQA data version')
    parser.add_argument('--num_answers', type=int, default=3000,
                        help='output answers nums')
    parser.add_argument('--questoin_max_length', type=int, default=26,
                        help='output answers nums')
    parser.add_argument('--data_dir', type=str, default="/home/hbliu/vqa/vqa-models/data",
                        help='output answers nums')
    parser.add_argument('--save_path', type=str, default="./save/dual_cross_guided_att_vqamodel",
                        help='save path')
    args = parser.parse_args()
    print(args)
    train(args=args)

if __name__ == '__main__':
    main()