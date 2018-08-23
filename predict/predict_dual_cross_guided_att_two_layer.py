import json
import pdb

import h5py
import numpy as np
import tensorflow as tf

rnn_cell = tf.nn.rnn_cell
import argparse

import sys
sys.path.insert(0, "../")
import data_loader
from model import dual_cross_guided_att_vqamodel


def get_next_batch(batch_no, batch_size, max_question_length, qa_data, image_features):
    train_length = len(qa_data["questions"])
    si = (batch_no * batch_size) % train_length
    # unique_img_train[img_pos_train[0]]
    ei = min(train_length, si + batch_size)
    n = ei - si
    pad_n = n
    n = batch_size
    image_feature = np.ndarray((n, 36, 2048))
    # ---------------
    question = np.ndarray((n, max_question_length), dtype=np.int32)
    count = 0
    # pdb.set_trace()
    mc_answer=[]
    questions_id = []
    images_id = []
    for i in range(si, ei):
        img_id = qa_data["images_id"][i]
        mc_answer.append(qa_data["mc_answers"][i])
        image_feature[count, :] = image_features[img_id]
        question[count, :] = qa_data["questions"][i]
        questions_id.append(qa_data["questions_id"][i])
        images_id.append(qa_data["images_id"][i])
        count += 1

    for i in range(n - pad_n):
        r_num = np.random.randint(0, train_length - 1)
        img_id = qa_data["images_id"][r_num]
        mc_answer.append(qa_data["mc_answers"][r_num])
        image_feature[count, :] = image_features[img_id]
        question[count, :] = qa_data["questions"][r_num]
        questions_id.append(qa_data["questions_id"][r_num])
        images_id.append(qa_data["images_id"][r_num])
        count += 1
    return image_feature, question, images_id, questions_id, mc_answer

def test(args):
    loader = data_loader.Data_loader()
    # pdb.set_trace()
    print("Loading data...............")
    dataset, test_data = loader.get_qa_data(args.data_dir, data_set="test")
    # pdb.set_trace()
    print("Loading image features ...")
    image_features = loader.get_image_feature(train=False)
    print("Image feat length ", len(image_features))
    print("Done !")
    # print("Image data length is %d " %len(image_features))
    word_num = len(dataset["ix_to_word"])
    print("Vocab size is %d " % word_num)
    print("Answer num is %d " % len(dataset["ix_to_ans"]))
    print("Loading pre-word-embedding ......")
    word_embed = loader.get_pre_embedding(args.data_dir)
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
        dim_image=[6, 6, 2048],
        dim_hidden=1200,
        dim_attention=512,
        max_words_q=args.questoin_max_length,
        vocabulary_size=word_num,
        drop_out_rate=1,
        num_output=args.num_answers,
        pre_word_embedding=word_embed)
    tf_image, tf_question, tf_answer_prob,tf_pure_vis_prob, tf_vis_prob1,tf_vis_prob2, tf_ques_prob1, tf_ques_prob2 = model.solver()
    sess.run(tf.global_variables_initializer())
    # Logging
    saver = tf.train.Saver()
    model_file = '/home/hbliu/vqa/vqa-models/save/dual_cross_guided_att_vqamodel/model-360'
    print("Restore models ...\n", model_file)
    saver.restore(sess, model_file)
    # pdb.set_trace()
    for itr in range(1):
        batch_no = 0
        # pdb.set_trace()
        open_result = []
        mc_result = []
        test_length = len(test_data["questions"])
        print("length ---------------",test_length)
        length = 1
        pure = []
        vis1 = []
        vis2 = []
        ques1 = []
        ques2 = []
        images_id = []
        questions_id = []
        bbox = []
        boxes = np.load("/home/hbliu/vqa/vqa-models/vis/test_image_box_top_36.npy").item()
        while (batch_no * args.batch_size) < test_length:
            # image_feature, question, answer,images_id, questions_id
            curr_image_feat, curr_question, curr_image_id, curr_ques_id, mc_answers = get_next_batch(
                batch_no=batch_no,
                batch_size=args.batch_size,
                max_question_length=args.questoin_max_length,
                qa_data=test_data,
                image_features=image_features)
            answer_prob, pure_prob, vis_prob1,vis_prob2,ques_prob1,ques_prob2 = sess.run([tf_answer_prob,
                                                                                          tf_pure_vis_prob,
                                                                                          tf_vis_prob1,
                                                                                          tf_vis_prob2,
                                                                                          tf_ques_prob1,
                                                                                          tf_ques_prob2],
                                                                                         feed_dict={tf_image: curr_image_feat,
                                                                                                    tf_question: curr_question})
            batch_no += 1
            top_ans = np.argmax(answer_prob, axis=1)
            for i in range(len(answer_prob)):
                print("...................")
                print(length + 1)
                ans = dataset['ix_to_ans'][str(top_ans[i] + 1)]

                if length <= test_length:
                    print("OE ans ---------", ans)
                    open_result.append({u'answer': ans, u'question_id': int(curr_ques_id[i])})
                    ########## Save Att maps ################
                    pure.append(pure_prob[i])
                    vis1.append(vis_prob1[i])
                    vis2.append(vis_prob2[i])
                    ques1.append(ques_prob1[i])
                    ques2.append(ques_prob2[i])
                    images_id.append(curr_image_id[i])
                    questions_id.append(curr_ques_id[i])
                    img_id = curr_image_id[i]
                    box = boxes[img_id]
                    bbox.append(box)
                # ##compute MC
                mc_prob = np.ndarray([1, 18], dtype=np.float32)
                # pdb.set_trace()
                for j in range(18):
                    mc_index = mc_answers[i][j]
                    if mc_index == 0:
                        mc_prob[0, j] = 0
                    else:
                        mc_prob[0, j] = answer_prob[i][mc_index-1]
                        # if mc_index in range(1,1001):
                        #     mc_prob[0, j] = answer_prob[i][mc_index]
                        # else:
                        #     mc_prob[0, j] = 0
                # pdb.set_trace()
                mc_max_index = np.argmax(mc_prob)
                mc_ans = dataset['ix_to_ans'][str(mc_answers[i][mc_max_index])]
                if length <= test_length:
                    print("MC ans ---------", mc_ans)
                    mc_result.append({u'answer': mc_ans, u'question_id': int(curr_ques_id[i])})
                length += 1
        print("Total OE answers ", len(open_result))
        print("Total MC answers ", len(mc_result))
        print("saving results .....")

        with h5py.File("test_dual_cross_guided_att_weights_two_layer.h5", "w") as f:
            f.create_dataset("vis1", data=vis1)
            f.create_dataset("vis2", data=vis2)
            f.create_dataset("ques1", data=ques1)
            f.create_dataset("ques2", data=ques2)
            f.create_dataset("image_id",data=images_id)
            f.create_dataset("ques_id", data=questions_id)
            f.create_dataset("boxes", data=bbox)
            f.create_dataset("pure", data=pure)

        with open("vqa_OpenEnded_mscoco_test2015_dualCrossTwo390_results.json", "w") as f:
            json.dump(open_result, f)

        with open("vqa_MultipleChoice_mscoco_test2015_dualCrossTwo390_results.json", "w") as f:
            json.dump(mc_result, f)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch Size')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='rnn size')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
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
    test(args=args)

if __name__ == '__main__':
    main()