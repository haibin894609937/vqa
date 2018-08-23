import numpy as np
import pickle
import h5py
import pdb

def get_glove_embedding_index(glove_path):
    embeddings_index={}
    print("Loading glove data.............")
    with open(glove_path, 'rb') as glove_file:
        i =0
        # glove_len = len(glove_file.readlines())
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
           #  print(word)
            embeddings_index[word] = coefs
            if i%1000 == 0:
                print("Processing %d"%(i))
            i=i+1
    print("glove embedding length : %d" %len(embeddings_index))
    print("Finish loading glove................")
    return embeddings_index

#
"""
# single question embedding
# Parameters
# embeddings_index is glove embedding
"""
def question_embedding(question, embeddings_index, embedding_dim, q_max_len):
    j=0
    for k,v in embeddings_index.items():
        print(k)
        print(v)
        if(j>2):
            break
        j=j+1
    embedding_matrix = np.zeros((q_max_len, embedding_dim))
    base = q_max_len - len(question)
    i = 0
    for word in question:
        vec_emb = embeddings_index[word]
        if vec_emb is not None:
            embedding_matrix[base + i] = vec_emb
        i = i+1
    return embedding_matrix


def make_embedding_matrix(vacab_file_path, embedding_index={}, embedding_dim=300):
    with open(vacab_file_path, "rb") as f:
        vacab = pickle.load(f)
        q_vacab = vacab["question_vocab"]
        print("question vocab length is ",len(q_vacab))
        embedding_matrix = np.zeros((len(q_vacab)+1, embedding_dim))
        res =[]
        for word, i in q_vacab.items():
            word = word.lower().encode('ascii')
            embedding_vector = embedding_index.get(word)
            print(embedding_vector)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                res.append(word)
        print("can't encode %d"%len(res))
        for w in res:
            print(w)
    with h5py.File("question_vocab_embedding.h5", 'w') as f:
        f.create_dataset('question_embedding_matrix', data=embedding_matrix)
    return embedding_matrix

def load_glove_embedding(embdding_path):
    with h5py.File(embdding_path, 'r') as hf:
        embedding_data = np.array(hf.get("question_embedding_matrix"))
    return embedding_data
import json
def get_train_data(input_json):
    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]
    return dataset

if __name__ == "__main__":

    # make_embedding_matrix("./1000_vocab_file1.pkl",
    #                      get_glove_embedding_index("D:/Project/Python/VQA/Data/embedding/glove.42B.300d.txt"),
    #                      300)
    data = get_train_data("semantic_concepts_vocab.json")
    embedding_index = get_glove_embedding_index("/home/hbliu/data/embedding/glove/glove.42B.300d.txt")
    embedding_matrix = np.zeros((len(data["itow"]) + 1, 300))
    res = []

    pdb.set_trace()
    index = 1
    for i, word in data["itow"].items():
        print(index)
        index +=1
        word = word.lower().encode('ascii')
        words = word.split(" ")
        word_f = words[-1]
        embedding_vector = embedding_index.get(word_f)
        i = i.encode("utf-8")
        m = int(i)

        if embedding_vector is not None:
            embedding_matrix[m, :] = embedding_vector
        else:
            res.append(word)

    for w in res:
        print(w)
    print("can't encode %d" % len(res))
    pdb.set_trace()
    np.save("semantic_concepts_embedding.npy", embedding_matrix)
    # with h5py.File("3000_train_val_glove_embedding.h5", 'w') as f:
    #     f.create_dataset('question_embedding_matrix', data=embedding_matrix)

