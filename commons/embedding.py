import os
import time

import pickle
import gensim
import numpy as np
import smart_open
import logging
import collections
import random


def timer_check(prompt, timer):
    print(f"===== {prompt}: {time.time() - timer}")
    return time.time()


# 训练语料库包含标签，标签名为行号。测试语料库只是一个列表，不包含任何标签。
def read_corpus_from_list(train_data: list, tokens_only=False):
    for i, line in enumerate(train_data):
        # tokens = gensim.utils.simple_preprocess(line)
        tokens = line.split(' ')
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


# 训练pv-dm模型
def train_pvdm_model(train_data: list):
    train_corpus = list(read_corpus_from_list(train_data))
    # print(train_corpus[:20])
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, epochs=2000)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # 对模型进行评价
    evaluate_model(train_corpus, model)

    return model


# 使用训练好的模型对一个sentence进行embedding
def convert_sentence_to_vector(sentence: str, model: gensim.models.Doc2Vec):
    # tokens = gensim.utils.simple_preprocess(sentence)
    # timer = time.time()
    tokens = sentence.split(' ')
    vector = model.infer_vector(tokens)
    # timer = timer_check("pvdm嵌入路径一条路径", timer)

    return vector


def evaluate_model(train_corpus, model: gensim.models.Doc2Vec):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    counter = collections.Counter(ranks)
    print(counter)


def write_sentences_to_file(sentences, file):
    """
    将良性sentences以追加方式写入文件,作为训练样本
    :param sentences:
    :param file:
    :return:
    """
    with open(file=file, mode='a+') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    return


def read_sentences_from_file(file='../embedding_dataset/embedding_data.txt'):
    """
    将良性sentences训练样本读出
    :param file:
    :return:
    """
    with open(file=file, mode='r') as f:
        # sentences = f.readlines()
        sentences = f.read().splitlines()

    return sentences


def train_model_from_benign_dataset(file='../embedding_dataset/embedding_data.txt'):
    """
    通过良性数据训练 pvdm model
    :param file:
    :return:
    """
    sentences = read_sentences_from_file(file)
    sentences = list(set(sentences))
    random.shuffle(sentences)
    pvdm_model = train_pvdm_model(sentences)

    return pvdm_model


def get_trained_pvdm_model(file='../pvdm_model/pvdm.pkl'):
    """
    得到训练好的pvdm model
    :param file:
    :return:
    """
    pic = open(file, 'rb')
    pvdm_model = pickle.load(pic)
    return pvdm_model


def store_benign_sentences():
    """
    预存良性数据的数组
    :return:
    """
    pvdm_model = get_trained_pvdm_model("../pvdm_model/pvdm.pkl")
    embedded_benign_sentences = []
    # 获取良性数据
    benign_sentences = read_sentences_from_file('../embedding_dataset/embedding_data.txt')
    benign_sentences = list(set(benign_sentences))

    for sentence in benign_sentences:
        embedded_sentence = convert_sentence_to_vector(sentence, pvdm_model)
        embedded_benign_sentences.append(embedded_sentence)

    embedded_benign_sentences = np.array(embedded_benign_sentences)

    pic = open(r'../embedding_dataset/embedded_benign_sentences.pkl', 'wb')
    pickle.dump(embedded_benign_sentences, pic)
    pic.close()



if __name__ == "__main__":
    # 1.训练模型
    pvdm_model = train_model_from_benign_dataset()
    pic = open(r'../pvdm_model/pvdm.pkl', 'wb')
    pickle.dump(pvdm_model, pic)
    pic.close()

    # 2.推理良性数据，对结果进行存储
    # store_benign_sentences()


    # s = "hello world"
    # vector1 = convert_sentence_to_vector(s, pvdm_model)
    # print(vector1)
    # pvdm_model = get_trained_pvdm_model()
    #
    # vector2 = convert_sentence_to_vector(s, pvdm_model)
    #
    # print(vector2)
    # print(vector1 ==vector2)