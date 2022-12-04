from commons.embedding import read_sentences_from_file, write_sentences_to_file
import numpy as np
import pandas as pd


def rm_duplication(benign_file='../embedding_dataset/origin_dataset_benign.txt',malicious_file='../embedding_dataset/origin_dataset_malicious.txt'):
    # 获取良性数据
    benign_sentences = read_sentences_from_file(benign_file)
    benign_sentences = list(set(benign_sentences))

    malicious_sentences = read_sentences_from_file(malicious_file)
    malicious_sentences = list(set(malicious_sentences))

    with open(file=benign_file, mode='w+') as f:
        for sentence in benign_sentences:
            f.write(sentence + '\n')

    with open(file=malicious_file, mode='w+') as f:
        for sentence in malicious_sentences:
            f.write(sentence + '\n')

    return

def label_dataset(benign_file='../embedding_dataset/origin_dataset_benign.txt',malicious_file='../embedding_dataset/origin_dataset_malicious.txt'):
    pass







if __name__ == "__main__":
    rm_duplication()
    # label_dataset()