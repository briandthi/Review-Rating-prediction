import numpy as np
import tensorflow as tf
import csv
import re

def read_glove_vecs(glove_file = 'data/glove.6B.50d.txt'):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def read_csv(words_to_index, min_len, max_len, filename = "data/tripadvisor_hotel_reviews.csv"):
    phrase = []
    emoji = []

    with open (filename, errors="ignore") as csvDataFile:
        next(csvDataFile)
        csvReader = csv.reader(csvDataFile,)

        for row in csvReader:
            flag , first_sentence = delete_sentence(row[0], words_to_index, min_len, max_len)
            if flag:
                phrase.append(first_sentence)
                emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def delete_sentence(sentence, word_to_index, min_len, max_len):
    sentence = ponctuation_padding(sentence)
    sentence_to_words = sentence.lower().split(".")
    sentence_to_words = sentence_to_words[0].split()
    if (len(sentence_to_words) <= max_len) & (len(sentence_to_words) >= min_len):
        for w in sentence_to_words:
            try:
                word_to_index[w]
            except KeyError:
                return False, ""
        return True, sentence.lower().split(".")[0]
    else :
        return False, ""

def ponctuation_padding(sentence):
    s = re.sub("(?<! )(?=[.,!?()/'])|(?<=[.,!?()/'])(?! )", r' ', sentence)
    return s

def convert_to_oh(Y, C = 5):
    Y = tf.one_hot(Y, C, on_value=1.0, off_value = 0.0)
    return Y


def read_csv2(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y
### --test-- ###
# word_to_index, index_to_word, word_to_vec_map = read_glove_vecs()
# X, Y = read_csv(word_to_index)
# print(len(X))