import pandas as pd
from utils import * 
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

X, Y = read_csv("tripadvisor_hotel_reviews.csv")

# padding ponctuation of comments
for i in range(len(X)):
    X[i] = ponctuation_padding(X[i])

X_train, Y_train = X[0:4096], Y[0:4096]
X_valid, Y_valid = X[15000:17500], Y[15000:17500]
X_test, Y_test = X[17500:], Y[17500:]

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

maxLen = max([len(sentence.split()) for sentence in X])
print("Sentences max length: ", maxLen)

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    """
    m = X.shape[0]

    X_indices = np.zeros((m, max_len))
    missing_words = 0
    for i in range(m):

        sentence_to_words = X[i].lower().split()
        j = 0

        for w in sentence_to_words:
            try:
                X_indices[i, j] = word_to_index[w]
            except KeyError:
                missing_words += 1
                X_indices[i, j] = word_to_index[',']
            j += 1
    print("Number of missing words: ", missing_words)
    return X_indices

# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
# X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
# print("X1 =", X1)
# print("X1_indices =\n", X1_indices)

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    """
    
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    

    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]


    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

def commentary(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the commentary model's graph.
    
    """
    
    sentence_indices = Input(input_shape, dtype='int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(128, return_sequences= True)(embeddings)
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences= False)(X)
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with 5 units
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)
    
    return model

model = commentary((maxLen,), word_to_vec_map, word_to_index)
model.summary()
print("Model compile......")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_oh(Y_train)

print("Model fit......")
model.fit(X_train_indices, Y_train_oh, epochs = 10, batch_size = 16, shuffle=True)
print("End of fitting......")

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_oh(Y_test)

print("Model evaluate......")
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)