from keras.models import Model
from keras.layers import Embedding ,Dense, Input, Conv1D, MaxPool1D, Concatenate, Flatten, Dropout
from keras.layers import GRU, concatenate, TimeDistributed, Lambda, GRU, GRU, LSTM, Bidirectional
from keras import backend as K
import numpy as np
from sklearn.model_selection import KFold

import tensorflow
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
#from tensorflow import set_random_seed
from numpy.random import seed


np.random.seed(1337)
maxlen = 100  # max words in a sentence
max_words = 10000  # max unique words in the dataset

romanData = pd.read_csv("/home/teamlead/PycharmProjects/Roman_Urdu_Text_Classification_sentiment/data/Three_smALL.csv",
                        low_memory=False)
print(len(romanData))
text = romanData['text']
print(len(text))
labels = romanData['labels']
print(set(labels))
print(len(set(labels)))

print(len(labels))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(romanData)
print("Total documents: ", tokenizer.document_count)

cleanText = list()
for i in text:
    # print(i)
    # remove retweet
    textremoveretweet = re.sub("^(RT|rt)( @\w*)?[: ]", '', i)

    # remove url
    textwithoutUrl = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', textremoveretweet)

    # remove Tags
    textwithoutHashTag = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", textwithoutUrl).split())

    # remove extra spaces
    # textremoveSpace = i.strip()

    # remove more extra spaces
    text_remove_line = re.sub("\s\s+", " ", i)

    # remove numeric values non-english characters
    # text_eng_chars = re.sub("\S*\d\S*", "", text_remove_line).strip()

    # text_eng_chars = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text_eng_chars).split())

    # remove single character
    # text_eng_chars = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text_eng_chars)

    textengchars = text_remove_line + ' '
    textengchars = textengchars.lower()
    # apply rules
    textengchars = re.sub("ain ", "ein ", textengchars)
    # textengchars = re.sub("(?<![ ])ar", 'r', textengchars)  # problem should not for first letter
    #convert ai into ae...
    textengchars = re.sub("ai", "ae", textengchars)
    # textengchars = re.sub("iy*", "i", textengchars)
    textengchars = re.sub("ih+", "eh", textengchars)
    # textengchars = re.sub("ay ", "e ", textengchars)
    textengchars = re.sub("ey ", "e ", textengchars)
    textengchars = re.sub("s+", "s", textengchars)
    textengchars = re.sub("ie ", "y ", textengchars)
    textengchars = re.sub("ry(?<![ ])", "ri", textengchars)
    textengchars = re.sub(" es", " is", textengchars)
    textengchars = re.sub("sy(?<![ ])", "si", textengchars)
    textengchars = re.sub("a+", "a", textengchars)
    textengchars = re.sub("ty(?<![ ])", "ti", textengchars)
    textengchars = re.sub("j+", "j", textengchars)
    textengchars = re.sub("o+", "o", textengchars)
    textengchars = re.sub("ee+", "i", textengchars)
    # textengchars = re.sub("(?<!s)i ", "y ", textengchars) # check this comndition not true for "a"
    textengchars = re.sub("d+", "d", textengchars)
    textengchars = re.sub("u", "o", textengchars)
    # textengchars = re.sub("ah", "a", textengchars)
    # #textengchars = re.sub("ch", "c", textengchars)
    # textengchars = re.sub("eh", "e", textengchars)
    # textengchars = re.sub("fh", "f", textengchars)
    # textengchars = re.sub("gh", "g", textengchars)
    # textengchars = re.sub("hh", "h", textengchars)
    # textengchars = re.sub("ih", "i", textengchars)
    # textengchars = re.sub("jh", "j", textengchars)
    # textengchars = re.sub("lh", "l", textengchars)
    # textengchars = re.sub("mh", "m", textengchars)
    # textengchars = re.sub("nh", "n", textengchars)
    # #textengchars = re.sub("oh", "o", textengchars)
    # textengchars = re.sub("qh", "q", textengchars)
    # textengchars = re.sub("rh", "r", textengchars)
    # #textengchars = re.sub("sh", "s", textengchars)
    # #textengchars = re.sub("th", "t", textengchars)
    # #textengchars = re.sub("uh", "u", textengchars)
    # textengchars = re.sub("vh", "v", textengchars)
    # #textengchars = re.sub("wh", "w", textengchars)
    # textengchars = re.sub("xh", "x", textengchars)
    # #textengchars = re.sub("yh", "y", textengchars)
    # textengchars = re.sub("zh", "z", textengchars)
    # textengchars = re.sub("haha(ha)*", "haha", textengchars)
    #

    cleanText.append(textengchars + "\n")


text_with_label = []
for i in range(len(cleanText)):
    text_with_label.append(cleanText[i] + ' : ' + str(labels[i]) + ' \n')


tokenizer = Tokenizer(num_words=max_words)
print(type(tokenizer))
tokenizer.fit_on_texts(cleanText)
sequences = tokenizer.texts_to_sequences(cleanText)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

print("labels before going into split:\n ",labels )
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print("labels after Label Encoder:\n ",labels )
# labels = to_categorical(labels)
# print("labels after to_categorical:\n ",labels )

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, stratify=labels, random_state=123)

#X_val, X_test, y_val, y_test = train_test_split(test_x, test_y, test_size=.33, stratify=test_y, random_state=123)

print('Shape of X_train tensor:', X_train.shape)
print('Shape of y_train tensor:', y_train.shape)

# print('Shape of X_val tensor:', X_val.shape)
# print('Shape of y_cal tensor:', y_val.shape)

print('Shape of X_test tensor:', X_test.shape)
print('Shape of y_test tensor:', y_test.shape)

print("y_train looks like as of now: \n",y_train)



# ###Preparing the word-embeddings matrix

# ------------------------------- Embeddings -----------------------------------

f = open('/home/teamlead/PycharmProjects/Roman_Urdu_Text_Classification_sentiment/Embeddings/full_data_word2vec.txt')
embeddings_index = {}
i = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[-200:], dtype='float32')
    embeddings_index[word] = coefs
    # if i%100000 == 0:
    #     print(i)
    # i += 1
f.close()
print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 200
embedding_matrix1 = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix1[i] = embedding_vector

f = open('/home/teamlead/PycharmProjects/Roman_Urdu_Text_Classification_sentiment/Embeddings/full_data_fasttext.txt')
embeddings_index = {}
i = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[-200:], dtype='float32')
    embeddings_index[word] = coefs
    # if i%100000 == 0:
    #     print(i)
    # i += 1
f.close()
print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 200
embedding_matrix2 = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix2[i] = embedding_vector


f = open('/home/teamlead/PycharmProjects/Roman_Urdu_Text_Classification_sentiment/Embeddings/gloVe_vectors_200.txt')
embeddings_index = {}
i = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[-200:], dtype='float32')
    embeddings_index[word] = coefs
    # if i%100000 == 0:
    #     print(i)
    # i += 1
f.close()
print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 200
embedding_matrix3 = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix3[i] = embedding_vector


# --------------------------------- Word2vec -------------------------------------------

# ###Model

document       = Input(shape=(None,), dtype = "int32")
left_context   = Input(shape=(None,), dtype = "int32")
right_context  = Input(shape=(None,), dtype = "int32")

#Embedding word2vwec
document_embedding       = Embedding(max_words, embedding_dim, weights=[embedding_matrix1],
                                     input_length=maxlen, trainable=False)(document)
left_context_embedding   = Embedding(max_words, embedding_dim, weights=[embedding_matrix1],
                                     input_length=maxlen, trainable=True)(left_context)
right_context_embedding  = Embedding(max_words, embedding_dim, weights=[embedding_matrix1],
                                     input_length=maxlen, trainable=False)(right_context)

forward_rnn     = GRU(100, return_sequences = True)(left_context_embedding)
backward_rnn    = GRU(100, return_sequences = True, go_backwards = True)(right_context_embedding)
doc_rnn    = GRU(100, return_sequences = True, go_backwards = True)(document_embedding)

recurrent_layer_w2v = concatenate([forward_rnn, doc_rnn, backward_rnn], axis = 2) #document_embedding

#Embedding FastText
document_embedding       = Embedding(max_words, embedding_dim, weights=[embedding_matrix2],
                                     input_length=maxlen, trainable=False)(document)
left_context_embedding   = Embedding(max_words, embedding_dim, weights=[embedding_matrix2],
                                     input_length=maxlen, trainable=True)(left_context)
right_context_embedding  = Embedding(max_words, embedding_dim, weights=[embedding_matrix2],
                                     input_length=maxlen, trainable=False)(right_context)

forward_rnn     = GRU(100, return_sequences = True)(left_context_embedding)
backward_rnn    = GRU(100, return_sequences = True, go_backwards = True)(right_context_embedding)
doc_rnn    = GRU(100, return_sequences = True, go_backwards = True)(document_embedding)

recurrent_layer_FastText = concatenate([forward_rnn, doc_rnn, backward_rnn], axis = 2)

#Embedding Glove
document_embedding       = Embedding(max_words, embedding_dim, weights=[embedding_matrix3],
                                     input_length=maxlen, trainable=False)(document)
left_context_embedding   = Embedding(max_words, embedding_dim, weights=[embedding_matrix3],
                                     input_length=maxlen, trainable=True)(left_context)
right_context_embedding  = Embedding(max_words, embedding_dim, weights=[embedding_matrix3],
                                     input_length=maxlen, trainable=False)(right_context)

forward_rnn     = GRU(100, return_sequences = True)(left_context_embedding)
backward_rnn    = GRU(100, return_sequences = True, go_backwards = True)(right_context_embedding)
doc_rnn    = GRU(100, return_sequences = True, go_backwards = True)(document_embedding)

recurrent_layer_Glove = concatenate([forward_rnn, doc_rnn, backward_rnn], axis = 2)

recurrent_layer = concatenate([recurrent_layer_w2v, recurrent_layer_FastText, recurrent_layer_Glove], axis = 2)

# recurrent_layer2 = Bidirectional(GRU(32, return_sequences=True)) (recurrent_layer) #add
#

# convs = []
# filter_sizes = [10, 15, 20]
#
# for filter_size in filter_sizes:
#     l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='valid', activation='relu')(recurrent_layer2)
#     l_pool = MaxPool1D(filter_size)(l_conv)
#     convs.append(l_pool)
#
# l_merge = Concatenate(axis=1)(convs)
# l_flat = Flatten()(l_merge)
# cnn_dense = Dense(128, activation='relu')(l_flat)

latent_semantic_tensor = TimeDistributed(Dense(200, activation = "tanh"))(recurrent_layer)
maxpool_rcnn = Lambda(lambda x: K.max(x, axis = 1))(latent_semantic_tensor)
output_layer = Dense(3, activation = "softmax")(maxpool_rcnn)

model_1 = Model(inputs = [left_context, document, right_context], outputs=[output_layer])
model_1.summary()
model_1.summary()


y_train = to_categorical(y_train)


Optimizer = 'rmsprop'
Loss = 'categorical_crossentropy'
BATCH_SIZE = 8
Epochs = 50

model_1.compile(optimizer=Optimizer,
              loss=Loss,
              metrics=['acc'])


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model_1.fit([X_train,X_train, X_train], y_train,
                    epochs=Epochs,
                    batch_size=BATCH_SIZE,
                      validation_split=0.1,
                      verbose=2,
                      callbacks=[early_stopping])


y_test = to_categorical(y_test)


y_pred = model_1.predict([X_test, X_test,X_test])


cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)
cm_df = pd.DataFrame(cm,
                     index = ['positive', 'negative', 'neutral'],
                     columns = ['positive', 'negative', 'neutral'])
plt.figure(figsize=(5.5, 4))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.show()
# plt.savefig("4th_best.png")


print('Model_12 Accuracy:  {0:.5f}'.format(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))))
print('Model_12 Precision :  {0:.5f}'.format(precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')))
#print('Model_7 Precision Macro:  {0:.3f}'.format(precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))
print('Model_12 Recall :  {0:.5f}'.format(recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')))
#print('Model_7 Recall Macro:  {0:.3f}'.format(recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')))
print(('Model_12 F1:  {0:.5f}'.format(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted'))))
#print(('Model_7 F1 Macro:  {0:.3f}'.format(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro'))))


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
precision_micro = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
precision_macro = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
recall_micro = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
recall_macro = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
f1_micro = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
f1_macro = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')



path = "/home/teamlead/PycharmProjects/Roman_Urdu_Text_Classification_sentiment/Models/Results"
if not os.path.exists(path):
    os.makedirs(path)
filename = 'full_vocab_results.log'
with open(os.path.join(path, filename), 'a') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow([''])
    file_writer.writerow(['Multi_Channel_RCNN'])
    file_writer.writerow(['Embedding : our vary, add Bi-GRU'])
    file_writer.writerow(['Epoch', 'Loss', 'Optimizer', 'BatchSize', 'Accuracy', 'Precision', 'Precision_micro','Precision_macro','Recall','recall_micro',
                          'recall_macro', 'FMeasure','F1_micro','F1_macro' ])
    file_writer.writerow(
        [str(Epochs), Loss, Optimizer, str(BATCH_SIZE), round(accuracy, 4), round(precision, 4), round(precision_micro, 4), round(precision_macro, 4), round(recall, 4),
         round(recall_micro, 4),round(recall_macro, 4), round(f1, 4), round(f1_micro, 4),round(f1_macro, 4)])
