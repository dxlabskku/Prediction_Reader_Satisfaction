import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import PIL
import scipy.sparse as sp
import resource
import time
from PIL import Image
from PIL import ImageFile
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import json
import re
import pandas
import numpy as np
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.text import Tokenizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

import warnings 
warnings.filterwarnings(action='ignore')

def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tf.Session(config=config)
    session
set_env()

###########################
## Data Load and Ready
###########################

from dataloader import ReviewCoverLoader
from data_init import DataReady

data_ready = DataReady()
data_ready.drop_reset()
data_ready.get_img_label_score()
obj_list, score_list = data_ready.get_obj_score()
df = data_ready.df

num2resizedImg = data_ready.get_imgarray()
InputSize_wh = data_ready.inputSize_wh

# review, number, label 값 저장
review = df['review'].values.astype(str)
title = df["title"].values.astype(str)
author = df["author"].values.astype(str)
publisher = df["publisher"].values.astype(str)
number = df['number'].values
label = df['label'].values

###########################
## Preprocessing and Modeling
###########################

num = 5
kf = KFold(n_splits=num, random_state=None, shuffle=False) # default -> 5 times

ohe = OneHotEncoder()
obj_ohe = ohe.fit(obj_list).transform(obj_list).toarray()
obj = obj_ohe*score_list*0.01

maxlen = 100
nEmbeddingDim = 128
neurons = 128
epochs = 50
kernel_size = 5
filters = 32 #64
pool_size = 4

batch_size = 32

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

total_result = pd.DataFrame()
accuracy_avg = 0

for ind, (train_index, test_index) in enumerate(kf.split(review)):
    print("\n{}{}th {}".format("-"*20,ind+1,"-"*20))
    train_number, train_review, train_title, train_author, train_publisher, train_obj, train_label = number[train_index], review[train_index], title[train_index], author[train_index], publisher[train_index], obj[train_index], label[train_index]
    test_number, test_review, test_title, test_author, test_publisher, test_obj, test_label = number[test_index], review[test_index], title[test_index], author[test_index], publisher[test_index], obj[test_index], label[test_index]
    valid_number, test_number, valid_review, test_review, valid_title, test_title, valid_author, test_author, valid_publisher, test_publisher, valid_obj, test_obj, valid_label, test_label = train_test_split(test_number, test_review, test_title, test_author, test_publisher, test_obj, test_label, test_size=.5)
    
    print("total #: {}".format(len(review)))
    print("train #: {}".format(len(train_review)))
    print("valid #: {}".format(len(valid_review)))
    print("test #: {}".format(len(test_review)))
    
    # review 데이터 tokenizing
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_review)
    sequencesTrain = tokenizer.texts_to_sequences(train_review)
    sequencesValid = tokenizer.texts_to_sequences(valid_review)
    sequencesTest = tokenizer.texts_to_sequences(test_review)
    
    # vocab_size 계산
    vocab_size = len(tokenizer.word_index)+1
    
    # preprocessing
    maxlen = 100
    pad_Train = tf.keras.preprocessing.sequence.pad_sequences(sequencesTrain, maxlen=maxlen)
    pad_Valid = tf.keras.preprocessing.sequence.pad_sequences(sequencesValid, maxlen=maxlen)
    pad_Test = tf.keras.preprocessing.sequence.pad_sequences(sequencesTest, maxlen=maxlen)
    
    # title 데이터 tokenizing, preprocessing
    title_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    title_tokenizer.fit_on_texts(train_title)
    title_sequencesTrain = title_tokenizer.texts_to_sequences(train_title)
    title_sequencesValid = title_tokenizer.texts_to_sequences(valid_title)
    title_sequencesTest = title_tokenizer.texts_to_sequences(test_title)
    
    t = np.array([len(a) for a in title_sequencesTrain]).max()
    
    title_pad_Train = tf.keras.preprocessing.sequence.pad_sequences(title_sequencesTrain, maxlen=t)
    title_pad_Valid = tf.keras.preprocessing.sequence.pad_sequences(title_sequencesValid, maxlen=t)
    title_pad_Test = tf.keras.preprocessing.sequence.pad_sequences(title_sequencesTest, maxlen=t)
    
    # author 데이터 tokenizing, preprocessing
    author_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    author_tokenizer.fit_on_texts(train_author)
    author_sequencesTrain = author_tokenizer.texts_to_sequences(train_author)
    author_sequencesValid = author_tokenizer.texts_to_sequences(valid_author)
    author_sequencesTest = author_tokenizer.texts_to_sequences(test_author)
    
    a = np.array([len(a) for a in author_sequencesTrain]).max()
    
    author_pad_Train = tf.keras.preprocessing.sequence.pad_sequences(author_sequencesTrain, maxlen=a)
    author_pad_Valid = tf.keras.preprocessing.sequence.pad_sequences(author_sequencesValid, maxlen=a)
    author_pad_Test = tf.keras.preprocessing.sequence.pad_sequences(author_sequencesTest, maxlen=a)
    
    # publisher 데이터 tokenizing, preprocessing
    publisher_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    publisher_tokenizer.fit_on_texts(train_publisher)
    publisher_sequencesTrain = publisher_tokenizer.texts_to_sequences(train_publisher)
    publisher_sequencesValid = publisher_tokenizer.texts_to_sequences(valid_publisher)
    publisher_sequencesTest = publisher_tokenizer.texts_to_sequences(test_publisher)
    
    p = np.array([len(a) for a in publisher_sequencesTrain]).max()
    
    publisher_pad_Train = tf.keras.preprocessing.sequence.pad_sequences(publisher_sequencesTrain, maxlen=p)
    publisher_pad_Valid = tf.keras.preprocessing.sequence.pad_sequences(publisher_sequencesValid, maxlen=p)
    publisher_pad_Test = tf.keras.preprocessing.sequence.pad_sequences(publisher_sequencesTest, maxlen=p)
    
    # review + title + author + publisher concat
    pad_Train = np.concatenate((pad_Train,title_pad_Train,author_pad_Train,publisher_pad_Train),axis=1)
    pad_Valid = np.concatenate((pad_Valid,title_pad_Valid,author_pad_Valid,publisher_pad_Valid),axis=1)
    pad_Test = np.concatenate((pad_Test,title_pad_Test,author_pad_Test,publisher_pad_Test),axis=1)
    
    print(pad_Train.shape)
    print(pad_Valid.shape)
    print(pad_Test.shape)
    
    total_len = pad_Train.shape[1]
    
    tf.keras.backend.clear_session()
    
    ####################
    # review+meta flow #
    ####################
    
    # review flow
    reviewInputTensor = tf.keras.layers.Input(shape=(total_len,))
    reviewFlow = tf.keras.layers.Embedding(vocab_size
                                       ,nEmbeddingDim,input_length=maxlen)(reviewInputTensor)
    reviewFlow = tf.keras.layers.Dropout(0.25)(reviewFlow)
    reviewFlow = tf.keras.layers.Conv1D(filters
                    ,kernel_size,padding='valid',activation='relu',strides=1)(reviewFlow)
    reviewFlow = tf.keras.layers.MaxPool1D(pool_size = pool_size)(reviewFlow)
    
    # reviewFlow = tf.keras.layers.LSTM(64)(reviewFlow)
    
    reviewFlow = tf.keras.layers.Flatten()(reviewFlow)
#     reviewFlow = tf.keras.layers.Dense(512,activation='relu')(reviewFlow)
#     reviewFlow = tf.keras.layers.BatchNormalization()(reviewFlow)
    reviewFlow = tf.keras.layers.Dense(256,activation='relu')(reviewFlow)
#     reviewFlow = tf.keras.layers.BatchNormalization()(reviewFlow)
#     reviewFlow = tf.keras.layers.Dense(128,activation='relu')(reviewFlow)
#     reviewFlow = tf.keras.layers.Dense(64,activation='relu')(reviewFlow)
    #softmax = tf.keras.layers.Dense(6,activation='softmax')(reviewFlow)
    
    ###############
    # image flow #
    ##############
    
    # image flow 
    imageInputTensor = tf.keras.layers.Input(shape=(InputSize_wh, InputSize_wh, 3))

    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imageInputTensor)
    imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
    imgflow = tf.keras.layers.MaxPooling2D(pool_size=(4,4))(imgflow)
    imgflow = tf.keras.layers.Dropout(0.25)(imgflow)

#     imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
#     imgflow = tf.keras.layers.MaxPooling2D(pool_size=(4,4))(imgflow)
#     imgflow = tf.keras.layers.Dropout(0.25)(imgflow)

#     imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
#     imgflow = tf.keras.layers.Dropout(0.25)(imgflow)

#     imgflow = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = 'relu')(imgflow)
#     imgflow = tf.keras.layers.Dropout(0.25)(imgflow)

    imgflow = tf.keras.layers.Flatten()(imgflow)
    imgflow = tf.keras.layers.Dense(512,activation='relu')(imgflow)
    imgflow = tf.keras.layers.BatchNormalization()(imgflow)
    imgflow = tf.keras.layers.Dense(256,activation='relu')(imgflow)
    imgflow = tf.keras.layers.BatchNormalization()(imgflow)
    imgflow = tf.keras.layers.Dense(128,activation='relu')(imgflow)
    imgflow = tf.keras.layers.Dense(64,activation='relu')(imgflow)
    
    ###############
    # object flow #
    ##############
    
    # object flow
    objectInputTensor = tf.keras.layers.Input(shape=(obj.shape[1],))
    objectflow = tf.keras.layers.Dense(32,activation='relu')(objectInputTensor)
#     objectflow = tf.keras.layers.BatchNormalization()(objectflow)
#     objectflow = tf.keras.layers.Dense(32,activation='relu')(objectflow)
    
    ###############
    # concat flow #
    ##############
    
    # concat

    flow = tf.keras.layers.concatenate(inputs=[reviewFlow,imgflow, objectflow])
    # flow = reviewFlow
#     flow = tf.keras.layers.Dense(512,activation='relu')(flow)
#     flow = tf.keras.layers.BatchNormalization()(flow)
#     flow = tf.keras.layers.Dense(512,activation='relu')(flow)
#     flow = tf.keras.layers.BatchNormalization()(flow)
#     flow = tf.keras.layers.Dense(64,activation='relu')(flow)
    softmax = tf.keras.layers.Dense(6,activation='softmax')(flow)

    model = tf.keras.Model(inputs=[reviewInputTensor,imageInputTensor,objectInputTensor],
                          outputs=[softmax])
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    print('\nClass Weight: {}\n'.format(class_weights))
    
    adam = tf.keras.optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    # model compile
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])
    
    checkpoint_file = "504-checkpoint_21.h5"
    callback = tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', 
                                patience = 5)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            checkpoint_file,             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                              verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                            save_weights_only=True,
                             mode='auto'           
                            )
    
    # model fit
    print("batch size: {}".format(batch_size))
    model.fit_generator(
#     verbose = 1,
    generator = ReviewCoverLoader(train_number,pad_Train,num2resizedImg,train_obj,train_label,batch_size), 
    validation_data = ReviewCoverLoader(valid_number,pad_Valid, num2resizedImg, valid_obj, valid_label, batch_size),
#     validation_steps=len(valid_label)//batch_size,
    epochs = epochs,
    callbacks=[callback,checkpoint],
    workers=10, use_multiprocessing=True, class_weight=class_weights)
    
    model.load_weights(checkpoint_file)
    
    fused_predict = model.predict_generator(ReviewCoverLoader(
        test_number,pad_Test,num2resizedImg,test_obj,test_label,batch_size,shuffle=False))
    report_score = classification_report(test_label[:len(fused_predict)], fused_predict.argmax(axis=1),digits=4, output_dict=True)
    accuracy_avg+=report_score["accuracy"]
    report_score = pd.DataFrame(report_score)
    
    if(ind==0):total_result = report_score
    else:total_result += report_score
    
    print('\n',report_score.T)
    print()

print("평균 accuracy: ",accuracy_avg/num)
print(total_result/num)