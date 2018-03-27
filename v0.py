# coding: utf-8
import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input

import argparse
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Fashion AI')
parser.add_argument('cur_class_idx', default=1, type=int, help='The index of label to classify')
parser.add_argument('--width', default=299, type=int, help='width of input data')
parser.add_argument('--resume', default='', type=str, help='checkpoint path')
parser.add_argument('--backbone', default='InceptionResNetV2', type=str, help='Backbone Structure')
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('-vb', '--val_batch_size', default=256, type=int)
# TODO
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

args = parser.parse_args()

classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels',
           'pant_length_labels']
cur_class = classes[args.cur_class_idx]

def load_train_data(cur_class):
    train_root = "/runspace/liubin/tianchi2018_fashion-tag/data/fashionAI_attributes_train_20180222"
    csv_path = os.path.join(train_root, "Annotations/label.csv")
    df_train = pd.read_csv(csv_path, header=None)
    df_train.columns = ['image_id', 'class', 'label']
    df_load = df_train[(df_train['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']

    n = len(df_load)
    n_class = len(df_load['label'][0])

    X = np.zeros((n, args.width, args.width, 3), dtype=np.uint8)
    y = np.zeros((n, n_class), dtype=np.uint8)

    for i in tqdm(range(n)):
        tmp_label = df_load['label'][i]
        if len(tmp_label) > n_class:
            print(df_load['image_id'][i])
        X[i] = cv2.resize(cv2.imread(os.path.join(train_root, df_load['image_id'][i])), (args.width, args.width))
        y[i][tmp_label.find('y')] = 1
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=42)
    return X_train, X_valid, y_train, y_valid

def load_test_data(cur_class):
    test_root = "/runspace/liubin/tianchi2018_fashion-tag/data/fashionAI_attributes_test_a_20180222"
    df_test = pd.read_csv(os.path.join(test_root, 'Tests/question.csv'), header=None)
    df_test.columns = ['image_id', 'class', 'x']
    del df_test['x']

    df_load = df_test[(df_test['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']

    n = len(df_load)
    X_test = np.zeros((n, args.width, args.width, 3), dtype=np.uint8)

    for i in tqdm(range(n)):
        X_test[i] = cv2.resize(cv2.imread(os.path.join(test_root, df_load['image_id'][i])), (args.width, args.width))
    return df_load, X_test


def build_model(n_class):
    if args.backbone == "InceptionResNetV2":
        cnn_model = InceptionResNetV2(include_top=False, input_shape=(args.width, args.width, 3), weights='imagenet')
    elif args.backbone == "ResNet50":
        cnn_model = ResNet50(include_top=False, input_shape=(args.width, args.width, 3), weights='imagenet')
    else:
        print("Unsupported backbone")
    inputs = Input((args.width, args.width, 3))

    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax', name='softmax')(x)

    model = Model(inputs, x)

    if args.resume != "":
        model.load_weights(args.resume)

    return model

def write_results(df_load, test_np, save_path):
    result = []

    for i, row in df_load.iterrows():
        tmp_list = test_np[i]
        tmp_result = ''
        for tmp_ret in tmp_list:
            tmp_result += '{:.4f};'.format(tmp_ret)

        result.append(tmp_result[:-1])

    df_load['result'] = result
    df_load.to_csv(save_path, header=None, index=False)


def main():
    prefix_cls = cur_class.split('_')[0]
    X_train, X_valid, y_train, y_valid = load_train_data(cur_class)
    model = build_model(y_train.shape[1])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    filepath = os.path.join('models', '{epoch:02d}_{val_acc:.2f}' + '_{}_{}.best.h5'.format(prefix_cls, args.backbone))
    checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

    h = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.max_epoch,
                callbacks=[checkpointer],
                shuffle=True,
                validation_split=0.1)
    model.evaluate(X_valid, y_valid, batch_size=args.val_batch_size)

    # test and write results
    df_load, X_test = load_test_data(cur_class)
    test_np = model.predict(X_test, batch_size=args.val_batch_size)
    write_results(df_load, test_np, './result/{}_0326a.csv'.format(prefix_cls))

if __name__ == "__main__":
    main()
