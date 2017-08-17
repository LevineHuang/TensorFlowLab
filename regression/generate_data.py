#coding=utf-8


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

def generate_data(data_num, data_dim, show_data=True):
    X, Y = make_classification(n_samples=data_num, n_features=data_dim, n_redundant=0, n_clusters_per_class=1,  n_classes=2)
    
    if show_data:
        plt.scatter(X[:,0], X[:,1], marker="o", c=Y)
        plt.title("raw data")
        plt.xlabel("X[:,0]")
        plt.ylabel("X[:,1]")
        plt.show()

    return X, Y

def make_example(features, label):
    ex = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "data": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])) # [label]必须用“[]”[]括起来
            }
        )
    )
    return ex

def generate_tfrecords(data_num, data_dim, filename):
    X, Y = generate_data(data_num, data_dim, False)
    print(X)
    print(Y)
    writer = tf.python_io.TFRecordWriter(filename)
    print(zip(X,Y))
    for x, y in zip(X, Y):
        ex = make_example(x, y)
        writer.write(ex.SerializeToString())
    writer.close()

if __name__ == "__main__":
    print("generate test data")
    generate_tfrecords(1000, 2, "lr.tfrecords")