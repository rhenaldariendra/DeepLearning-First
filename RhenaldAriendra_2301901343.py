import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def import_data():
  df = pd.read_csv("gender_classification.csv")
  
  x = df[["long_hair","forehead_width_cm","forehead_height_cm","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"]]
  y = df[["gender"]]
  return x, y

input_dataset, output_dataset = import_data()

# Data Preprocessing
scaler = MinMaxScaler()
input_dataset = scaler.fit_transform(input_dataset)
output_dataset = OneHotEncoder(sparse= False).fit_transform(output_dataset)
x_train, x_test, y_train, y_test = train_test_split(input_dataset, output_dataset, test_size = 0.8)

# Dictionary

layers = {
    'input' : 7,
    'hidden' : 7,
    'output' : 2
}

input_hidden = {
    "weight": tf.Variable(tf.random.normal([layers['input'], layers['hidden']])),
    "bias": tf.Variable(tf.random.normal([layers['hidden']]))
}

hidden_output = {
    "weight": tf.Variable(tf.random.normal([layers['hidden'], layers['output']])),
    "bias": tf.Variable(tf.random.normal([layers['output']]))
}

def activation(output):
    return tf.nn.sigmoid(output)

def feed_forward(input_data):
    # Input ke Hidden
    x1 = tf.matmul(input_data, input_hidden["weight"] + input_hidden["bias"])
    y1 = activation(x1)

    # Hidden ke Output
    x2 = tf.matmul(y1, hidden_output["weight"] + hidden_output["bias"])
    y2 = activation(x2)

    return y2, y1

# Placeholder
input_placeholder = tf.placeholder(tf.float32, [None, layers['input']])
output_placeholder = tf.placeholder(tf.float32, [None, layers['output']])

output, hiddenOut = feed_forward(input_placeholder)
error = tf.reduce_mean(0.5 * (output_placeholder - output)** 2)

learning_rate = 0.1
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

epoch = 5000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    hidden_dict = {
        input_placeholder   : x_train,
        output_placeholder  : y_train
    }

    print(f'"Hidden": {sess.run(hiddenOut, feed_dict = hidden_dict) * 100} %\n')
    tf.summary.scalar('Loss', error)
    tf.summary.histogram('Weight to Hidden', input_hidden[  'weight'])
    

    writer = tf.summary.FileWriter(logdir = './summary')
    summary = tf.summary.merge_all()

    for i in range(epoch):

        train_dict = {
            input_placeholder : x_train,
            output_placeholder : y_train
        }

        sess.run(train, feed_dict = train_dict)
        loss = sess.run(error, feed_dict = train_dict)
        
        if i % 200 == 0:
            print(f"Epoch: {i}, Loss: {loss}") 

    accuracy = tf.equal(tf.argmax(output_placeholder, axis = 1), tf.argmax(output, axis = 1))
    result = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    
    test_dict = {
        input_placeholder : x_test,
        output_placeholder : y_test
    }

    print("Accuracy : {}%\n" .format(sess.run(result, feed_dict = test_dict) * 100))