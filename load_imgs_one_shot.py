import sys
import numpy as np
# import pandas as pd
import PIL
# from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import cv2
import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

def loadimgs(path,n = 0):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (105, 105))
                # print(image.shape)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1

    print("Manish")
    y = np.vstack(y)
    # X = np.array(X)
    # print(X.shape)
    X = np.stack(X)
    print("Jagnani")
    return X,y,lang_dict

def initialize_weights(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    value = np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    return K.variable(value, name=name, dtype=dtype)

def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    value = np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    return K.variable(value, name=name, dtype=dtype)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net

def get_batch(batch_size, s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        pass
        # X = Xval
        # categories = val_classes
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)
    print(categories)
    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))
    # print("Hello")
    # pairs = np.array(pairs)
    # print(pairs.shape)
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        # print("get")
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        # print("put")
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)


def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        pass
        # X = Xval
        # categories = val_classes
    # print(X.shape)
    # print(categories)
    n_classes, n_examples, w, h = X.shape

    indices = rng.randint(0, n_examples, size=(N,))
    print(indices)
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)
        print(categories)

    else:  # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)
        print(categories)
    categories[0] = 154

    true_category = categories[0]
    # ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    ex1 = 1
    ex2 = 2
    print(X[categories[0], ex1].shape)
    plt.imshow(X[categories[0], ex1])
    plt.show()
    plt.imshow(X[categories[0], ex2])
    plt.show()
    test_image = np.asarray([X[true_category, ex1, :, :]] * N).reshape(N, w, h, 1)
    print(test_image.shape)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    # targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets, categories


def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets, cat = make_oneshot_task(N,s)
        plot_oneshot_task(inputs)
        probs = model.predict(inputs)
        print(probs)
        # plot_oneshot_task(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct

def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc, h , w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(pairs[0][0].reshape(105,105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# image = cv2.imread("./dataset/angry/angry_out/b1.jpg", 1)
# bigger = cv2.resize(image, ())
# plt.savefig("b1.jpg")
X,y,c=loadimgs("./dataset")

# print(X.shape)

# val_folder = './Omniglot Dataset/images_evaluation/'
# save_path = './Omniglot Dataset/data/'

# import pathlib
# abspath = pathlib.Path(os.path.join(save_path,"train.pickle")).absolute()
#
# print(abspath)
# with open(str(abspath), "wb") as f:
#     pickle.dump((X,c),f)

# Xval,yval,cval=loadimgs(val_folder)
# abspathVal = pathlib.Path(os.path.join(save_path,"val.pickle")).absolute()
# with open(str(abspathVal), "wb") as f:
#     pickle.dump((Xval,cval),f)

print("Hello")
model = get_siamese_model((105, 105, 1))
print(model.summary())

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

# with open(abspath, "rb") as f:
#     (Xtrain, train_classes) = pickle.load(f)
#
# print("Training alphabets: \n")
# print(list(train_classes.keys()))
#
# with open(abspathVal, "rb") as f:
#     (Xval, val_classes) = pickle.load(f)
#
# print("Validation alphabets:", end="\n\n")
# print(list(val_classes.keys()))

Xtrain = X
train_classes = c
# val_classes = cval

# Hyper parameters
evaluate_every = 200 # interval for evaluating on one-shot tasks
batch_size = 6
n_iter = 1000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1

model_path = './weights/'


print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    print("Train1")
    (inputs,targets) = get_batch(batch_size)
    # print(inputs.shape)
    # print(targets.shape)
    print("Train1")
    loss = model.train_on_batch(inputs, targets)
    print(loss)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss))
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc

print("Manish")
model.load_weights(os.path.join(model_path, "weights.200.h5"))

test_oneshot(model, 16, 8)

pairs, targets, cat = make_oneshot_task(16,"train","Sanskrit")
plot_oneshot_task(pairs)

print("Hello")