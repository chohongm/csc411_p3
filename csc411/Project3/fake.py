import numpy as np
import math
from random import shuffle
import os
#from pylab import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from logistic_regression_classifier import LogisticRegression

import operator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz

fn_fake, fn_real = 'clean_fake.txt', 'clean_real.txt'

# Loads data into categorical sets for each class where each data is a list of words in a line.
def load_data(fn, class_label, limit=None):
    # class_label is 0 for non-spam/real, and 1 for spam/fake
    
    train_ratio, test_ratio, validation_ratio = 0.70, 0.15, 0.15
    ratios = [train_ratio, test_ratio, validation_ratio]

    with open(fn, 'r') as f:
        lines = f.readlines()
        # For part 4 and 6 use only 1500 data size. If entire data set is used, 
        # performance accuracy will be 100% for both with and without 
        # regularization (enough data to classify without help of regularization) 
        # so won't be able to see the difference.
        if limit is not None:
            lines = lines[:1500]
        lines = np.array([line.split() for line in lines])
        f.close()
    
    num_lines = len(lines)
    sizes = [int(num_lines * ratio) for ratio in ratios]
    train_end_idx = sizes[0]
    test_end_idx = train_end_idx + sizes[1]
    validation_end_idx = test_end_idx + sizes[2]
    
    # separate data
    train_xs = lines[:train_end_idx]
    test_xs = lines[train_end_idx:test_end_idx]
    validation_xs = lines[test_end_idx:validation_end_idx]
    
    # build labels
    train_ys = np.empty(sizes[0])
    test_ys = np.empty(sizes[1])
    validation_ys = np.empty(sizes[2])
    train_ys.fill(class_label)
    test_ys.fill(class_label)
    validation_ys.fill(class_label)
    
    return train_xs, test_xs, validation_xs, train_ys, test_ys, validation_ys

def get_words_counts(data_real, data_fake):
    words_counts = {}

    for hl in data_real:
        hl = list(set(hl))  # remove duplicate words
        for word in hl:
            if word not in words_counts:
                words_counts[word] = [1, 0]
            else:
                words_counts[word][0] += 1

    for hl in data_fake:
        hl = list(set(hl))  # remove duplicate words
        for word in hl:
            if word not in words_counts:
                words_counts[word] = [0, 1]
            else:
                words_counts[word][1] += 1

    return words_counts

def get_prob_words_given_label(words_counts, label, num_labeled_data, m, p):
    # again, label is 0 for real and 1 for fake. This number refers to index
    # of words_counts[word] in which word count for that label is stored.
    P_w_l = {}
    for word, counts in words_counts.items():
        P_w_l[word] = (counts[label] + m*p) / float((num_labeled_data + m))

    return P_w_l

def get_product_of_small_nums(small_nums):
    prod = 0
    
    for small_num in small_nums:
        try:
            prod += math.log(float(small_num))
        except ValueError:
            print "ValueError:", small_num
    
    return math.exp(prod)

def get_prob_of_hl_given_label(P_w_l, words_in_line):

    P_words_in_hl = np.empty([len(P_w_l)])
    i = 0
    for word, P_w in P_w_l.items():
        if word in words_in_line:
            P_words_in_hl[i] = P_w
        else:
            P_words_in_hl[i] = 1 - P_w
        i += 1

    return P_words_in_hl
    
def get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, xs_all):

    P_hl_f = [get_product_of_small_nums(get_prob_of_hl_given_label(P_w_f, hl)) for hl in xs_all]
    P_hl_r = [get_product_of_small_nums(get_prob_of_hl_given_label(P_w_r, hl)) for hl in xs_all]
    P_hl = [((P_f * P_hl_f[i]) + (P_r * P_hl_r[i])) for i in range(len(xs_all))]

    # print "Probabilities for word: ", xs_all[0]
    # print P_hl_f, P_hl_r, P_hl;

    # P_f_hl = P_f * P_hl_f
    P_f_hl = np.array([((P_f * P_hl_f[i]) / float(P_hl[i])) for i in range(len(xs_all))])
    P_r_hl = np.array([((P_r * P_hl_r[i]) / float(P_hl[i])) for i in range(len(xs_all))])

    # print P_f_hl, P_r_hl
    return P_f_hl, P_r_hl

def get_NB_probs_presence(P_r, P_f, P_w_r, P_w_f):

    P_w = (P_w_r * P_r) + (P_w_f * P_f)

    P_r_w = (P_w_r * P_r) / float(P_w)
    P_f_w = (P_w_f * P_f) / float(P_w)

    # print P_f_w, P_r_w

    return P_f_w, P_r_w

def get_NB_probs_absence(P_r, P_f, P_w_r, P_w_f):

    P_w = (P_w_r * P_r) + (P_w_f * P_f)

    P_r_w = ((float(1) - P_w_r) * P_r) / (float(1) - float(P_w))
    P_f_w = ((float(1) - P_w_f) * P_f) / (float(1) - float(P_w))

    return P_f_w, P_r_w

def remove_stopwords(pres_f, pres_r, abs_f, abs_r):

    for stopword in ENGLISH_STOP_WORDS:
        if stopword in pres_f:
            pres_f.remove(stopword)
        if stopword in pres_r:
            pres_r.remove(stopword)
        if stopword in abs_f:
            abs_f.remove(stopword)
        if stopword in abs_r:
            abs_r.remove(stopword)

def get_keywords_list(dataset):

    words = []

    for hl in dataset:
        # words_list = hl.split()
        for word in hl:
            if word not in words:
                words.append(word)

    return words

def create_hl_vector(x, y, train_set):

    # get words in training set
    words = get_keywords_list(train_set)
    hl_x = []

    for hl in x:
        hl_words = []
        for word in words:
            if word in hl:
                hl_words.append(float(1))
            else:
                hl_words.append(float(0))
        hl_x.append(hl_words)

    hl_x = np.vstack(tuple(hl_x)).astype('float64')
    hl_y = np.vstack(tuple(y))

    return hl_x, hl_y

def create_hl_matrix(headlines, labels, train_set):
    # get words in training set
    all_words = np.unique(np.hstack(train_set))
    n = (len(all_words))  # num_features
    m = len(headlines)  # num_samples
    xs = np.empty((m, n), float)
    ys = np.empty((m, 2), int)

    for i, hl in enumerate(headlines):
        x = np.empty(n)
        for j, word in enumerate(all_words):
            if word in hl:
                x[j] = float(1)
            else:
                x[j] = float(0)
        xs[i] = x

        y = [0, 1] if labels[i] == 1 else [1, 0]
        ys[i] = y

    return xs, ys

def get_accuracy(target, prediction):

    count = 0
    for i in range(len(target)):
        if prediction[i][0] >= 0.5 and target[i][0] == 1:
            count += 1
        elif prediction[i][0] < 0.5 and target[i][0] == 0:
            count += 1

    acc = (count / float(len(target))) * 100
    acc = round(acc, 2)

    return acc

def vectorize_data_for_regression(train_xs_r, test_xs_r, validation_xs_r, \
                                  train_ys_r, test_ys_r, validation_ys_r, \
                                  train_xs_f, test_xs_f, validation_xs_f, \
                                  train_ys_f, test_ys_f, validation_ys_f):
    train_xs = np.concatenate((train_xs_r, train_xs_f))
    train_ys = np.concatenate((train_ys_r, train_ys_f))
    validation_xs = np.concatenate((validation_xs_r, validation_xs_f))
    validation_ys = np.concatenate((validation_ys_r, validation_ys_f))
    test_xs = np.concatenate((test_xs_r, test_xs_f))
    test_ys = np.concatenate((test_ys_r, test_ys_f))

    X_train, Y_train = create_hl_matrix(train_xs, train_ys, train_xs)
    X_validation, Y_validation = create_hl_matrix(validation_xs, validation_ys, train_xs)
    X_test, Y_test = create_hl_matrix(test_xs, test_ys, train_xs)

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def perform_logistic_regression(train_xs, train_ys, validation_xs, validation_ys, r_flag):

    # Hyper Parameters
    m, n = train_xs.shape
    input_size = n
    num_classes = 2
    num_epochs = 75
    num_batches = 18
    learning_rate = 0.001
    if r_flag is not None:
        # use regularization
        reg_param = 10000

    # LR model
    model = LogisticRegression(input_size, num_classes)

    # training set using minibatch
    x_batches, y_batches = get_minibatch(train_xs, train_ys, num_batches)
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    for i in range(num_batches):
        x_batches[i] = Variable(torch.from_numpy(x_batches[i]), requires_grad=False).type(dtype_float)
        y_batches[i] = Variable(torch.from_numpy(np.argmax(y_batches[i], 1)), requires_grad=False).type(dtype_long)

    # sets for computing performance
    x_validation = Variable(torch.from_numpy(validation_xs), requires_grad=False).type(dtype_float)
    y_validation = Variable(torch.from_numpy(np.argmax(validation_ys, 1)), requires_grad=False).type(dtype_long)
    x_train = Variable(torch.from_numpy(train_xs), requires_grad=False).type(dtype_float)
    y_train = Variable(torch.from_numpy(np.argmax(train_ys, 1)), requires_grad=False).type(dtype_long)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    perfs_train = []
    perfs_validation = []
    epochs = []
    # Training the Model
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(x_batches, y_batches)):
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            if r_flag is not None:
                # only do this if regularization flag is not None
                params, gradParams = model.parameters()
                for theta in params:
                    loss += (reg_param / (2 * m)) * torch.sum(torch.pow(theta, 2))

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            print 'Epoch: [%d/%d], Batch: [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, num_batches, loss.data[0])
        epochs.append(epoch)

        # Train set performance
        y_pred_train = model(x_train).data.numpy()
        perf_train = np.mean(np.argmax(y_pred_train, 1) == np.argmax(y_train, 1))
        perfs_train.append(perf_train)
        print 'Training set performance: {0:.2f}%'.format(perf_train * 100)

        # Validation set performance
        y_pred_validation = model(x_validation).data.numpy()
        perf_validation = np.mean(np.argmax(y_pred_validation, 1) == np.argmax(y_validation, 1))
        perfs_validation.append(perf_validation)
        print 'Validation set performance: {0:.2f}%'.format(perf_validation * 100)
        print "\n"

    return epochs, perfs_train, perfs_validation, model

def get_minibatch(x, y, num_batches=10, num_classes=2):
    x_batches = []
    y_batches = []
    total = x.shape[0]
    class_size = int(total / num_classes)
    batch_size_per_class = int(class_size / num_batches)

    for i in range(num_batches):
        x_batch, y_batch = [], []

        for j in range(num_classes):
            from_i = j * class_size + i * batch_size_per_class
            to_i = j * class_size + (i + 1) * batch_size_per_class
            x_batch.append(x[from_i:to_i, :])
            y_batch.append(y[from_i:to_i, :])

        x_batches.append(np.vstack(x_batch))
        y_batches.append(np.vstack(y_batch))

    return x_batches, y_batches

def get_top_ten_min_max_words(trained_weights, all_words, include_stopwords=True):
    sorted_ws_idxs = trained_weights.argsort()

    max_ten_words, min_ten_words = {}, {}
    if include_stopwords:
        max_ten_ws_idxs = sorted_ws_idxs[-10:]
        min_ten_ws_idxs = sorted_ws_idxs[:10]
        max_words = all_words[max_ten_ws_idxs]
        min_words = all_words[min_ten_ws_idxs]
        for i, j in zip(max_ten_ws_idxs, min_ten_ws_idxs):
            max_word = all_words[i]
            min_word = all_words[j]
            max_ten_words[max_word] = round(trained_weights[i], 4)
            min_ten_words[min_word] = round(trained_weights[j], 4)
    else:
        count = 0
        for w_i in reversed(sorted_ws_idxs):
            word = all_words[w_i]
            if word not in ENGLISH_STOP_WORDS:
                max_ten_words[word] = round(trained_weights[w_i], 4)
                count += 1
                if count == 10:
                    break
        count = 0
        for w_i in sorted_ws_idxs:
            word = all_words[w_i]
            if word not in ENGLISH_STOP_WORDS:
                min_ten_words[word] = round(trained_weights[w_i], 4)
                count += 1
                if count == 10:
                    break

    return max_ten_words, min_ten_words

def part1(train_real, train_fake):
    # get words count from each real and fake dataset
    words_counts = get_words_counts(train_real, train_fake)
    words_counts.pop('trump')

    real_common = get_prob_words_given_label(words_counts, 0, len(train_real), 1, 0.1)
    fake_common = get_prob_words_given_label(words_counts, 1, len(train_fake), 1, 0.1)

    print "5 most common words in real headlines: "
    print sorted(real_common.items(), key=operator.itemgetter(1), reverse=True)[:5]
    print "5 most common words in fake headlines: "
    print sorted(fake_common.items(), key=operator.itemgetter(1), reverse=True)[:5]
    print "\n"

    print "3 keywords that may be useful: "
    print "Word: 'donald'"
    print "Probability of appearing in real headlines: ", real_common['donald']
    print "Probability of appearing in fake headlines: ", fake_common['donald']
    print "Word: 'the'"
    print "Probability of appearing in real headlines: ", real_common['the']
    print "Probability of appearing in fake headlines: ", fake_common['the']
    print "Word: 'us'"
    print "Probability of appearing in real headlines: ", real_common['us']
    print "Probability of appearing in fake headlines: ", fake_common['us']

def part2_graph(params, accs):
    x = range(len(params))
    y = accs
    labels = ['m = {}\np = {}'.format(param[0], param[1]) for param in params]

    plt.plot(x, y)
    plt.xticks(x, labels, rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(left=0.125, bottom=0.25, right=0.9, top=0.9)

    plt.xlabel("Values for m and p")
    plt.ylabel("Validation accuracy")
    plt.title("Validation Accuracy with Varying Values of m and p")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(os.getcwd() + '/part2_graph.png')

def part2(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, \
          validation_ys_f, test_xs_r, test_xs_f, test_ys_r, test_ys_f):
    # Refer to page 18-23 in http://www.teach.cs.toronto.edu/~csc411h/winter/lec/week5/generative.pdf
    words_counts = get_words_counts(train_xs_r, train_xs_f)

    num_real_data = len(train_ys_r)
    num_fake_data = len(train_ys_f)
    num_total_data = num_real_data + num_fake_data
    validation_xs_all = np.concatenate((validation_xs_f, validation_xs_r))
    validation_ys_all = np.concatenate((validation_ys_f, validation_ys_r))
    train_xs_all = np.concatenate((train_xs_f, train_xs_r))
    train_ys_all = np.concatenate((train_ys_f, train_ys_r))
    test_xs_all = np.concatenate((test_xs_f, test_xs_r))
    test_ys_all = np.concatenate((test_ys_f, test_ys_r))

    P_r = num_real_data / float(num_total_data)
    P_f = 1 - P_r

    ms = [0.01, 0.1, 1, 10, 100]
    ps = [0.00001, 0.001, 0.1]
    params = []
    accs = []

    print "Naive-Bayes classification (validation performance)\n"
    i = 1
    for m in ms:
        for p in ps:
            P_w_r = get_prob_words_given_label(words_counts, 0, num_real_data, m, p)
            P_w_f = get_prob_words_given_label(words_counts, 1, num_fake_data, m, p)

            P_f_hl, P_r_hl = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, validation_xs_all)
            # Since there are more than one class, P_f_hl = (P_f * P_hl_f) / sum(P_c * P_hl_c for each class c)
            predicted_ys = np.round(P_f_hl / (P_f_hl + P_r_hl))
            validation_accuracy = np.sum(predicted_ys == validation_ys_all) / float(len(validation_ys_all))
            print "===== Test {} =====\nm: {}\np: {}\naccuracy: {}\n".format(i, m, p, validation_accuracy)

            params.append(tuple((m, p)))
            accs.append(validation_accuracy)

            i += 1

    # plot graph of test results
    part2_graph(params, accs)
    plt.clf()


    P_w_r = get_prob_words_given_label(words_counts, 0, num_real_data, 1, 0.1)
    P_w_f = get_prob_words_given_label(words_counts, 1, num_fake_data, 1, 0.1)

    P_f_hl_train, P_r_hl_train = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, train_xs_all)
    predicted_ys_train = np.round(P_f_hl_train / (P_f_hl_train + P_r_hl_train))
    train_accuracy = np.sum(predicted_ys_train == train_ys_all) / float(len(train_ys_all))

    P_f_hl_val, P_r_hl_val = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, validation_xs_all)
    predicted_ys_val = np.round(P_f_hl_val / (P_f_hl_val + P_r_hl_val))
    val_accuracy = np.sum(predicted_ys_val == validation_ys_all) / float(len(validation_ys_all))

    P_f_hl_test, P_r_hl_test = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, test_xs_all)
    predicted_ys_test = np.round(P_f_hl_test / (P_f_hl_test + P_r_hl_test))
    test_accuracy = np.sum(predicted_ys_test == test_ys_all) / float(len(test_ys_all))

    print "Final training, validation, and test performance for Naive Bayes: "
    print "Training accuracy: {0:.2f}%".format(train_accuracy * 100)
    print "Validation accuracy: {0:.2f}%".format(val_accuracy * 100)
    print "Test accuracy: {0:.2f}%\n".format(test_accuracy * 100)

def part3(train_xs_r, train_xs_f, train_ys_r, train_ys_f):

    words_counts = get_words_counts(train_xs_r, train_xs_f)

    num_real_data = len(train_ys_r)
    num_fake_data = len(train_ys_f)
    num_total_data = num_real_data + num_fake_data

    P_r = num_real_data / float(num_total_data)
    P_f = 1 - P_r

    m = 1
    p = 0.1
    P_w_r = get_prob_words_given_label(words_counts, 0, num_real_data, m, p)
    P_w_f = get_prob_words_given_label(words_counts, 1, num_fake_data, m, p)

    words = words_counts.keys()

    # compute NB probs of f & r given word for each word in the entire data set.
    # the top ten in Ps_f_w represents the ten words whose presence most strongly predicts that the news is fake.
    Ps_f_w, Ps_r_w = {}, {}
    Ps_f_nw, Ps_r_nw = {}, {}
    for word in words:
        P_f_w, P_r_w = get_NB_probs_presence(P_r, P_f, P_w_r[word], P_w_f[word])
        Ps_f_w[word] = P_f_w
        Ps_r_w[word] = P_r_w
        P_f_nw, P_r_nw = get_NB_probs_absence(P_r, P_f, P_w_r[word], P_w_f[word])
        Ps_f_nw[word] = P_f_nw
        Ps_r_nw[word] = P_r_nw

    pres_f = sorted(Ps_f_w.keys(), key=Ps_f_w.get, reverse=True)
    pres_r = sorted(Ps_r_w.keys(), key=Ps_r_w.get, reverse=True)
    abs_f = sorted(Ps_f_nw.keys(), key=Ps_f_nw.get, reverse=True)
    abs_r = sorted(Ps_r_nw.keys(), key=Ps_r_nw.get, reverse=True)

    print "(a) Including stop-words --------------------------------------------------"
    print "10 words whose presence most strongly predicts that the news is real: "
    print pres_r[:10]
    print "10 words whose absence most strongly predicts that the news is real: "
    print abs_r[:10]
    print "10 words whose presence most strongly predicts that the news is fake: "
    print pres_f[:10]
    print "10 words whose absence most strongly predicts that the news is real: "
    print abs_f[:10]
    print "\n"

    remove_stopwords(pres_f, pres_r, abs_f, abs_r)

    print "(b) Not including stop-words --------------------------------------------------"
    print "10 non-stopwords whose presence most strongly predicts that the news is real: "
    print pres_r[:10]
    print "10 non-stopwords whose absence most strongly predicts that the news is real: "
    print abs_r[:10]
    print "10 non-stopwords whose presence most strongly predicts that the news is fake: "
    print pres_f[:10]
    print "10 non-stopwords whose absence most strongly predicts that the news is real: "
    print abs_f[:10]

def part4_graph(epochs, perfs_train, perfs_validation, r_flag):

    # Change this plot setup depending on with/without regularization
    x_axis = epochs
    plt.plot(x_axis, perfs_train)
    plt.plot(x_axis, perfs_validation)
    if r_flag is not None:
        plt.title('Performance Curve with Regularization')
    else:
        plt.title('Performance Curve without Regularization')
    plt.xlabel('epoch')
    plt.ylabel('Performance')
    plt.legend(['train', 'validation', 'y = 3x', 'y = 4x'], loc='lower right')
    if r_flag is not None:
        plt.savefig("part4_" + r_flag + ".png")
    else:
        plt.savefig("part4.png")
    # plt.show()

def part4(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
          train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f, r_flag=None):
              
    train_xs, train_ys, validation_xs, validation_ys, test_xs, test_ys = \
        vectorize_data_for_regression(train_xs_r, test_xs_r, validation_xs_r, \
                                      train_ys_r, test_ys_r, validation_ys_r, \
                                      train_xs_f, test_xs_f, validation_xs_f, \
                                      train_ys_f, test_ys_f, validation_ys_f)

    epochs, perfs_train, perfs_validation, trained_model = \
        perform_logistic_regression(train_xs, train_ys, validation_xs, validation_ys, r_flag)
    state_dict = trained_model.state_dict()
    trained_weights = state_dict['linear.weight']
    np.save("trained_weights.npy", trained_weights)
    
    # sets for computing performance
    x_train = Variable(torch.from_numpy(train_xs), requires_grad=False).type(torch.FloatTensor)
    y_train = Variable(torch.from_numpy(np.argmax(train_ys, 1)), requires_grad=False).type(torch.LongTensor)
    x_val = Variable(torch.from_numpy(validation_xs), requires_grad=False).type(torch.FloatTensor)
    y_val = Variable(torch.from_numpy(np.argmax(validation_ys, 1)), requires_grad=False).type(torch.LongTensor)
    x_test = Variable(torch.from_numpy(test_xs), requires_grad=False).type(torch.FloatTensor)
    y_test = Variable(torch.from_numpy(np.argmax(test_ys, 1)), requires_grad=False).type(torch.LongTensor)

    y_pred_train = trained_model(x_train).data.numpy()
    perf_train = np.mean(np.argmax(y_pred_train, 1) == np.argmax(y_train, 1))
    y_pred_val = trained_model(x_val).data.numpy()
    perf_val = np.mean(np.argmax(y_pred_val, 1) == np.argmax(y_val, 1))
    y_pred_test = trained_model(x_test).data.numpy()
    perf_test = np.mean(np.argmax(y_pred_test, 1) == np.argmax(y_test, 1))

    print "Final training, validation, and test performance for Logistic Regression: "
    print 'Training set accuracy: {0:.2f}%'.format(perf_train * 100)
    print 'Validation set accuracy: {0:.2f}%'.format(perf_val * 100)
    print 'Test set accuracy: {0:.2f}%\n'.format(perf_test * 100)

    part4_graph(epochs, perfs_train, perfs_validation, r_flag)
    plt.clf()
    
    return trained_model

def part6(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
          train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f):

    trained_weights = np.load("trained_weights.npy")
    train_xs = np.concatenate((train_xs_r, train_xs_f))
    all_words = np.unique(np.hstack(train_xs))
    
    # 1. Including stop-words --------------------------------------------------
    print "(a) Including stop-words --------------------------------------------------"
    # Weights for c = real
    trained_weights_r = trained_weights[0] #this weight is for c = real where y = 1 and [1, 0] in vectorized form
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_r, all_words)
    print "The top 10 positive weights and their corresponding words for c = real: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = real: \n", min_ten_words
    
    # Weights for c = fake
    trained_weights_f = trained_weights[1]
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_f, all_words)
    print "The top 10 positive weights and their corresponding words for c = fake: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = fake: \n", min_ten_words
    print "\n"

    # 2. Not-including stop-words ----------------------------------------------
    print "(b) Not including stop-words ----------------------------------------------"
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_r, all_words, False)
    print "The top 10 positive weights and their corresponding words for c = real: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = real: \n", min_ten_words
    
    # Weights for c = fake
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_f, all_words, False)
    print "The top 10 positive weights and their corresponding words for c = fake: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = fake: \n", min_ten_words

def part7_graph(depths, train_accs, val_accs):

    # x_labels = [str(d) for d in depths]

    plt.yticks(np.arange(65, 110, step=5.0))
    # plt.xticks(depths, x_labels)
    plt.plot(depths, train_accs, 'r', label="Training Set")
    plt.plot(depths, val_accs, 'b', label="Validation Set")
    plt.xlabel("Depth of tree")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on training and validation sets")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(os.getcwd() + '/part7_graph.png')

def part7(train_x, train_y, validation_x, validation_y, test_x, test_y, train_words):

    max_depth = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    words_list = get_keywords_list(train_words)

    n_features = train_x.shape[1]
    n_samples = train_x.shape[0]
    depths = []
    train_accs = []
    val_accs = []

    for depth in max_depth:

        print "Depth: ", str(depth)
        # use random seed 0
        clf_tree = DecisionTreeClassifier(max_depth=depth, random_state=0, min_samples_split=8, max_features=0.25)
        clf_tree.fit(train_x, train_y)

        # calculate accuracy on training and validation sets
        train_acc = clf_tree.score(train_x, train_y) * 100
        val_acc = clf_tree.score(validation_x, validation_y) * 100
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        depths.append(depth)

        print "Training accuracy: ", train_acc
        print "Validation accuracy: ", val_acc
        print "\n"

    # plot graph
    part7_graph(depths, train_accs, val_accs)
    plt.clf()

    # best validation performance at max_depth=64
    clf = DecisionTreeClassifier(max_depth=64, random_state=0, min_samples_split=8, max_features=0.25)
    clf.fit(train_x, train_y)

    # produce visualization
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=words_list, class_names=['real', 'fake'],
                                    filled=True, rounded=True, special_characters=True, max_depth=2)
    graph = graphviz.Source(dot_data)
    graph.render('tree')

    # measure performance on test set
    train_acc = clf.score(train_x, train_y) * 100
    val_acc = clf.score(validation_x, validation_y) * 100
    test_acc = clf.score(test_x, test_y) * 100

    print "Final training, validation, and test performance for Decision Tree: "
    print "Training accuracy: {0:.2f}%".format(train_acc)
    print "Validation accuracy: {0:.2f}%".format(val_acc)
    print "Test accuracy: {0:.2f}%\n".format(test_acc)

def part8(train_x_r, train_x_f, x_i, m, p):

    words_count = get_words_counts(train_x_r, train_x_f)
    num_real = len(train_x_r)
    num_fake = len(train_x_f)
    num_hl = num_real + num_fake

    # values in the left and right nodes after the split
    right_real = words_count[x_i][0]
    right_fake = words_count[x_i][1]
    left_real = num_real - right_real
    left_fake = num_fake - right_fake
    right_vals = [right_real, right_fake]
    left_vals = [left_real, left_fake]

    # class probabilities
    P_real = num_real / float(num_hl)
    P_fake = 1 - P_real
    # probabilities of x_i in the split
    P_x_real = get_prob_words_given_label(words_count, 0, num_real, m, p)[x_i]     # P(x_i|real)
    P_x_fake = get_prob_words_given_label(words_count, 1, num_fake, m, p)[x_i]     # P(x_i|fake)

    # calculate entropies
    h_y = -(P_real * math.log(P_real, 2)) - (P_fake * math.log(P_fake, 2))
    # entropy of Y given that x_i is classified as 'real' in the split
    h_y_x_real = -((float(left_vals[0]) / sum(left_vals)) * math.log((float(left_vals[0]) / sum(left_vals)), 2)) - \
                 ((float(left_vals[1]) / sum(left_vals)) * math.log((float(left_vals[1]) / sum(left_vals)), 2))
    # entropy of Y given that x_i is classified as 'fake' in the split
    h_y_x_fake = -((float(right_vals[0]) / sum(right_vals)) * math.log((float(right_vals[0]) / sum(right_vals)), 2)) - (
            (float(right_vals[1]) / sum(right_vals)) * math.log((float(right_vals[1]) / sum(right_vals)), 2))

    # mutual information after the split
    I_y_xi = h_y - ((P_x_real * h_y_x_real) + (P_x_fake * h_y_x_fake))

    print "Mutual information for word '{}' after the split: ".format(x_i), I_y_xi


if __name__ == '__main__':
    train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r = load_data(fn_real, 0)
    train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f = load_data(fn_fake, 1)

    train_xs = np.concatenate((train_xs_r, train_xs_f))
    train_ys = np.concatenate((train_ys_r, train_ys_f))
    val_xs = np.concatenate((validation_xs_r, validation_xs_f))
    val_ys = np.concatenate((validation_ys_r, validation_ys_f))
    test_xs = np.concatenate((test_xs_r, test_xs_f))
    test_ys = np.concatenate((test_ys_r, test_ys_f))

    train_x, train_y = create_hl_vector(train_xs, train_ys, train_xs)
    val_x, val_y = create_hl_vector(val_xs, val_ys, train_xs)
    test_x, test_y = create_hl_vector(test_xs, test_ys, train_xs)

    #### load dataset again for part4 and 6 ####
    train_xs_r_4, test_xs_r_4, validation_xs_r_4, train_ys_r_4, test_ys_r_4, validation_ys_r_4 = load_data(fn_real, 0, limit=1500)
    train_xs_f_4, test_xs_f_4, validation_xs_f_4, train_ys_f_4, test_ys_f_4, validation_ys_f_4 = load_data(fn_fake, 1, limit=1500)

    print "=============== Starting part1 ==============="
    part1(train_xs_r, train_xs_f)
    print "\n=============== Starting part2 ==============="
    part2(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, \
          validation_ys_f, test_xs_r, test_xs_f, test_ys_r, test_ys_f)
    print "\n=============== Starting part3 ==============="
    part3(train_xs_r, train_xs_f, train_ys_r, train_ys_f)
    print "\n=============== Starting part4 ==============="
    print "Performing Logistic Regression without regularization"
    part4(train_xs_r_4, test_xs_r_4, validation_xs_r_4, train_ys_r_4, test_ys_r_4, validation_ys_r_4, \
          train_xs_f_4, test_xs_f_4, validation_xs_f_4, train_ys_f_4, test_ys_f_4, validation_ys_f_4)
    print "Performing Logistic Regression with regularization"
    part4(train_xs_r_4, test_xs_r_4, validation_xs_r_4, train_ys_r_4, test_ys_r_4, validation_ys_r_4, \
          train_xs_f_4, test_xs_f_4, validation_xs_f_4, train_ys_f_4, test_ys_f_4, validation_ys_f_4, r_flag='r')
    print "\n=============== Starting part6 ==============="
    part6(train_xs_r_4, test_xs_r_4, validation_xs_r_4, train_ys_r_4, test_ys_r_4, validation_ys_r_4, \
          train_xs_f_4, test_xs_f_4, validation_xs_f_4, train_ys_f_4, test_ys_f_4, validation_ys_f_4)
    print "\n=============== Starting part7 ==============="
    part7(train_x, train_y, val_x, val_y, test_x, test_y, train_xs)
    print "\n=============== Starting part8 ==============="
    x_i = "is"
    x_j = "a"
    part8(train_xs_r, train_xs_f, x_i, 1, 0.1)      # part 8a
    part8(train_xs_r, train_xs_f, x_j, 1, 0.1)      # part 8b
