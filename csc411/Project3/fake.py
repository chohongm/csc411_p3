import numpy as np
import math
from random import shuffle
import os
#from pylab import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import operator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz

fn_fake, fn_real = 'clean_fake.txt', 'clean_real.txt'

# Loads data into categorical sets for each class where each data is a list of words in a line.
def load_data(fn, class_label):
    # class_label is 0 for non-spam/real, and 1 for spam/fake
    
    train_ratio, test_ratio, validation_ratio = 0.70, 0.15, 0.15
    ratios = [train_ratio, test_ratio, validation_ratio]

    with open(fn, 'r') as f:
        lines = f.readlines()
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
    # print train_xs
    
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

    stopwords = []

    for stopword in ENGLISH_STOP_WORDS:
        if stopword in (pres_f or pres_r or abs_f or abs_r):
            stopwords.append(stopword)

    for stopword in stopwords:
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

    print "Final performance on training and test sets with m = 1 and p = 0.1"
    P_w_r = get_prob_words_given_label(words_counts, 0, num_real_data, 1, 0.1)
    P_w_f = get_prob_words_given_label(words_counts, 1, num_fake_data, 1, 0.1)

    P_f_hl_train, P_r_hl_train = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, train_xs_all)
    # Since there are more than one class, P_f_hl = (P_f * P_hl_f) / sum(P_c * P_hl_c for each class c)
    predicted_ys_train = np.round(P_f_hl_train / (P_f_hl_train + P_r_hl_train))
    train_accuracy = np.sum(predicted_ys_train == train_ys_all) / float(len(train_ys_all))

    P_f_hl_test, P_r_hl_test = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, test_xs_all)
    # Since there are more than one class, P_f_hl = (P_f * P_hl_f) / sum(P_c * P_hl_c for each class c)
    predicted_ys_test = np.round(P_f_hl_test / (P_f_hl_test + P_r_hl_test))
    test_accuracy = np.sum(predicted_ys_test == test_ys_all) / float(len(test_ys_all))

    print "Training accuracy: {0:.2f}%".format(train_accuracy * 100)
    print "Test accuracy: {0:.2f}%".format(test_accuracy * 100)

def part3(train_xs_r, train_xs_f, train_ys_r, train_ys_f):
    ##### PART 3A #####
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

    print "10 words whose presence most strongly predicts that the news is real: ", pres_r[:10]
    print "10 words whose absence most strongly predicts that the news is real: ", abs_r[:10]
    print "10 words whose presence most strongly predicts that the news is fake: ", pres_f[:10]
    print "10 words whose absence most strongly predicts that the news is fake: ", abs_f[:10]
    print "\n"

    ##### PART 3B #####
    remove_stopwords(pres_f, pres_r, abs_f, abs_r)
    print "10 non-stopwords whose presence most strongly predicts that the news is real: ", pres_r[:10]
    print "10 non-stopwords whose absence most strongly predicts that the news is real: ", abs_r[:10]
    print "10 non-stopwords whose presence most strongly predicts that the news is fake: ", pres_f[:10]
    print "10 non-stopwords whose absence most strongly predicts that the news is fake: ", abs_f[:10]

def part4_graph(train_accs, val_accs, test_accs, epochs):

    range = list(set(train_accs + val_accs + test_accs))

    plt.yticks(np.arange(min(range), max(range)))
    plt.plot(epochs, train_accs, 'r', label="Training Set")
    plt.plot(epochs, val_accs, 'b', label="Validation Set")
    plt.plot(epochs, test_accs, 'g', label="Test Set")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on all sets")
    plt.grid(axis='y', linestyle='--')
    plt.savefig(os.getcwd() + 'part4_graph.png')

def part4():
    # TODO: finish this function

    # load data
    train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r = load_data(fn_real, 0)
    train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f = load_data(fn_fake, 1)

    train_xs = np.concatenate((train_xs_r, train_xs_f))
    train_ys = np.concatenate((train_ys_r, train_ys_f))
    validation_xs = np.concatenate((validation_xs_r, validation_xs_f))
    validation_ys = np.concatenate((validation_ys_r, validation_ys_f))
    test_xs = np.concatenate((test_xs_r, test_xs_f))
    test_ys = np.concatenate((test_ys_r, test_ys_f))

    x_train, y_train = create_hl_matrix(train_xs, train_ys, train_xs)
    x_validation, y_validation = create_hl_matrix(validation_xs, validation_ys, train_xs)
    x_test, y_test = create_hl_matrix(test_xs, test_ys, train_xs)
    print x_test
    print y_test
    # Hyper Parameters
    input_size = len(x_train.T)
    num_classes = 1
    num_epochs = 200
    batch_size = 64
    learning_rate = 0.1

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, num_classes),
        torch.nn.Sigmoid(),
        torch.nn.Softmax(dim=1),
    )

    # Loss and Optimizer
    # Set parameters to be updated.
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reg_lambda = 0.1

    train_accs = []
    val_accs = []
    test_accs = []
    epochs = []
    # Training the Model
    for epoch in range(num_epochs):

        print "Epoch: ", epoch
        epochs.append(epoch)

        # calculate regularization term
        l2_reg = None
        for W in model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        np.random.seed(0)
        # torch.manual_seed(0)
        train_idx = np.random.permutation(range(x_train.shape[0]))

        for i in range(0, x_train.shape[0], batch_size):
            idx = train_idx[i:i + batch_size]  # get indices of current batch

            # TODO: improve accuracy
            # Get pair of (X, y) of the current minibatch/chunk
            x_mb = Variable(torch.from_numpy(x_train[idx]), requires_grad=False).float()
            y_mb = Variable(torch.from_numpy(y_train[idx]), requires_grad=False).float()

            y_pred = model(x_mb)
            loss = loss_fn(y_pred, y_mb) + (reg_lambda * l2_reg)

            if i % 50 == 0:
                print "Loss: ", loss

            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward(retain_graph=True)  # Compute the gradient
            optimizer.step()  # Use the gradient information to make a step


        # predict accuracy
        train_x = Variable(torch.from_numpy(x_train), requires_grad=False).float()
        train_y_pred = model(train_x).data.numpy()
        print train_y_pred
        print y_train
        acc_train = get_accuracy(y_train, train_y_pred)
        train_accs.append(acc_train)

        validation_x = Variable(torch.from_numpy(x_validation), requires_grad=False).float()
        validation_y_pred = model(validation_x).data.numpy()
        acc_validation = get_accuracy(y_validation, validation_y_pred)
        val_accs.append(acc_validation)

        test_x = Variable(torch.from_numpy(x_test), requires_grad=False).float()
        test_y_pred = model(test_x).data.numpy()
        acc_test = get_accuracy(y_test, test_y_pred)
        test_accs.append(acc_test)

        print "Epoch: ", epoch
        print "Training accuracy: {}%".format(acc_train)
        print "Validation accuracy: {}%".format(acc_validation)
        print "Test accuracy: {}%".format(acc_test)

    # plot graph
    part4_graph(train_accs, val_accs, test_accs, epochs)
    plt.clf()

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

def part7(train_x, train_y, validation_x, validation_y, train_words):

    max_depth = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    words_list = get_keywords_list(train_words)
    print len(words_list)

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

    # calculate necessary probabilities
    P_real = num_real / float(num_hl)
    P_fake = 1 - P_real
    # probabilities of x_i in the split
    P_x_real = get_prob_words_given_label(words_count, 0, num_real, m, p)[x_i]     # P("is"|real)
    P_x_fake = get_prob_words_given_label(words_count, 1, num_fake, m, p)[x_i]     # P("is"|fake)

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

    train_x, train_y = create_hl_vector(train_xs, train_ys, train_xs)
    val_x, val_y = create_hl_vector(val_xs, val_ys, train_xs)

    # part1(train_xs_r, train_xs_f)
    # part2(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, \
    #       validation_ys_f, test_xs_r, test_xs_f, test_ys_r, test_ys_f)
    # part3(train_xs_r, train_xs_f, train_ys_r, train_ys_f)

    # part4()
    # part7(train_x, train_y, val_x, val_y, train_xs)
    # TODO: part 3c, part 7c
    ##### PART 8 #####
    x_i = "is"
    x_j = "the"
    x_k = "brexit"
    part8(train_xs_r, train_xs_f, x_i, 1, 0.1)      # part 8a
    part8(train_xs_r, train_xs_f, x_j, 1, 0.1)      # part 8b
    part8(train_xs_r, train_xs_f, x_k, 1, 0.1)