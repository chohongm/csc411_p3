import numpy as np
import math
from random import shuffle
import os
#from pylab import *
import matplotlib.pyplot as plt


fn_fake, fn_real = 'clean_fake.txt', 'clean_real.txt'


# Loads data into categorical sets for each class where each data is a list of words in a line.
def load_data(fn, class_label):
    # class_label is 0 for non-spam/real, and 1 for spam/fake
    
    train_ratio, test_ratio, validation_ratio = 0.70, 0.15, 0.15
    ratios = [train_ratio, test_ratio, validation_ratio]
    
    lines = [] 
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
        for word in hl:
            if word not in words_counts:
                words_counts[word] = [1, 0]
            else:
                words_counts[word][0] += 1

    for hl in data_fake:
        for word in hl:
            if word not in words_counts:
                words_counts[word] = [0, 1]
            else:
                words_counts[word][1] += 1

    return words_counts


def get_prob_words_given_label(words_counts, label, num_labeled_data, m , p):
    # again, label is 0 for real and 1 for fake. This number refers to index
    # of words_counts[word] in which word count for that label is stored.
    P_w_l = {}
    for word, counts in words_counts.items():
        P_w_l[word] = (min(counts[label], num_labeled_data)  + m*p) / (num_labeled_data + m)         
        
    return P_w_l


def get_product_of_small_nums(small_nums):
    prod = 0
    
    for small_num in small_nums:
        try:
            prod += math.log(small_num)
        except ValueError:
            print(small_num)
    
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
    
    
def part1(train_real, train_fake):

    # get words count from each real and fake dataset
    words_counts = get_words_counts(train_real, train_fake)
    words_counts.pop('trump')

    # get the most common words in each dataset
    real_words = {}
    fake_words = {}
    for word in words_counts:
        real_words[word] = words_counts[word][0]
        fake_words[word] = words_counts[word][1]

    real_common = []
    fake_common = []

    for i in range(10):
        max_word_real = max(real_words, key=real_words.get)
        max_val_real = real_words.pop(max_word_real)
        real_common.append(tuple((max_word_real, max_val_real)))
        max_word_fake = max(fake_words, key=fake_words.get)
        max_val_fake = fake_words.pop(max_word_fake)
        fake_common.append(tuple((max_word_fake, max_val_fake)))

    print "10 most common words in real headlines: "
    print real_common
    print "10 most common words in fake headlines: "
    print fake_common

    print "\n"
    print "Word: 'donald'"
    print "# of appearances in real headlines: ", words_counts['donald'][0]
    print "# of appearances in fake headlines: ", words_counts['donald'][1]
    print "Word: 'the'"
    print "# of appearances in real headlines: ", words_counts['the'][0]
    print "# of appearances in fake headlines: ", words_counts['the'][1]
    print "Word: 'us'"
    print "# of appearances in real headlines: ", words_counts['us'][0]
    print "# of appearances in fake headlines: ", words_counts['us'][1]


def get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, xs_all):
    
    # P_f_hl = P_f * P_hl_f
    P_f_hl = np.array([P_f * get_product_of_small_nums(get_prob_of_hl_given_label(P_w_f, hl)) for hl in xs_all])
    P_r_hl = np.array([P_r * get_product_of_small_nums(get_prob_of_hl_given_label(P_w_r, hl)) for hl in xs_all])
    
    return P_f_hl, P_r_hl


def part2_graph(params, accs):

    x = range(len(params))
    y = accs
    labels = ['m = {}\np = {}'.format(param[0], param[1]) for param in params]

    plt.plot(x, y)
    plt.xticks(x, labels, rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)

    plt.xlabel("Values for m and p")
    plt.ylabel("Validation accuracy")
    plt.title("Validation Accuracy with Varying Values of m and p")
    plt.grid(axis='y', linestyle='--')
    # plt.show()
    plt.savefig(os.getcwd() + 'part2_graph.png')


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





def part3a(train_xs_r, train_xs_f, train_ys_r, train_ys_f, \
          test_xs_r, test_xs_f, test_ys_r, test_ys_f):
    words_counts = get_words_counts(train_xs_r, train_xs_f)
    
    num_real_data = len(train_ys_r)
    num_fake_data = len(train_ys_f)
    num_total_data = num_real_data + num_fake_data
    test_xs_all = np.concatenate((test_xs_f, test_xs_r))
    test_ys_all = np.concatenate((test_ys_f, test_ys_r))
    
    P_r = num_real_data / float(num_total_data)
    P_f = 1 - P_r
    
    m = 1
    p = 0.1
    P_w_r = get_prob_words_given_label(words_counts, 0, num_real_data, m, p)
    P_w_f = get_prob_words_given_label(words_counts, 1, num_fake_data, m, p)

    words = words_counts.keys()
    num_words = len(words)
    
    # compute NB probs of f & r given word for each word in the entire data set.
    # the top ten in Ps_f_w represents the ten words whose presence most strongly predicts that the news is fake.
    Ps_f_w, Ps_r_w = np.empty((num_words, 2), dtype=object), np.empty((num_words, 2), dtype=object)
    print "Starting for loop"
    for i, word in enumerate(words):
        print i, word
        P_f_w, P_r_w = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, np.array([[word]]))
        Ps_f_w[i] = [word, P_f_w]
        Ps_r_w[i] = [word, P_r_w]
    print "Done with first for loop"
    # The prob of fake given not word is sum of probs of fake given each word minus the prob of fake given the word.
    Ps_f_nw, Ps_r_nw = np.empty((num_words, 2), dtype=object), np.empty((num_words, 2), dtype=object)
    Ps_f_w_sum = np.sum(Ps_f_w[:,1])
    Ps_r_w_sum = np.sum(Ps_r_w[:,1])
    for i, word in enumerate(words):
        print i, word
        Ps_f_nw[i] = [word, Ps_f_w_sum - Ps_f_w[i,1]]
        Ps_r_nw[i] = [word, Ps_r_w_sum - Ps_r_w[i,1]]
    print "Done with second for loop"
    pres_f = Ps_f_w[Ps_f_w[:,1].argsort()][:10,0]
    pres_r = Ps_r_w[Ps_r_w[:,1].argsort()][:10,0]
    abs_f = Ps_f_nw[Ps_f_nw[:,1].argsort()][:10,0]
    abs_r = Ps_r_nw[Ps_r_nw[:,1].argsort()][:10,0]
    
    print "10 words whose presence most strongly predicts that the news is real: ", pres_r
    print "10 words whose absence most strongly predicts that the news is real: ", abs_r
    print "10 words whose presence most strongly predicts that the news is fake: ", pres_f
    print "10 words whose absence most strongly predicts that the news is fake: ", abs_f
    
    #---------------------------------------------------------------------------
    # continue from here
    #---------------------------------------------------------------------------

def get_keywords_list(dataset):

    words = []

    for hl in dataset:
        # words_list = hl.split()
        for word in hl:
            if word not in words:
                words.append(word)

    return words

def create_hl_matrix(x, y):

    words = get_keywords_list(x)
    num_words = len(words)
    hl_x = []

    for hl in x:
        hl_words = []
        for word in words:
            if word in hl:
                hl_words.append(1)
            else:
                hl_words.append(0)
        hl_x.append(hl_words)

    hl_x = np.vstack(tuple(hl_x))
    hl_y = np.vstack(tuple(y))

    return hl_x, hl_y

if __name__ == '__main__':
    train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r = load_data(fn_real, 0)
    train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f = load_data(fn_fake, 1)

    train_xs = np.concatenate((train_xs_r, train_xs_f))
    train_ys = np.concatenate((train_ys_r, train_ys_f))
    # print train_ys

    train_x_vector, train_y_vector = create_hl_matrix(train_xs, train_ys)

    # print len(train_x_vector)
    # print len(train_y_vector)

    # print train_matrix

    # part1(train_xs_r, train_xs_f)
    # part2(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, \
    #       validation_ys_f, test_xs_r, test_xs_f, test_ys_r, test_ys_f)
    # part3a(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, validation_ys_f)
    
    

    
        