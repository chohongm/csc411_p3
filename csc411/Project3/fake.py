import numpy as np
import math
from random import shuffle
import os
#from pylab import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


from logistic_regression_classifier import LogisticRegression
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


fn_fake, fn_real = 'clean_fake.txt', 'clean_real.txt'


# Loads data into categorical sets for each class where each data is a list of words in a line.
def load_data(fn, class_label):
    # class_label is 0 for non-spam/real, and 1 for spam/fake
    
    train_ratio, test_ratio, validation_ratio = 0.70, 0.15, 0.15
    ratios = [train_ratio, test_ratio, validation_ratio]
    
    lines = [] 
    with open(fn, 'r') as f:
        lines = f.readlines()
        # For part 4 and 6 use only 1500 data size. If entire data set is used, 
        # performance accuracy will be 100% for both with and without 
        # regularization (enough data to classify without help of regularization) 
        # so won't be able to see the difference.
        # lines = lines[:1500]
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
            print small_num
    
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
    print "Word: 'the'")
    print "# of appearances in real headlines: ", words_counts['the'][0]
    print "# of appearances in fake headlines: ", words_counts['the'][1]
    print "Word: 'us'")
    print "# of appearances in real headlines: ", words_counts['us'][0]
    print "# of appearances in fake headlines: ", words_counts['us'][1]


def get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, xs_all):

    P_hl_f = [get_product_of_small_nums(get_prob_of_hl_given_label(P_w_f, hl)) for hl in xs_all]
    P_hl_r = [get_product_of_small_nums(get_prob_of_hl_given_label(P_w_r, hl)) for hl in xs_all]
    P_hl = [((P_f * P_hl_f[i]) + (P_r * P_hl_r[i])) for i in range(len(xs_all))]
    
    # P_f_hl = P_f * P_hl_f
    P_f_hl = np.array([(P_f * P_hl_f[i]) / float(P_hl[i]) for i in range(len(xs_all))])
    P_r_hl = np.array([(P_r * P_hl_r[i]) / float(P_hl[i]) for i in range(len(xs_all))])
    
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

    for i, word in enumerate(words):
        P_f_w, P_r_w = get_naive_bayes_probs(P_r, P_f, P_w_r, P_w_f, np.array([[word]]))
        Ps_f_w[i] = [word, P_f_w]
        Ps_r_w[i] = [word, P_r_w]

    # The prob of fake given not word is sum of probs of fake given each word minus the prob of fake given the word.
    Ps_f_nw, Ps_r_nw = np.empty((num_words, 2), dtype=object), np.empty((num_words, 2), dtype=object)
    Ps_f_w_sum = np.sum(Ps_f_w[:,1])
    Ps_r_w_sum = np.sum(Ps_r_w[:,1])
    for i, word in enumerate(words):
        Ps_f_nw[i] = [word, Ps_f_w_sum - Ps_f_w[i,1]]
        Ps_r_nw[i] = [word, Ps_r_w_sum - Ps_r_w[i,1]]

    pres_f = Ps_f_w[Ps_f_w[:,1].argsort()][:10,0]
    pres_r = Ps_r_w[Ps_r_w[:,1].argsort()][:10,0]
    abs_f = Ps_f_nw[Ps_f_nw[:,1].argsort()][:10,0]
    abs_r = Ps_r_nw[Ps_r_nw[:,1].argsort()][:10,0]
    
    print "10 words whose presence most strongly predicts that the news is real: ", pres_r
    print "10 words whose absence most strongly predicts that the news is real: ", abs_r
    print "10 words whose presence most strongly predicts that the news is fake: ", pres_f
    print "10 words whose absence most strongly predicts that the news is fake: ", abs_f
    

def create_hl_matrix(headlines, labels, train_set):

    # get words in training set
    all_words = np.unique(np.hstack(train_set))
    n = (len(all_words)) # num_features
    m = len(headlines) # num_samples
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
                
        y = [0,1] if labels[i] == 1 else [1,0]
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


def part4(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
          train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f):
              
    train_xs, train_ys, validation_xs, validation_ys, test_xs, test_ys = \
        vectorize_data_for_regression(train_xs_r, test_xs_r, validation_xs_r, \
                                      train_ys_r, test_ys_r, validation_ys_r, \
                                      train_xs_f, test_xs_f, validation_xs_f, \
                                      train_ys_f, test_ys_f, validation_ys_f)
    
    epochs, perfs_train, perfs_validation, trained_model = perform_logistic_regression(train_xs, train_ys, validation_xs, validation_ys)
    state_dict = trained_model.state_dict()
    trained_weights = state_dict['linear.weight']
    np.save("trained_weights.npy", trained_weights)
    
    # test set for computing performance
    x_test = Variable(torch.from_numpy(test_xs), requires_grad=False).type(torch.FloatTensor)
    y_test = Variable(torch.from_numpy(np.argmax(test_ys, 1)), requires_grad=False).type(torch.LongTensor)
    
    # Test set performance
    y_pred_test = trained_model(x_test).data.numpy()
    perf_test = np.mean(np.argmax(y_pred_test, 1) == np.argmax(y_test, 1))
    print 'Test set performance: ', perf_test
    
    x_axis = epochs
    plt.plot(x_axis, perfs_train)
    plt.plot(x_axis, perfs_validation)
    plt.title('Performance Curve without regulatization')
    plt.xlabel('epoch')
    plt.ylabel('Performance')
    plt.legend(['train', 'validation', 'y = 3x', 'y = 4x'], loc='lower right')
    plt.savefig("part4-2.jpg")
    plt.show()
    
    return trained_model
    """
    # Hyper Parameters
    input_size = len(X_train.T)
    num_classes = 2
    num_epochs = 500
    batch_size = 64
    learning_rate = 0.01

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
        train_idx = np.random.permutation(range(X_train.shape[0]))

        for i in range(0, X_train.shape[0], batch_size):
            idx = train_idx[i:i + batch_size]  # get indices of current batch

            # TODO: improve accuracy
            # Get pair of (X, y) of the current minibatch/chunk
            x_mb = Variable(torch.from_numpy(X_train[idx]), requires_grad=False).float()
            y_mb = Variable(torch.from_numpy(Y_train[idx]), requires_grad=False).float()

            y_pred = model(x_mb)
            loss = loss_fn(y_pred, y_mb) + (reg_lambda * l2_reg)

            if i % 50 == 0:
                print "Loss: ", loss

            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward(retain_graph=True)  # Compute the gradient
            optimizer.step()  # Use the gradient information to make a step


        # predict accuracy
        train_x = Variable(torch.from_numpy(X_train), requires_grad=False).float()
        train_y_pred = model(train_x).data.numpy()
        print train_y_pred
        print Y_train
        acc_train = get_accuracy(Y_train, train_y_pred)
        train_accs.append(acc_train)

        validation_x = Variable(torch.from_numpy(X_validation), requires_grad=False).float()
        validation_y_pred = model(validation_x).data.numpy()
        acc_validation = get_accuracy(Y_validation, validation_y_pred)
        val_accs.append(acc_validation)

        test_x = Variable(torch.from_numpy(X_test), requires_grad=False).float()
        test_y_pred = model(test_x).data.numpy()
        acc_test = get_accuracy(Y_test, test_y_pred)
        test_accs.append(acc_test)

        print "Epoch: ", epoch
        print "Training accuracy: {}%".format(acc_train)
        print "Validation accuracy: {}%".format(acc_validation)
        print "Test accuracy: {}%".format(acc_test)

    # plot graph
    part4_graph(train_accs, val_accs, test_accs, epochs)
    plt.clf()
    """


def perform_logistic_regression(train_xs, train_ys, validation_xs, validation_ys):

    train_xs, train_ys, validation_xs, validation_ys, test_xs, test_ys = \
        vectorize_data_for_regression(train_xs_r, test_xs_r, validation_xs_r, \
                                      train_ys_r, test_ys_r, validation_ys_r, \
                                      train_xs_f, test_xs_f, validation_xs_f, \
                                      train_ys_f, test_ys_f, validation_ys_f)
        
    # Hyper Parameters 
    m, n = train_xs.shape
    input_size = n
    num_classes = 2
    num_epochs = 75
    num_batches = 18
    learning_rate = 0.001
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
        
    # validation set for computing performance
    x_validation = Variable(torch.from_numpy(validation_xs), requires_grad=False).type(dtype_float)
    y_validation = Variable(torch.from_numpy(np.argmax(validation_ys, 1)), requires_grad=False).type(dtype_long)
    
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
            
            # Comment this part out for 'train without regularization'
            params, gradParams = model.parameters()
            for theta in params:
                loss += (reg_param / (2*m)) * torch.sum(torch.pow(theta, 2))
            
            loss.backward()
            optimizer.step()
            
        if epoch % 5 == 0:
            print 'Epoch: [%d/%d], Batch: [%d/%d], Loss: %.4f' 
                % (epoch+1, num_epochs, i+1, num_batches, loss.data[0])
            epochs.append(epoch)
            
            # Train set performance
            y_pred_train = model(x_train).data.numpy()
            perf_train = np.mean(np.argmax(y_pred_train, 1) == np.argmax(y_train, 1))
            perfs_train.append(perf_train)
            print 'Training set performance: ', perf_train
            
            # Validation set performance
            y_pred_validation = model(x_validation).data.numpy()
            perf_validation = np.mean(np.argmax(y_pred_validation, 1) == np.argmax(y_validation, 1))
            perfs_validation.append(perf_validation)
            print 'Validation set performance: ', perf_validation
    
    return epochs, perfs_train, perfs_validation, model
            

def get_minibatch(x, y, num_batches=10, num_classes=2):
    
    x_batches = []
    y_batches = []
    total = x.shape[0]
    class_size = int(total/num_classes)
    batch_size_per_class =  int(class_size/num_batches)
    
    for i in range(num_batches):
        x_batch, y_batch = [], []
    
        for j in range(num_classes):
            from_i = j*class_size + i*batch_size_per_class
            to_i = j*class_size + (i+1)*batch_size_per_class
            x_batch.append(x[from_i:to_i, :])
            y_batch.append(y[from_i:to_i, :])
    
        x_batches.append(np.vstack(x_batch))
        y_batches.append(np.vstack(y_batch))
    
    return x_batches, y_batches

def get_top_ten_min_max_words(trained_weights, all_words, include_stopwords = True):
    
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
   

def part6(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
          train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f):

    trained_weights = np.load("trained_weights.npy")
    train_xs = np.concatenate((train_xs_r, train_xs_f))
    all_words = np.unique(np.hstack(train_xs))
    
    # 1. Including stop-words --------------------------------------------------
    print "1. Including stop-words --------------------------------------------------"
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
    
    # 2. Not-including stop-words ----------------------------------------------
    print "2. Not-including stop-words ----------------------------------------------")
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_r, all_words, False)
    print "The top 10 positive weights and their corresponding words for c = real: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = real: \n", min_ten_words
    
    # Weights for c = fake
    max_ten_words, min_ten_words = get_top_ten_min_max_words(trained_weights_f, all_words, False)
    print "The top 10 positive weights and their corresponding words for c = fake: \n", max_ten_words
    print "The top 10 negative weights and their corresponding words for c = fake: \n", min_ten_words
    


if __name__ == '__main__':
    train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r = load_data(fn_real, 0)
    train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f = load_data(fn_fake, 1)

    # train_xs = np.concatenate((train_xs_r, train_xs_f))
    # train_ys = np.concatenate((train_ys_r, train_ys_f))
    # print train_ys

    # print len(train_x_vector)
    # print len(train_y_vector)

    # print train_matrix

    # part1(train_xs_r, train_xs_f)
    # part2(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, \
    #      validation_ys_f, test_xs_r, test_xs_f, test_ys_r, test_ys_f)
    # part3a(train_xs_r, train_xs_f, train_ys_r, train_ys_f, validation_xs_r, validation_xs_f, validation_ys_r, validation_ys_f)
    
    # part4(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
    #       train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f)
    
    # part6(train_xs_r, test_xs_r, validation_xs_r, train_ys_r, test_ys_r, validation_ys_r, \
    #       train_xs_f, test_xs_f, validation_xs_f, train_ys_f, test_ys_f, validation_ys_f)

    


