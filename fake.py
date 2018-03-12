from pylab import *
import numpy as np
import matplotlib.pyplot as plt

def get_words_count(data):

    words_count = {}

    for line in open(data):
        words = line.split()
        for word in words:
            if word not in words_count:
                count = 1
                words_count[word] = count
            else:
                words_count[word] += 1

    return words_count


def split_data(dataset1, dataset2):

    training, validation, test = [], [], []

    # number of headlines in each dataset
    count1 = len(open(dataset1).readlines())
    count2 = len(open(dataset2).readlines())

    training_count1 = int(count1 * 0.7)
    training_count2 = int(count2 * 0.7)
    val_count1 = int(count1 * 0.15)
    val_count2 = int(count2 * 0.15)

    # add first dataset to list
    cur_count1 = 0
    for line in open(dataset1):
        words = line.split()
        if cur_count1 < training_count1:
            training.append(words)
        elif cur_count1 < (training_count1 + val_count1):
            validation.append(words)
        elif cur_count1 < count1:
            test.append(words)
        cur_count1 += 1

    # add second dataset to list
    cur_count2 = 0
    for line in open(dataset2):
        words = line.split()
        if cur_count2 < training_count2:
            training.append(words)
        elif cur_count2 < (training_count2 + val_count2):
            validation.append(words)
        elif cur_count2 < count2:
            test.append(words)
        cur_count2 += 1

    return training, validation, test


def part1():

    # get words count from each real and fake dataset
    real_words = get_words_count("clean_real.txt")
    fake_words = get_words_count("clean_fake.txt")

    # # get the most common words in each dataset
    # real_common = []
    # fake_common = []
    #
    # for i in range(3):
    #     max_word_real = max(real_words, key=real_words.get)
    #     max_val_real = real_words.pop(max_word_real)
    #     real_common.append(tuple((max_word_real, max_val_real)))
    #     max_word_fake = max(fake_words, key=fake_words.get)
    #     max_val_fake = fake_words.pop(max_word_fake)
    #     fake_common.append(tuple((max_word_fake, max_val_fake)))

    # print real_common
    # print fake_common
    # print "'the' in real headlines: ", real_words['the']
    # print "'donald' in fake headlines: ", fake_words['donald']

    print "Word: 'trump'"
    print "# of appearances in real headlines: ", real_words['trump']
    print "# of appearances in fake headlines: ", fake_words['trump']
    print "Word: 'donald'"
    print "# of appearances in real headlines: ", real_words['donald']
    print "# of appearances in fake headlines: ", fake_words['donald']
    print "Word: 'the'"
    print "# of appearances in real headlines: ", real_words['the']
    print "# of appearances in fake headlines: ", fake_words['the']

    # split dataset
    training, validation, test = split_data("clean_real.txt", "clean_fake.txt")

    return training, validation, test

if __name__ == '__main__':

    ##### PART 1 ######
    training, validation, test = part1()

    # print training
    # print validation
    # print test




