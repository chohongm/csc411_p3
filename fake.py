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



if __name__ == '__main__':

    ##### PART 1 ######
    part1()



