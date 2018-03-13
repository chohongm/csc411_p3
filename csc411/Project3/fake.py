#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt


fn_fake, fn_real = 'clean_fake.txt', 'clean_real.txt'


# Loads data into 3 categories for each class where each data is a list of words in a line.
def load_data(fn):
    train_ratio, test_ratio, validation_ratio = 0.70, 0.15, 0.15
    ratios = [train_ratio, test_ratio, validation_ratio]
    
    lines = [] 
    with open(fn, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        f.close()
    
    num_lines = len(lines)
    sizes = [int(num_lines * ratio) for ratio in ratios]
    
    train_set = lines[:sizes[0]]
    test_set = lines[sizes[0]:sizes[1]]
    validation_set = lines[:sizes[1]:sizes[2]]
    
    return train_set, test_set, validation_set


def get_words_count(data_set):    
    words_count = {}

    for words in data_set:
        for word in words:
            if word not in words_count:
                count = 1
                words_count[word] = count
            else:
                words_count[word] += 1

    return words_count
    

def part1(train_real, train_fake):

    # get words count from each real and fake dataset
    real_words = get_words_count(train_real)
    fake_words = get_words_count(train_fake)

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
    train_fake, test_fake, validation_fake = load_data(fn_fake)
    train_real, test_real, validation_real = load_data(fn_real)
    
    part1(train_real, train_fake)
    
    
    

    
        