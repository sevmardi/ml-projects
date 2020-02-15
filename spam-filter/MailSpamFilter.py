import re
from collections import defaultdict
import math


def tokenize(message):
    message = message.lower()  # convert message to lowercase
    all_words = re.findall("[a-z0-9]+", message)  # extract the words
    return set(all_words)  # remove duplicates


def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets w, p(w | spam) and p(w| ~spam)"""
    return [(w, (spam + k) / (total_spams + 2 * k), (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]


def spam_probability(words_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    # iterate through each word in our vocabluary
    for word, prob_if_spam, prob_if_not_spam in words_probs:
        # if *word* appears in the message, add the log probability of seeing
        # it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        # if *word* doens't appear in the message add the log probability of _not_ seeing it
        # which is log(1-probability of seeing it)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def p_spam_given_word(word_prob):
    """Uses byes therom to compute p(spam | message contains word)"""
    word, prob_if_spam, prob_if_not_spam=word_prob
    return prob_if_spam/(prob_if_spam+prob_if_not_spam)

def drop_final_s(word):
    """ really  simple  stemmer function  """
    return re.sub("s$", "", word)
    