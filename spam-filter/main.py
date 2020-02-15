import re
import glob
from NaiveBayesClassifier import NaiveBayesClassifier
import random
from collections import Counter

path = '/dataset/spam/'

data = []

for fn in glob.glob(path):
    is_spam = "ham" not in fn

    with open(fn, 'r') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = re.sub(r"^Subject: ", "", line).strip()
                data.append((subject, is_spam))


def split_data(data, prob):
    """split data into fractions [prob, 1- prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results
# split the data    into    training    data    and test    data,   and then    we’re   ready   to  build   a
# classifier
random.seed(0)
train_data, test_data = split_data(data, 0.75)

classifier = NaiveBayesClassifier()
classifier.train(train_data)

# triples (subject, actual is_spam, precicated spam prob)
classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

# assume that spam_prob > 0.5
counts = Counter((is_spam, spam_prob > 0.5)
                 for _, is_spam, spam_prob in classified)
# # sort    by  spam_probability    from    smallest    to  largest
classified.sort(key=lambda row: row[2])
# the   highest predicted   spam    probabilities   among   the non-spams
spammiest_hams = filter(lambda row: row[1], classified)[-5:]
# the   lowest  predicted   spam    probabilities   among   the actual  spams
hammiest_spams = filter(lambda row:    row[1], classified)[:5]


words = sorted(classifier.words_probs, key=p_spam_given_word)

spammiest_words =   words[-5:]
hammiest_words  =   words[:5]

