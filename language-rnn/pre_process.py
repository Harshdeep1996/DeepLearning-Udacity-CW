import csv
import nltk
import itertools

vocab_size = 8000
sentence_start = 'SENTENCE_START'
sentence_end = 'SENTENCE_END'
unknown_token = 'UNKNOWN_TOKEN'


def pre_process_data():

    print('Reading data...')

    # skipinitialspace for csv reader, means that the space just after
    # the delimiter is ignored

    with open('data/reddit-comments-2015-08.csv', 'r') as data:
        reader = csv.reader(data, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(
            *[nltk.sent_tokenize(l[0].decode('utf-8').lower()) for l in reader]
        )
        # Put tokens in front of the sentence
        sentences = [
            '%s %s %s' % (sentence_start, s, sentence_end) for s in sentences
        ]
    print('Total number of sentences: {}'.format(len(sentences)))

    tokens = [nltk.word_tokenize(s) for s in sentences]
    # Get frequencies for all the words
    word_freq = nltk.FreqDist(itertools.chain(*tokens))
    vocab = word_freq.most_common(vocab_size - 1)
    # Create mapping between index and words with respect to the frequency
    index_to_word = [w[0] for w in vocab]
    index_to_word.append(unknown_token)

    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    for i, sent in enumerate(tokens):
        tokens[i] = [
            w if w in index_to_word else unknown_token for w in sent
        ]

    train_X = [[word_to_index[w] for w in s[:-1]] for s in tokens]
    train_Y = [[word_to_index[w] for w in s[1:]] for s in tokens]

    return train_X, train_Y
