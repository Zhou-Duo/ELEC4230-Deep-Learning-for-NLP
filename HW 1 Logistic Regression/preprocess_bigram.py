import numpy as np
import io
import re  
import string as s
import nltk  
import heapq 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  
from nltk.tokenize import TweetTokenizer
from sklearn import preprocessing
import contractions


def n_gram(tokens, n=1):
    if n == 1:
        return tokens
    else:
        results = list()
        for i in range(len(tokens)-n+1):
            # tokens[i:i+n] will return a sublist from i th to i+n th (i+n th is not included)
            results.append(" ".join(tokens[i:i+n]))
        return results
        
def read_data(_file, cleaning):
    revs = []
    bigram_list = []
    with io.open(_file, "r",  encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label == 'pos' else 0 # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())

            if cleaning:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            #revs is a list of dictionaries
            revs.append({ 'y': label_idx, 'txt': orig_rev})

            bigram_list += n_gram(orig_rev.split(),2)
    return revs, bigram_list

def clean_str(string):
    """
    TODO: Data cleaning
    """
    
    string = string.lower()
    string = string.translate(str.maketrans('', '', s.punctuation))
    string = re.sub("([^\x00-\x7F])+","",string)
    string = re.sub("url","",string)
    # expand words
    expanded_words = []     
    for word in string.split(): 
        # using contractions.fix to expand the shotened words 
        expanded_words.append(contractions.fix(word))    
    string = ' '.join(expanded_words)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
    tweet_tokens = tokenizer.tokenize(string)
    
    stemmer = PorterStemmer() 
    tweets_stem = [] 
    for word in tweet_tokens:
        stem_word = stemmer.stem(word)  # stemming word
        tweets_stem.append(stem_word)  # append to the list
    string = (" ").join(tweets_stem)
    return string

def build_vocab(words_list, max_vocab_size=-1):
    """
    TODO:
        Build a word dictionary, use max_vocab_size to limit the total number of vocabulary.
        if max_vocab_size==-1, then use all possible words as vocabulary. The dictionary should look like:
        ex:
            word2idx = { 'UNK': 0, 'i': 1, 'love': 2, 'nlp': 3, ... }

        top_10_words is a list of the top 10 frequently words appeared
        ex:
            top_10_words = ['a','b','c','d','e','f','g','h','i','j']
    """
    word2idx = {'UNK': 0} # UNK is for unknown word
    word2freq = {}
    top_10_words = []
    for word in words_list:
        if word in word2freq:
            word2freq[word] += 1
        else:
            word2freq[word] = 1 
    top_10_words = heapq.nlargest(10, word2freq, key=word2freq.get)
    if max_vocab_size > -1:
        word2freq = sorted(word2freq.items(),key=lambda item:item[1],reverse=True)
        word2freq = word2freq[:max_vocab_size]
        word2freq = dict(word2freq)
    n = 1
    for k in word2freq.keys():
        word2idx[k] = n
        n += 1
    return word2idx, top_10_words

def get_info(revs, words_list):
    """
    TODO:
        First check what is revs. Then calculate max len among the sentences and the number of the total words
        in the data.
        nb_sent, max_len, word_count are scalars
    """
    nb_sent, max_len, word_count = 0, 0, 0
    nb_sent = len(revs)
    sent_len = lambda sentence: len(sentence.split())
    max_len_review = max((rev['txt'] for rev in revs), key = sent_len)
    max_len = sent_len(max_len_review) 
    for rev in revs:
        word_count += sent_len(rev['txt'])
    return nb_sent, max_len, word_count

def data_preprocess(_file, cleaning, max_vocab_size):
    revs, words_list = read_data(_file, cleaning)
    nb_sent, max_len, word_count = get_info(revs, words_list)
    word2idx, top_10_words = build_vocab(words_list, max_vocab_size)
    # data analysis
    print("Number of words: ", word_count)
    print("Max sentence length: ", max_len)
    print("Number of sentences: ", nb_sent)
    print("Number of vocabulary: ", len(word2idx))
    print("Top 10 most frequently words", top_10_words)

    return revs, word2idx

def feature_extraction_bow(revs, word2idx):
    """
    TODO:
        Convert sentences into vectors using BoW.
        data is a 2-D array with the size (nb_sentence*nb_vocab)
        label is a 2-D array with the size (nb_sentence*1)
    """
    
    data = np.zeros((len(revs), len(word2idx)))
    label = []
    index = 0
    for rev in revs:
        word_list = n_gram(rev['txt'].strip().split(),2)
        for word in word_list:
            if word in word2idx.keys():
                data[index,word2idx[word]] += 1
        index += 1
    
    for sent_info in revs:
        label.append([sent_info['y']])
    
    return np.array(data), np.array(label)

def normalization(data):
    """
    TODO:
        Normalize each dimension of the data into mean=0 and std=1
    """
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f', '--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    args = vars(parser.parse_args())
    print(args)

    revs, word2idx = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

    data, label = feature_extraction_bow(revs, word2idx)
    data = normalization(data)
