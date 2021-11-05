from helper import remove_punc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download as dl
from nltk.stem import PorterStemmer
import numpy as np
from collections import OrderedDict

dl('punkt')
dl('stopwords')
#Clean and lemmatize the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words stemmed
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def readAndCleanDoc(doc) :
    f = open(doc, 'r')
    s_str = f.read()
    f.close()
    words = word_tokenize(s_str)
    words = [word for word in words if word.isalpha()]
    words = remove_punc(words)
    words = [word.lower() for word in words]
    words = [word for word in words if not word in stopwords.words('english')]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return words


#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the wordlist output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
def buildDocWordMatrix(doclist) :
    #1. Create word lists for each cleaned doc (use readAndCleanDoc)\
    wordlists = []
    for doc in doclist:
        app = readAndCleanDoc(doc)
        app.sort()
        wordlists.append(app)
    #2. Use these word lists to build the doc word matrix
    dic = {word: 0 for wl in wordlists for word in wl}
    dic = dict(OrderedDict(sorted(dic.items())))
    wordlist = [word for word in dic]
    docword = np.full((len(doclist), len(dic)), fill_value=0)
    for doc_i in range(len(doclist)):
        for word_i, keyword in enumerate(dic):
            docword[doc_i][word_i] = wordlists[doc_i].count(keyword)
    return docword, wordlist


#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def buildTFMatrix(docword) :
    tf = np.full(docword.shape, fill_value=0, dtype='f')
    for doc_i in range(docword.shape[0]):
        for word_i in range(docword.shape[1]):
            tf[doc_i][word_i] = docword[doc_i][word_i] / np.sum(docword[doc_i])
    return tf


#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of words in the doc word matrix)
#Don't forget the log factor!
def buildIDFMatrix(docword) :
    import math
    occ = [0] * len(docword[0])
    for i in range(len(occ)):
        for j in range(len(docword)):
            if docword[j][i] > 0:
                occ[i] += 1
    idf = np.full((1, len(docword[0])), fill_value=0, dtype='f')
    for i in range(len(docword[0])):
        idf[0][i] = math.log10(len(docword) / occ[i])
    return idf


#Builds a tf-idf matrix given a doc word matrix
def buildTFIDFMatrix(docword) :
    tf = buildTFMatrix(docword)
    idf = buildIDFMatrix(docword)
    tfidf = tf * idf
    return tfidf


# * Find the three most distinctive words, according to TFIDF, in each document

# * Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
#   (corresponding to rows)

# * Output: a dictionary, mapping each document name from doclist to an (ordered
#   list of the three most common words in each document
def findDistinctiveWords(docword, wordlist, doclist) :
    tfidf = -buildTFIDFMatrix(docword)
    distinctiveWords = {}
    for i, key in enumerate(doclist):
        distinctiveWords[key] = []
        distinctiveWords[key].append(wordlist[np.argsort(tfidf[i])[0]])
        distinctiveWords[key].append(wordlist[np.argsort(tfidf[i])[1]])
        distinctiveWords[key].append(wordlist[np.argsort(tfidf[i])[2]])
    return distinctiveWords
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '1_vidText.txt')
    path2 = join(directory, '2_vidText.txt')
    # Uncomment and recomment ths part where you see fit for testing purposes

    print("*** Testing readAndCleanDoc ***")
    print(readAndCleanDoc(path1)[0:5])
    print("*** Testing buildDocWordMatrix ***") 
    doclist =[path1, path2]
    docword, wordlist = buildDocWordMatrix(doclist)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    print("*** Testing buildTFMatrix ***") 
    tf = buildTFMatrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis =1))
    print("*** Testing buildIDFMatrix ***") 
    idf = buildIDFMatrix(docword)
    print(idf[0][0:10])
    print("*** Testing buildTFIDFMatrix ***") 
    tfidf = buildTFIDFMatrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])
    print("*** Testing findDistinctiveWords ***")
    print(findDistinctiveWords(docword, wordlist, doclist))
