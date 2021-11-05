from nltk import trigrams
import nltk.tokenize as tk
from helper import remove_punc
from itertools import islice


def getFormattedText(filename):

    lines = []
    with open(filename) as myfile:
        for line in myfile:
            if len(line.split()) == 0:
                continue  # skip empty lines
            else:
                lines.append("__" + line.strip().lower() + "__")  # collect lines

    return lines


def getNgrams(line):

    nGrams = list(trigrams(line))

    return nGrams


def getDict(filename):
    tri_tokens = []
    nGramDict = {}

    formattedTexts = getFormattedText(filename)
    content = ' '.join(formattedTexts)
    tokens = tk.word_tokenize(content)  # tokenize
    tokens = remove_punc(tokens)  # remove punctuations

    for token in tokens:
        tri_tokens.extend(getNgrams(token))
    #     nGramDict.fromkeys(item, tri_tokens.count(item))

    nGramDict = [{''.join(item): tri_tokens.count(item)} for item in sorted(set(tri_tokens))]

    return nGramDict


def topNCommon(filename, N):
    commonN = []

    nGramDict = {}

    for dic in getDict(filename):
        nGramDict.update(dict(dic))

    commonN = {k: v for k, v in sorted(nGramDict.items(), reverse=True, key=lambda item: item[1])}

    return list(islice(commonN, N))


def getAllDicts(fileNamesList):
    langDicts = []

    for fileName in fileNamesList:
        langDicts.append(getDict(fileName))

    return langDicts


def dictUnion(listOfDicts):
    unionNGrams = []

    for dic in listOfDicts:
        unionNGrams.extend(dic)

    return unionNGrams


def getAllNGrams(langFiles):

    allNGrams = []

    for file in langFiles:
        formattedTexts = getFormattedText(file)
        content = ' '.join(formattedTexts)
        tokens = tk.word_tokenize(content)  # tokenize
        tokens = remove_punc(tokens)  # remove punctuations

        for token in tokens:
            allNGrams.extend(getNgrams(token))

    return allNGrams


def compareLang(testFile, langFiles, N):
    langMatch = ''

    testFileTopNCommon = topNCommon(testFile, N)

    currentMatch = 0

    for file in langFiles:
        print("checking... " + file)

        if len(set(topNCommon(file, N)) & set(testFileTopNCommon)) > currentMatch:
            currentMatch = len(set(topNCommon(file, N)) & set(testFileTopNCommon))
            langMatch = file

    return langMatch


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext

    # Test topNCommon()
    path = join('ngrams', 'english.txt')
    print(topNCommon(path, 10))

    # Compile ngrams across all 6 languages and determine a mystery language
    path = 'ngrams'
    fileList = [f for f in listdir(path) if isfile(join(path, f))]
    pathList = [join(path, f) for f in fileList if 'mystery' not in f]  # conditional excludes mystery.txt
    print(getAllNGrams(pathList))  # list of all n-grams spanning all languages

    testFile = join(path, 'mystery.txt')
    print(compareLang(testFile, pathList, 20))  # determine language of mystery file
    print(compareLang(testFile, pathList, 20))#determine language of mystery file
   
