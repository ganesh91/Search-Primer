from os import listdir
from multiprocessing import cpu_count,Pool
from math import floor,ceil
from re import sub as regexsub
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as stpwrds
import json
from functools import reduce
import time
from collections import Counter,defaultdict
from random import sample
from sklearn.feature_extraction.text import TfidfVectorizer

start=time.time()

stopwords =set(stpwrds.words('english'))
stemmer = SnowballStemmer('english')

def listFiles(URI):
    return(listdir(URI))

def filterFiles(URI,extension,limitFiles=None):
    filenames=[]
    for fl in listFiles(URI):
        if fl.split(".")[-1] == extension:
            filenames.append(fl)
    if limitFiles is None:
        return(filenames)
    else:
        return(filenames[:limitFiles])

def getCPUCount(fraction=0.5):
    return(floor(fraction*cpu_count()))

def stripSymbols(text):
    return(regexsub('[^A-Za-z0-9\. ]+', '', text).replace("."," "))

def loadJson(file):
    jsonList=[]
    with open(file,'r',encoding="utf-8") as fl:
        for ln in fl:
            jsonList.append(json.loads(ln))
    return(jsonList)

def stemAndRemoveStopWords(text):
    symbolsRemoved=stripSymbols(text)
    stemmed=" ".join([stemmer.stem(token) for token in symbolsRemoved.split(" ") if token != "" and token not in stopwords])
    return(stemmed)

def buildReadPipleine(filenames):
    fileJobs=[]
    for fl in filenames:
        fileContents=loadJson(fl)
        # Space Optimization, Write to the same memory location.
        for i in range(len(fileContents)):
            if len(fileContents[i]["categories"]) > 0:
                fileJobs.append((fileContents[i]["categories"],stemAndRemoveStopWords(fileContents[i]["text"])))
    return(fileJobs)

def splitListByPool(lists,poolCount):
    i=0;
    listLen=len(lists)
    buckets=ceil(listLen/poolCount)
    splitList=[]
    while i < len(lists):
        splitList.append(lists[i:i+buckets])
        i+=buckets
    return(splitList)

def test_train_split(array,percentage=0.3):
    random_numbers=set(sample(range(0,len(array)),floor(len(array)*percentage)))
    main_array=[]
    sampled_array=[]
    for i in range(len(array)):
        if i in random_numbers:
            sampled_array.append(array[i])
        else:
            main_array.append(array[i])
    return((sampled_array,main_array))

def tupleAdd(x,y):
    return((x[0]+y[0],x[1]+y[1]))

def unpackCategories(list):
    X=[]
    Y=[]
    for item in list:
        for category in item[0]:
            Y.append(category)
            X.append(item[1])
    return((X,Y))

def calculateFScore(Y_actual,Y_predicted):
    """
    F-score = 2 * (precision*recall / precision + recall)
    precision = # of correctly predicted items / total predictions
    recall = # of correctly predicted items / actual predictions
    """
    fscore=[]
    allPrecision=[]
    allRecall=[]
    for i in range(len(Y_actual)):
        actual=Y_actual[i]
        predicted=Y_predicted[i]
        precision=0
        recall=0
        for label in actual:
            if label in predicted:
                precision+=1
        precision=precision/len(predicted)
        allPrecision.append(precision)
        for label in predicted:
            if label in actual:
                recall+=1
        recall=recall/len(recall)
        allRecall.append(recall)
        fscore.append(2 * ((precision*recall)/(precision+recall)))
    return((fscore,precision,recall))

if __name__ == "__main__":

    CPU_COUNT=getCPUCount(0.75)
    filenames=splitListByPool(filterFiles(".","json"),CPU_COUNT)
    combinedFileReads=[]
    with Pool(CPU_COUNT) as pl :
        combinedFileReads=reduce(lambda x,y: x+y,pl.map(buildReadPipleine,filenames))
    print("Stemming Operation took",(time.time()-start)/60)

    start=time.time()
    X=[]
    Y=[]
    trainpool=splitListByPool(combinedFileReads,CPU_COUNT)
    with Pool(CPU_COUNT) as pl:
        X,Y=reduce(lambda x,y: tupleAdd(x,y),pl.map(unpackCategories,trainpool))
    print("Unpacking Categories took",(time.time()-start)/60)
    print("# of Rows",len(X))



    start=time.time()

    vectorizer = TfidfVectorizer(ngram_range=(1, 3),min_df=20,sublinear_tf=True)
    X_test=vectorizer.fit_transform(X)
    print("Vectorizing Model took",(time.time()-start)/60)
    print("Shape",X_test.shape)
