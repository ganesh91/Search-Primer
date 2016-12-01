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
from sklearn import preprocessing
import numpy as np

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

def fit_estimators(tupleofestimators):
    start=time.time()
    estimator,trainX,trainY=tupleofestimators
    rt=[estimator.fit(trainX,trainY)]
    print(estimator,(time.time()-start)/60)
    return(rt)

def predict(tupleofestimators):

    estimator,predict=tupleofestimators
    return(estimator.predict_log_proba(predict))

if __name__ == "__main__":

    CPU_COUNT=getCPUCount(0.75)
    filenames=splitListByPool(filterFiles(".","json")[:5],CPU_COUNT)
    combinedFileReads=[]
    with Pool(CPU_COUNT) as pl :
        combinedFileReads=reduce(lambda x,y: x+y,pl.map(buildReadPipleine,filenames))
    print("Stemming Operation took",(time.time()-start)/60)

    start=time.time()
    whole_X=[]
    whole_Y=[]
    vocabulary=set()
    locationDict=defaultdict(list)
    for i in range(len(combinedFileReads)):
        whole_X.append(combinedFileReads[i][1])
        whole_Y.append(combinedFileReads[i][0])
        locationDict["".join(set(combinedFileReads[i][0]))].append(i)
        for item in combinedFileReads[i][0]:
            if item not in vocabulary:
                vocabulary=vocabulary.union([item])

    rlookup = dict(zip(range(len(vocabulary)),list(vocabulary)))
    lookup = dict(zip(list(vocabulary),range(len(vocabulary))))

    #Stratified Test Train Split
    X_test=[]
    X_Y_train=[]
    X_validation=[]
    Y_test=[]
    Y_validation=[]
    for key in locationDict.keys():
        test_index=set(sample(locationDict[key],floor(len(locationDict[key])*0.2)))
        trn_val_index=[i for i in locationDict[key] if i not in test_index]
        validation_index=set(sample(trn_val_index,floor(len(trn_val_index)*0.1)))
        for index in locationDict[key]:
            if index in test_index:
                X_test.append(whole_X[index])
                Y_test.append(list(map(lambda x: lookup[x],whole_Y[index])))
            elif index in validation_index:
                X_validation.append(whole_X[index])
                Y_validation.append(list(map(lambda x: lookup[x],whole_Y[index])))
            else:
                X_Y_train.append([list(map(lambda x: lookup[x],whole_Y[index])),whole_X[index]])

    print("Data Split took",(time.time()-start)/60)
    print("Test,Train,Validation length",(len(X_test),len(X_Y_train),len(Y_test)))


    start=time.time()
    X_train=[]
    Y_train=[]
    trainpool=splitListByPool(X_Y_train,CPU_COUNT)
    with Pool(CPU_COUNT) as pl:
        X_train,Y_train=reduce(lambda x,y: tupleAdd(x,y),pl.map(unpackCategories,trainpool))
    print("Unpacking Categories took",(time.time()-start)/60)
    print("# of Rows",len(X_train))

    start=time.time()

    vectorizer = TfidfVectorizer(min_df=20,sublinear_tf=True)
    Xvec_train=vectorizer.fit_transform(X_train)
    Xvec_test=vectorizer.transform(X_test)
    Xvex_validation=vectorizer.transform(X_validation)
    print("Vectorizing Model took",(time.time()-start)/60)

    #Models
    from sklearn.naive_bayes import GaussianNB,MultinomialNB
    from sklearn.ensemble import VotingClassifier,ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    gnb = MultinomialNB()
    lr = LogisticRegression(n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1,max_features='log2')
    svm = SVC(probability=True)

    start=time.time()
    models=[]
    with Pool(CPU_COUNT) as pl:
        models=reduce(lambda x,y: x+y,pl.map(fit_estimators,[(gnb,Xvec_train,Y_train),(et,Xvec_train,Y_train)]))
    print("Model Training",(time.time()-start)/60)

    log_probabilities=[0]

    start=time.time()
    with Pool(CPU_COUNT) as pl:
        log_probabilities=reduce(lambda x,y: np.add(x,y),pl.map(predict,list(zip(models,[Xvec_train for i in range(len(models))]))))
    print("Model predict",(time.time()-start)/60)

    print(models[0].score(Xvex_validation,Y_validation))
    print(models[1].score(Xvex_validation,Y_validation))
