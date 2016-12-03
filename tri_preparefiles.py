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
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

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

def log_proba_filter(ndarray,threshold):
    filteredList=[]
    for row in ndarray:
        filteredList.append(list(np.flatnonzero(row > threshold)))
    return(filteredList)


def calculateFScore(Y_actual,Y_predicted):
    """
    F-score = 2 * (precision*recall / precision + recall)
    precision = # of correctly predicted items / total predictions
    recall = # of correctly predicted items / actual predictions
    """
    fscore=0
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
        if len(predicted)==0:
            precision=0
        else:
            precision=precision/len(predicted)
        allPrecision.append(precision)
        for label in predicted:
            if label in actual:
                recall+=1
        recall=recall/len(actual)
        allRecall.append(recall)
    precision=sum(allPrecision)/len(allPrecision)
    recall=sum(allRecall)/len(allRecall)
    if precision==0 and recall==0:
        fscore=0
    else:
        fscore=2 *((precision*recall)/(precision+recall))
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
    filenames=splitListByPool(filterFiles(".","json"),CPU_COUNT)
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

    vectorizer = TfidfVectorizer(min_df=20,sublinear_tf=True,ngram_range=(1, 3))
    Xvec_train=vectorizer.fit_transform(X_train)
    Xvec_test=vectorizer.transform(X_test)
    Xvec_validation=vectorizer.transform(X_validation)
    print("Vectorizing Model took",(time.time()-start)/60)

    #Models
    from sklearn.naive_bayes import GaussianNB,MultinomialNB
    from sklearn.ensemble import VotingClassifier,ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import LSHForest
    from sklearn.metrics import f1_score

    gnb = MultinomialNB()

    # lr = LogisticRegression(n_jobs=-1)
    # et = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1,max_features='log2')
    # svm = SVC(probability=True)

    start=time.time()
    models=[]
    models=models+fit_estimators((gnb,Xvec_train,Y_train))
    # with Pool(CPU_COUNT) as pl:
    #     models=reduce(lambda x,y: x+y,pl.map(fit_estimators,[(gnb,Xvec_train,Y_train)]))
    #     #,(et,Xvec_train,Y_train)]))
    # print("Model Training",(time.time()-start)/60)

    log_probabilities=[0]

    log_probabilities=predict((models[0],Xvec_validation))

    # start=time.time()
    # with Pool(CPU_COUNT) as pl:
    #     log_probabilities=reduce(lambda x,y: np.add(x,y),pl.map(predict,list(zip(models,[Xvec_validation for i in range(len(models))]))))
    # print("Model predict : Validation",(time.time()-start)/60)

    start=time.time()
    results=[]
    pltFscores=[]
    pltPrecision=[]
    pltRecall=[]
    pltlogProbs=[]
    for i in np.arange(-5,0,0.25):
        Y_pred=log_proba_filter(log_probabilities,i)
        vfsc,vprec,vrec=calculateFScore(Y_validation,Y_pred)
        results.append(((vfsc,vprec,vrec),i))
        pltFscores.append(vfsc)
        pltlogProbs.append(i)
        pltRecall.append(vrec)
        pltPrecision.append(vprec)

    plt.plot(pltlogProbs,pltFscores,'-o')
    plt.xlabel("Log Probability threshold")
    plt.ylabel("F-score")
    plt.title("Validation : Log Probability threshold vs Fscore")
    plt.grid(True)
    plt.savefig("tri_fscore_log_proba.png")

    plt.plot(pltlogProbs,pltPrecision,'-o')
    plt.xlabel("Log Probability threshold")
    plt.ylabel("Precision")
    plt.title("Validation : Log Probability threshold vs Precision")
    plt.grid(True)
    plt.savefig("tri_prec_log_proba.png")

    plt.plot(pltlogProbs,pltRecall,'-o')
    plt.xlabel("Log Probability threshold")
    plt.ylabel("Recall")
    plt.title("Validation : Log Probability threshold vs Recall")
    plt.grid(True)
    plt.savefig("tri_fscore_log_proba.png")

    print("Validation Results - Grid Search")
    results=sorted(results,key=lambda x: x[0][0],reverse=True)

    print("----------------")
    print("Test Results")

    log_probabilities=[0]

    start=time.time()
    log_probabilities=predict((models[0],Xvec_test))
    # with Pool(CPU_COUNT) as pl:
    #     log_probabilities=reduce(lambda x,y: np.add(x,y),pl.map(predict,list(zip(models,[Xvec_test for i in range(len(models))]))))
    # print("Model predict: Test (",(time.time()-start)/60,") s")
    Y_pred=log_proba_filter(log_probabilities,results[0][1])
    print("Test F-Score,Precision,Recall,threshold",calculateFScore(Y_test,Y_pred),results[0][1])
