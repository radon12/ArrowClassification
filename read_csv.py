import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
def makeFeatures():
    features_train=[]
    labels_train=[]
    features_test=[]
    labels_test=[]
    f=open("outleft.txt",'r')
    for line in f:
        list=re.split(r'[,]',line)
        list=map(int,(list[0:-1]))
        features_train=features_train+[list]
        labels_train=labels_train+[1]

    f=open("outstraight.txt",'r')
    for line in f:
        list=re.split(r'[,]',line)
        list=map(int,list[0:-1])
        features_train=features_train+[list]
        labels_train=labels_train+[2]

    f=open("outright.txt",'r')
    for line in f:
        list=re.split(r'[,]',line)
        list=map(int,list[0:-1])
        features_train=features_train+[list]
        labels_train=labels_train+[3]

    features_train,labels_train=shuffle(features_train,labels_train,random_state=0)
    #features_train,features_test,labels_train,labels_test=train_test_split(features_train,labels_train,test_size=0.25,random_state=42)
    return features_train,labels_train
