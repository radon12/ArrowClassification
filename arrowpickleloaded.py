import cPickle
from time import time
from read_csv import makeFeatures

t0=time()
with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)
print "time take in loading classifier %0.3fs" %(time()-t0)

t0=time()
with open('my_dumped_pca.pkl', 'rb') as fid:
    pca = cPickle.load(fid)
print "time take in loading pca %0.3fs" %(time()-t0)

t0=time()
features_train,labels_train=makeFeatures()
print "time take in making features %0.3fs" %(time()-t0)

t0=time()
features_train=pca.transform(features_train)
print "time take in transforming features %0.3fs" %(time()-t0)

print clf.score(features_train,labels_train)
