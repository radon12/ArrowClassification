#imports
from read_csv import makeFeatures
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
#loading data
features_train,features_test,labels_train,labels_test=makeFeatures()

#PCA
t0=time()
pca=PCA(n_components=2)
pca.fit(features_train)
print "done fitting of PCA in %0.3fs" %(time()-t0)

print pca.explained_variance_ratio_

t0=time()
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)
print "done transforming features using PCA in %0.3fs" %(time()-t0)

#making classifier using grid search
print "Fitting the classifier to the training set"
t0=time()
param_grid={}
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }                                                           #parameters for tuning the classifier
clf=GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)  #rbf kernel->wiggly boundaries,class_weight->balanced=>class_weight=1/freq. of a class in data
clf.fit(features_train_pca,labels_train)
print "done fitting of classifier in %0.3fs" %(time()-t0)
print "best estimator found by grid search"
print clf.best_estimator_


#making predictions
print "making predictons on testing set"
t0 = time()
pred = clf.predict(features_test_pca)
print "done predicting in %0.3fs" % (time() - t0)

print clf.score(features_test_pca,labels_test)
#validation using accuracy and evaluation metrics

print classification_report(labels_test,pred,labels=[1,2,3],target_names=["left","straight","right"])
print confusion_matrix(labels_test, pred,labels=[1,2,3])

import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)

with open('my_dumped_pca.pkl', 'wb') as fid:
    cPickle.dump(pca, fid)      
#print clf.predict(list)
#visualizing the classifier
print features_train_pca
X=features_train_pca
x=features_train_pca
y=labels_train
h=0.2
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("classifier")

plt.show()
