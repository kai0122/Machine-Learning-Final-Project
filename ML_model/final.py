import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from packager import *
import pickle

modeList = ['naive', 'fourier', 'hilbert', 'DQ']
for mode in modeList:
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'	
	print 'Mode: ', mode
	
	fileName = [
		'sub017_baseline_ECG.csv',
		'sub017_film 2_ECG.csv',
		'sub017_film 1_ECG.csv',
	
		'sub018_baseline_ECG.csv',
		'sub018_film 2_ECG.csv',
		'sub018_film 1_ECG.csv',
		
		'sub022_baseline_ECG.csv',
		'sub022_film 2_ECG.csv',
		'sub022_film 1_ECG.csv',
		
		'sub004_baseline_ECG.csv',
		'sub004_film 2_ECG.csv',
		'sub004_film 1_ECG.csv'
	]
	
	target = [0,1,2,0,2,1,0,2,1,0,2,1]
	x,y = matrify(fileName, target, mode)
	#x,y = mat[:,:-1],np.concatenate(mat[:,-1:]+0.5).astype(int)
	
	# ***************************************
	# *		K-Fold			*
	# ***************************************
	
	knum = 10
	kf = StratifiedKFold(n_splits=knum, shuffle = True)
	for train, test in kf.split(x, y):
		x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
	
	# ****************
	
	
	# ***************************************
	# *		Decision Tree		*
	# ***************************************
	print 'DT Train...'
	clfDT = RandomForestClassifier(n_estimators=5,max_depth=3)
	clfDT = clfDT.fit(x_train,y_train)

	print confusion_matrix(y_test,clfDT.predict(x_test))
	print accuracy_score(clfDT.predict(x_test), y_test)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	
	# ***************************************
	# *		SVM			*
	# ***************************************
	print 'SVM Train...'
	clfSVM = svm.SVC()
	clfSVM = clfSVM.fit(x_train, y_train)
	
	print confusion_matrix(y_test, clfSVM.predict(x_test))
	print accuracy_score(clfSVM.predict(x_test), y_test)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	# ***************************************
	# *		Prob			*
	# ***************************************
	print 'Prob Train...'
	clfProb = GaussianNB()
	clfProb = clfProb.fit(x_train, y_train)
	
	print confusion_matrix(y_test, clfProb.predict(x_test))
	print accuracy_score(clfProb.predict(x_test), y_test)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	# ***************************************
	# *		NN			*
	# ***************************************
	print 'NN Train...'
	clfNN = MLPClassifier(solver = 'lbfgs', activation = 'tanh')
	#print len(x_train),' ',len(y_train)
	clfNN = clfNN.fit(x_train, y_train)
	
	print confusion_matrix(y_test, clfNN.predict(x_test))
	print accuracy_score(clfNN.predict(x_test), y_test)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	
	# ***************************************
	# *		KNN			*
	# ***************************************
	print 'KNN Train...'
	clfKNN = KNeighborsClassifier(n_neighbors=10, metric = 'manhattan', weights='distance')
	clfKNN = clfKNN.fit(x_train, y_train)
	
	print confusion_matrix(y_test, clfKNN.predict(x_test))
	print accuracy_score(clfKNN.predict(x_test), y_test)
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	
	print '*******************************************************'
	print '*			Predict				*'
	print '*******************************************************'
	
	# ****************
	
	newTest,someDummy = matrify(['sub019_baseline_ECG.csv'],[0],mode)
	
	# ****************
	clfList = [clfDT, clfSVM, clfProb, clfNN, clfKNN]
	clfName = ['DT','SVM','Prob','NN','KNN']
	
	for clfIdx,clfModel in enumerate(clfList):
		with open(clfName[clfIdx]+mode+'_ECGmixup'+'.model','wb') as f:
			pickle.dump(clfModel,f)
		answer = clfModel.predict(newTest)
		answer = np.argmax(np.bincount(answer))
		print clfModel.predict(newTest)
		print clfName[clfIdx],':',
		if answer == 0:
			print 'Normal'
		elif answer == 1:
			print 'Excited'
		else:
			print 'Scary'
	
