import numpy as np
import numpy.linalg as LA
import pylab as pl
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

import scipy.spatial as ss
import scipy.io
from scipy.special import digamma
from math import log
import numpy.random as nr
import random

import scipy.optimize as spo
import scipy.special as sps

from tensorflow.contrib import rnn

import csv

def PRD(p,beta):
	# uses accuracy rather than pred power
	# d(x,xhat) = p(x=xhat|sigma)
	# calculate p(x|sigma)
	pX = np.sum(p,0)
	pS = np.sum(p,1)
	pXgS = np.dot(np.diag(1/pS),p)
	# get distortion matrix
	d = pXgS
	#
	pXhatgS0 = np.random.uniform(size=len(pS))
	pXhatgS = np.vstack([pXhatgS0,1-pXhatgS0]).T
	pXhat = np.dot(pXhatgS.T,pS) #np.sum(np.dot(np.diag(pS),pXhatgS),0)
	# loop
	for t in range(5000): # fix this
		log_pXhatgS = np.meshgrid(np.log(pXhat),np.ones(len(pS)))[0]+beta*d
		pXhatgS = np.exp(log_pXhatgS)
		Zs = np.sum(pXhatgS,1)
		pXhatgS = np.dot(np.diag(1/Zs),pXhatgS)
		pXhat = np.dot(pXhatgS.T,pS)
	#
	R = -np.nansum(pXhat*np.log(pXhat))+np.dot(pS,np.nansum(pXhatgS*np.log(pXhatgS),1)) 
	D = np.dot(pS,np.sum(pXhatgS*pXgS,1))
	return R, D

def simulate_FSM(eM,T):
	# eM is the line of the eM name in the file
	eM = eM[3:-1]
	x = eM.find(']')
	Ss = np.arange(int(eM[x-1]))
	Xs = np.arange(2)
	eM = eM[x+3:-x-3]
	transition_fnctn = {}
	while len(eM)>=7:
		foo = eM[:7]
		transition_fnctn[str(int(foo[1])-1)+foo[3]] = int(foo[5])-1
		# get rid of the one you just did from the eM description
		if len(eM)>8:
			eM = eM[8:]
		else:
			eM = []
	# choose random probabilities
	probsEmission = np.zeros(len(Ss))
	for i in Ss:
		if str(i)+'0' in transition_fnctn:
			if str(i) + '1' in transition_fnctn:
				probsEmission[i] = np.random.uniform()
			else:
				probsEmission[i] = 1 # set to probsEmission = p(0)
		else:
			probsEmission[i] = 0
	# run the thing; simulate it
	s = np.random.randint(len(Ss))
	spasts = [s]
	xs = []
	for t in range(T):
		x = np.random.choice([0,1],p=[probsEmission[s],1-probsEmission[s]])
		xs.append(x)
		s = transition_fnctn[str(s)+str(x)]
		spasts.append(s)
	spasts = spasts[:-1]
	#
	# make Imems and Ipreds for PIB
	# get joint prob dist
	p_XY = np.zeros([len(Ss),2])
	T = np.zeros([len(Ss),len(Ss)])
	for key in transition_fnctn.keys():
		j = int(key[0])
		i = transition_fnctn[key]
		if key[1]=='0':
			T[i,j] += probsEmission[j]
		else:
			T[i,j] += 1-probsEmission[j]
	w, v = LA.eig(T)
	ind = np.abs(w-1)<1e-10
	pss = v[:,ind]
	pss /= np.sum(pss)
	for i in Ss:
		if str(i)+'0' in transition_fnctn:
			p_XY[i,0] = probsEmission[i]*pss[i][0]
		if str(i)+'1' in transition_fnctn:
			p_XY[i,1] = (1-probsEmission[i])*pss[i][0]
	#
	betas = np.linspace(0,1e2,5e3)
	Rs = []
	Accs = []
	for beta in betas:
		foo = PRD(p_XY,beta)
		Rs.append(foo[0])
		Accs.append(foo[1])
	#
	betas = np.hstack([betas,np.inf])
	#
	p0 = 0
	accuracy = 0
	for i in range(len(Ss)):
		accuracy += pss[i]*np.max([probsEmission[i],1-probsEmission[i]])
		if probsEmission[i]>0.5:
			p0 += pss[i]
	p1 = 1-p0
	H = -p0*np.log(p0)-p1*np.log(p1)
	Rs.append(H)
	Accs.append(accuracy)
	#
	Cmu = -np.nansum(pss*np.log(pss))
	hmus = -probsEmission*np.log(probsEmission)-(1-probsEmission)*np.log(1-probsEmission)
	hmus[np.isnan(hmus)] = 0
	hmu = np.dot(hmus,pss)
	return xs, spasts, betas, Rs, Accs, Cmu, hmu, probsEmission # could include sfuts for those that have bidirectional eMs

def GLM(xs,k=5):
	# scikit learn
	# train/test split
	xs_train = xs[:int(0.5*len(xs))]
	xs_val = xs[int(0.5*len(xs)):int(0.7*len(xs))]
	xs_test = xs[int(0.7*len(xs)):]
	# train
	hs_train = np.asarray([xs_train[i:i+k] for i in range(len(xs_train)-k)])
	reg1 = LogisticRegression(penalty='l1').fit(hs_train,xs_train[k:])
	reg2 = LogisticRegression(penalty='l2').fit(hs_train,xs_train[k:])
	# choose between L2 and L1
	hs_val = np.asarray([xs_val[i:i+k] for i in range(len(xs_val)-k)])
	hs_test = np.asarray([xs_test[i:i+k] for i in range(len(xs_test)-k)])
	score1 = reg1.score(hs_val,xs_val[k:])
	score2 = reg2.score(hs_val,xs_val[k:])
	if score1>score2:
		xpreds_test = reg1.predict(hs_test)
	else:
		xpreds_test = reg2.predict(hs_test)
	return xpreds_test, xs_test[k:]

def reservoir(xs,spasts,num_nodes,activation_fnctn='tanh'):
	# train/test split
	xs_train = xs[:int(0.5*len(xs))]
	xs_val = xs[int(0.5*len(xs)):int(0.7*len(xs))]
	xs_test = xs[int(0.7*len(xs)):]
	# run the reservoir; assumed that varW = 1
	W = np.random.randn(num_nodes,num_nodes)
	W = 0.95*W/np.max(np.abs(LA.eigvals(W)))
	v = np.random.randn(num_nodes)
	b = np.random.randn(num_nodes)
	#
	h = np.zeros(num_nodes)
	hs = [h]
	for t in range(len(xs_train)):
		if activation_fnctn=='linear':
			h = np.dot(W,h)+v*xs_train[t]+b
		elif activation_fnctn=='relu':
			h = (np.dot(W,h)+v*xs_train[t]+b)*(np.dot(W,h)+v*xs_train[t]+b>0)
		else:
			h = np.tanh(np.dot(W,h)+v*xs_train[t]+b)
		hs.append(h)
	# get xpreds by linear regression
	reg1 = LogisticRegression(penalty='l1').fit(hs[:-1],xs_train)
	reg2 = LogisticRegression(penalty='l2').fit(hs[:-1],xs_train)
	# choose between L1 and L2
	hs = [hs[-1]]
	for t in range(len(xs_val)):
		if activation_fnctn=='linear':
			h = np.dot(W,h)+v*xs_test[t]+b
		elif activation_fnctn=='relu':
			h = (np.dot(W,h)+v*xs_test[t]+b)*(np.dot(W,h)+v*xs_test[t]+b>0)
		else:
			h = np.tanh(np.dot(W,h)+v*xs_test[t]+b)
		hs.append(h)
	score1 = reg1.score(hs[:-1],xs_val)
	score2 = reg2.score(hs[:-1],xs_val)
	# get xpreds_test
	hs = [hs[-1]]
	for t in range(len(xs_test)):
		if activation_fnctn=='linear':
			h = np.dot(W,h)+v*xs_test[t]+b
		elif activation_fnctn=='relu':
			h = (np.dot(W,h)+v*xs_test[t]+b)*(np.dot(W,h)+v*xs_test[t]+b>0)
		else:
			h = np.tanh(np.dot(W,h)+v*xs_test[t]+b)
		hs.append(h)
	if score1>score2:
		xpreds_test = reg1.predict(hs[:-1])
	else:
		xpreds_test = reg2.predict(hs[:-1])
	return xpreds_test, xs_test

def LSTM(xs,num_nodes,lr=1e-5,num_classes=1,timesteps=28,num_epochs=5000):
	tf.reset_default_graph()
	# train an LSTM to predict in MSE sense
	batch_x = []; batch_y = []
	for i in range(timesteps,len(xs)):
		batch_x.append(xs[i-timesteps:i])
		batch_y.append([xs[i]])
	batch_x = np.asarray(batch_x)
	batch_y = np.asarray(batch_y)
	batch_x = batch_x.reshape((len(batch_x),timesteps,1))
	batch_y = batch_y.reshape((len(batch_y),num_classes))
	#
	batch_x_train = batch_x[:int(0.5*len(batch_x))]
	batch_y_train = batch_y[:int(0.5*len(batch_y))]
	batch_x_val = batch_x[int(0.5*len(batch_x)):int(0.7*len(batch_x))]
	batch_y_val = batch_y[int(0.5*len(batch_y)):int(0.7*len(batch_y))]
	batch_x_test = batch_x[int(0.7*len(batch_x)):]
	batch_y_test = batch_y[int(0.7*len(batch_y)):]
	#
	gammas = [0.1,0.3,1,3,10,100]
	foo_accs = []
	foo_xpreds = []
	for gamma in gammas:
		tf.reset_default_graph()
		#
		X = tf.placeholder("float", [None, timesteps, 1])
		Y = tf.placeholder("float", [None, num_classes])
		#
		weights = {'out': tf.Variable(tf.random_normal([num_nodes, num_classes]))}
		biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
		#
		def RNN(x, weights, biases):
			x = tf.unstack(x, timesteps, 1)
			lstm_cell = tf.nn.rnn_cell.LSTMCell(num_nodes, name='basic_lstm_cell', forget_bias=1.0)
			outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
			return tf.matmul(outputs[-1], weights['out']) + biases['out']
		#
		logit_ = RNN(X, weights, biases)
		prediction = tf.math.exp(logit_)/(1+tf.math.exp(logit_))
		#
		cross_ent = -tf.reduce_mean((1-Y)*tf.math.log(prediction)+Y*tf.math.log(1-prediction))
		acc = tf.reduce_mean((1-Y)*(1-prediction))+tf.reduce_mean(Y*prediction)
		#loss_op1 = cross_ent+gamma*tf.reduce_mean(tf.math.abs(weights['out']))
		loss_op1 = acc+gamma*tf.reduce_mean(tf.math.abs(weights['out']))
		#loss_op2 = cross_ent+gamma*tf.reduce_mean(tf.math.square(weights['out']))
		loss_op2 = acc+gamma*tf.reduce_mean(tf.math.square(weights['out']))
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		train_op1 = optimizer.minimize(loss_op1)
		train_op2 = optimizer.minimize(loss_op2)
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for step in range(1, num_epochs+1):
				sess.run(train_op1, feed_dict={X: batch_x_train, Y: batch_y_train})
			#cross_ent1 = sess.run(cross_ent, feed_dict={X: batch_x_val, Y: batch_y_val})
			acc1 = sess.run(acc, feed_dict={X: batch_x_val, Y: batch_y_val})
			xpreds1 = sess.run(prediction, feed_dict={X: batch_x_test})
		with tf.Session() as sess:
			sess.run(init)
			for step in range(1, num_epochs+1):
				sess.run(train_op2, feed_dict={X: batch_x_train, Y: batch_y_train})
			#cross_ent2 = sess.run(cross_ent, feed_dict={X: batch_x_val, Y: batch_y_val})
			acc2 = sess.run(cross_ent, feed_dict={X: batch_x_val, Y: batch_y_val})
			xpreds2 = sess.run(prediction, feed_dict={X: batch_x_test})
		if acc1>acc2: #cross_ent1>cross_ent2:
			#foo_accs.append(cross_ent1)
			foo_accs.append(acc1)
			xpreds = np.asarray([(x<0.5) for x in xpreds1])
		else:
			#foo_accs.append(cross_ent2)
			foo_accs.append(acc2)
			xpreds = np.asarray([(x<0.5) for x in xpreds2])
		foo_xpreds.append(xpreds)
	# find the maximum cross entropy over all gammas and return those xpreds
	ind = np.argmax(np.asarray(foo_accs))
	return xpreds[ind], batch_y_test

def empRD(xs,xpreds):
	p0 = np.sum(xpreds==np.ones(len(xpreds)))/len(xpreds); p1 = 1-p0
	if p0==0 or p0==1:
		H = 0
	else:
		H = -p0*np.log(p0)-p1*np.log(p1)
	#
	accuracy = np.sum((xs==np.ones(len(xs)))*(xpreds==np.ones(len(xpreds)))+(xs==np.zeros(len(xs)))*(xpreds==np.zeros(len(xpreds))))
	accuracy /= len(xs)
	return H, accuracy

for i in range(1,6):
	count = 1
	#
	f = open('EMLibrary/'+str(i+1)+'.hs')
	print('EMLibrary/'+str(i+1)+'.hs')
	eM = f.readline()
	print(eM)
	# do thing
	while eM!='':
		xs, spasts, betas, Rs, Accs, Cmu, hmu, probsEmission = simulate_FSM(eM,30000)
		print('simulation done')
		#
		# autoregressive data
		ks = np.arange(1,15)
		H_GLM = []
		acc_GLM = []
		for k in ks:
			xpreds_GLM, xs_GLM = GLM(xs,k=k)
			foo = empRD(xs_GLM,xpreds_GLM)
			H_GLM.append(foo[0])
			acc_GLM.append(foo[1])
		print('GLM done')
		# reservoirs
		ns = np.arange(1,61,5)
		H_res = []
		acc_res = []
		for n in ns:
			xpreds_res, xs_res = reservoir(xs,spasts,num_nodes=n,activation_fnctn='tanh')
			foo = empRD(xs_res,xpreds_res)
			H_res.append(foo[0])
			acc_res.append(foo[1])
		print('reservoir done')
		#
		ns = np.arange(1,121,10)
		H_lstm = []
		acc_lstm = []
		for n in ns:
			xpreds_lstm, xs_lstm = LSTM(xs,num_nodes=n,lr=1e-3,timesteps=10,num_epochs=1000)
			xpreds_lstm = np.reshape(xpreds_lstm,[1,len(xpreds_lstm)])[0]
			xs_lstm = np.reshape(xs_lstm,[1,len(xs_lstm)])[0]
			foo = empRD(xs_lstm,xpreds_lstm)
			H_lstm.append(foo[0])
			acc_lstm.append(foo[1])
		print('LSTM done')
		#
		np.savez('EM_'+str(i+1)+'_'+str(count)+'_PRD_v4d.npz',Rs=Rs,betas=betas,Accs=Accs,Cmu=Cmu,hmu=hmu,probsEmission=probsEmission,
			H_lstm=H_lstm,acc_lstm=acc_lstm,
			H_res=H_res,acc_res=acc_res,
			H_GLM=H_GLM,acc_GLM=acc_GLM)
		#
		count += 1
		#
		eM = f.readline()
		print(eM)
	#
	f.close()


# dat = np.load('EM_3_29_PRD.npz')
# Rs = dat['Rs']; Accs = dat['Accs']
# H_lstm = dat['H_lstm']; acc_lstm = dat['acc_lstm']
# H_res = dat['H_res']; acc_res = dat['acc_res']
# H_GLM = dat['H_GLM']; acc_GLM = dat['acc_GLM']
# pl.plot(Rs,Accs)
# pl.plot(H_lstm,acc_lstm,'o',label='LSTM')
# pl.plot(H_res,acc_res,'o',label='Reservoir')
# pl.plot(H_GLM,acc_GLM,'o',label='GLM')
# pl.legend(loc='best')
# pl.xlabel('$Rate$',size=20)
# pl.ylabel('$Accuracy$',size=20)
# #pl.savefig('DoubleEvenProcess2_PRD.pdf',bbox_inches='tight')
# pl.show()