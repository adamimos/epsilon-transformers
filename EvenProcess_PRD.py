import pylab as pl
import numpy as np

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
	print(probsEmission)
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
	return xs, spasts, betas, Rs, Accs # could include sfuts for those that have bidirectional eMs

def GLM(xs,k=5):
	# scikit learn
	hs = np.asarray([xs[i:i+k] for i in range(len(xs)-k)])
	reg = LogisticRegression().fit(hs,xs[k:])
	xpreds = reg.predict(hs)
	return xpreds, xs[k:]

def reservoir(xs,spasts,num_nodes,activation_fnctn='tanh'):
	# run the reservoir; assumed that varW = 1
	W = np.random.randn(num_nodes,num_nodes)
	W = W/np.max(np.abs(LA.eigvals(W)))
	v = np.random.randn(num_nodes)
	b = np.random.randn(num_nodes)
	#
	h = np.zeros(num_nodes)
	hs = [h]
	for t in range(len(xs)):
		if activation_fnctn=='linear':
			h = np.dot(W,h)+v*xs[t]+b
		elif activation_fnctn=='relu':
			h = (np.dot(W,h)+v*xs[t]+b)*(np.dot(W,h)+v*xs[t]+b>0)
		else:
			h = np.tanh(np.dot(W,h)+v*xs[t]+b)
		hs.append(h)
	hs = hs[:-1]
	# get xpreds by linear regression
	reg = LogisticRegression().fit(hs,xs)
	xpreds = reg.predict(hs)
	return xpreds, xs

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
	batch_x_train = batch_x[:int(len(batch_x)/2)]
	batch_y_train = batch_y[:int(len(batch_y)/2)]
	batch_x_test = batch_x[int(len(batch_x)/2):]
	batch_y_test = batch_y[int(len(batch_y)/2):]
	#
	X = tf.placeholder("float", [None, timesteps, 1])
	Y = tf.placeholder("float", [None, num_classes])
	#
	weights = {'out': tf.Variable(tf.random_normal([num_nodes, num_classes]))}
	biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
	#
	def RNN(x, weights, biases):
		# Prepare data shape to match `rnn` function requirements
    	# Current data input shape: (batch_size, timesteps, n_input)
    	# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		x = tf.unstack(x, timesteps, 1)

    	# Define a lstm cell with tensorflow
		lstm_cell = tf.nn.rnn_cell.LSTMCell(num_nodes, name='basic_lstm_cell', forget_bias=1.0)

    	# Get lstm cell output
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		# # dropout layer
		# outputs = tf.layers.dropout(outputs)

    	# Linear activation, using rnn inner loop last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	#
	logit_ = RNN(X, weights, biases)
	prediction = tf.math.exp(logit_)/(1+tf.math.exp(logit_))
	#
	#loss_op = tf.reduce_mean((prediction-Y)**2)
	loss_op = -tf.reduce_mean((1-Y)*tf.math.log(prediction)+Y*tf.math.log(1-prediction))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = optimizer.minimize(loss_op)
	init = tf.global_variables_initializer()
	losses = []
	with tf.Session() as sess:
		sess.run(init)
		for step in range(1, num_epochs+1):
			sess.run(train_op, feed_dict={X: batch_x_train, Y: batch_y_train})
		xpreds = sess.run(prediction, feed_dict={X: batch_x_test})
	xpreds = np.asarray([(x<0.5) for x in xpreds])
	return xpreds, batch_y_test

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

# f = open('EMLibrary/'+str(2)+'.hs')
# eM = f.readline()
# eM = f.readline()
# eM = f.readline()
# eM = f.readline()
# f.close()
#

eM = 'Fa [1,2,3] [(1,0,3),(1,1,2),(2,0,1),(2,1,1),(3,0,1)] [1,2,3]\n'
# simulate
xs, spasts, betas, Rs, Accs = simulate_FSM(eM,5000)

# autoregressive data
ks = np.arange(1,10)
H_GLM = []
acc_GLM = []
for k in ks:
	xpreds_GLM, xs_GLM = GLM(xs,k=k)
	foo = empRD(xs_GLM,xpreds_GLM)
	H_GLM.append(foo[0])
	acc_GLM.append(foo[1])

# reservoirs
ns = np.arange(1,61,5)
H_res = []
acc_res = []
for n in ns:
	xpreds_res, xs_res = reservoir(xs,spasts,num_nodes=n,activation_fnctn='tanh')
	foo = empRD(xs_res,xpreds_res)
	H_res.append(foo[0])
	acc_res.append(foo[1])

#
ns = np.arange(1,121,10)
H_lstm = []
acc_lstm = []
for n in ns:
	xpreds_lstm, xs_lstm = LSTM(xs,num_nodes=n,lr=1e-3,timesteps=10,num_epochs=500)
	xpreds_lstm = np.reshape(xpreds_lstm,[1,len(xpreds_lstm)])[0]
	xs_lstm = np.reshape(xs_lstm,[1,len(xs_lstm)])[0]
	foo = empRD(xs_lstm,xpreds_lstm)
	H_lstm.append(foo[0])
	acc_lstm.append(foo[1])

#
np.savez('DoubleEvenProcess2_PRD.npz',Rs=Rs,betas=betas,Accs=Accs,
	H_lstm=H_lstm,acc_lstm=acc_lstm,
	H_res=H_res,acc_res=acc_res,
	H_GLM=H_GLM,acc_GLM=acc_GLM)

pl.plot(Rs,Accs,'--k',linewidth=1.5)
pl.plot(H_GLM, acc_GLM,'+b',label='GLM')
pl.plot(H_res, acc_res,'og',label='Reservoir')
pl.plot(H_lstm, acc_lstm,'or',label='LSTM')
pl.legend(loc='best')
pl.xlabel('$Rate$',size=20)
pl.ylabel('$Accuracy$',size=20)
pl.savefig('DoubleEvenProcess2_PRD.pdf',bbox_inches='tight')
pl.show()
