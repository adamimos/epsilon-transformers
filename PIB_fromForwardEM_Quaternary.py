# Information bottleneck method for discrete state spaces
# Returns the theoretical rate-distortion curve and the features at any given beta
# Will not work for state spaces with |X|*|Y| > 2^32 or so, algorithm is nonconvex.

import numpy as np
from numpy import linalg as LA
import scipy as sp
import scipy.io	 # NOQA

from time import time as now

def PIB_fromForwardEM_Quaternary(T0s,T1s,T0t,T1t,eps,betamax,betamin,numsteps):
	"""
	T0, T1 are the transition matrices of the forward time EM
	eps is the coarse graining we allow on the mixed state spaces--
	two states are the same if each entry is within 10^-eps of the other
	This function returns a predictive information curve for this binary output PDFA.
	"""

	# Calculate the prob dist over forward time CS

	T = T0s+T0t+T1s+T1t
	D, V = LA.eig(T)
	V = V.T

	def near(a, b, rtol = 1e-5, atol = 1e-8):
		return np.abs(a-b)<(atol+rtol*np.abs(b))

	pi = V[near(D, 1.0)]
	pi = pi/np.sum(pi)
	pi = np.real(pi)

	# Calculate reverse time T0, T1

	D = np.diagflat(pi)
	Dinv = np.diagflat(1/pi)
	T0s = np.dot(np.dot(D,T0s.T),Dinv)
	T0t = np.dot(np.dot(D,T0t.T),Dinv)
	T1s = np.dot(np.dot(D,T1s.T),Dinv)
	T1t = np.dot(np.dot(D,T1t.T),Dinv)

	# Calculate mixed states
	# Say that two mixed states are equal if their L_infinity norm is less than 1e-eps
	# this requirement is codified by precision = eps option in np

	np.set_printoptions(precision=eps)
	all_MS = {} # key is the index of the mixed state, value is the mixed state value
	Trans0s = {} # keys are states we come from, values are (trans prob of 0, state to go to)
	Trans1s = {} # keys are states we come from, values are (trans prob of 1, state to go to)
	Trans0t = {}
	Trans1t = {}
	old_MS = [pi]
	new_MS = []

	# In this part, we make a mixed state presentation

	# First, burn in period.  We go three tree depths before we record stuff.
	for i in xrange(3):
		for j in old_MS:
			trans0s = np.sum(np.dot(T0s,j.T))
			if trans0s>0:
				newMS = (np.dot(T0s,j.T)/trans0s).T
				new_MS.append(newMS)
			trans0t = np.sum(np.dot(T0t,j.T))
			if trans0t>0:
				newMS = (np.dot(T0t,j.T)/trans0t).T
				new_MS.append(newMS)
			trans1s = np.sum(np.dot(T1s,j.T))
			if trans1s>0:
				newMS = (np.dot(T1s,j.T)/trans1s).T
				new_MS.append(newMS)
			trans1t = np.sum(np.dot(T1t,j.T))
			if trans1t>0:
				newMS = (np.dot(T1t,j.T)/trans1t).T
				new_MS.append(newMS)
		old_MS[:] = []
		for j in new_MS:
			old_MS.append(j)
		new_MS[:] = []

	# A state is declared redundant to an old state if Trans0 or Trans1 already has that key

	# To speed this part up: get rid of duplicate threads in old_MS
	count = 0
	while len(all_MS)<5000 and len(old_MS)>0:
		# get new states in new_MS
		for j in old_MS:
			trans0s = np.sum(np.dot(T0s,j.T))
			if trans0s>0:
				newMS = (np.dot(T0s,j.T)/trans0s).T
				new_MS.append(newMS)
				Trans0s[str(j)] = [trans0s, str(newMS)]
				if not all_MS.has_key(str(newMS)):
					all_MS[str(newMS)] = [count,newMS]
					count += 1
			trans0t = np.sum(np.dot(T0t,j.T))
			if trans0t>0:
				newMS = (np.dot(T0t,j.T)/trans0t).T
				new_MS.append(newMS)
				Trans0t[str(j)] = [trans0t, str(newMS)]
				if not all_MS.has_key(str(newMS)):
					all_MS[str(newMS)] = [count,newMS]
					count += 1
			trans1s = np.sum(np.dot(T1s,j.T))
			if trans1s>0:
				newMS = (np.dot(T1s,j.T)/trans1s).T
				new_MS.append(newMS)
				Trans1s[str(j)] = [trans1s, str(newMS)]
				if not all_MS.has_key(str(newMS)):
					all_MS[str(newMS)] = [count,newMS]
					count += 1
			trans1t = np.sum(np.dot(T1t,j.T))
			if trans1t>0:
				newMS = (np.dot(T1t,j.T)/trans1t).T
				new_MS.append(newMS)
				Trans1t[str(j)] = [trans1t, str(newMS)]
				if not all_MS.has_key(str(newMS)):
					all_MS[str(newMS)] = [count,newMS]
					count += 1
		# clear old_MS
		old_MS[:] = []
		for j in new_MS:
			# don't look through things we're already looking through
			if not (Trans0s.has_key(str(j)) or Trans0t.has_key(str(j)) or Trans1s.has_key(str(j)) or Trans1t.has_key(str(j))):
				old_MS.append(j)
		# clear new_MS
		new_MS[:] = []

	# for the remaining old_MS, just connect them to whatever states they are connected to.

	for j in old_MS:
		trans0s = np.sum(np.dot(T0s,j.T))
		if trans0s>0:
			newMS = (np.dot(T0s,j.T)/trans0s).T
			if all_MS.has_key(str(newMS)):
				Trans0s[str(j)] = [trans0s, str(newMS)]
		trans0t = np.sum(np.dot(T0t,j.T))
		if trans0t>0:
			newMS = (np.dot(T0t,j.T)/trans0t).T
			if all_MS.has_key(str(newMS)):
				Trans0t[str(j)] = [trans0t, str(newMS)]
		trans1s = np.sum(np.dot(T1s,j.T))
		if trans1s>0:
			newMS = (np.dot(T1s,j.T)/trans1s).T
			if all_MS.has_key(str(newMS)):
				Trans1s[str(j)] = [trans1s, str(newMS)]
		trans1t = np.sum(np.dot(T1t,j.T))
		if trans1t>0:
			newMS = (np.dot(T1t,j.T)/trans1t).T
			if all_MS.has_key(str(newMS)):
				Trans1t[str(j)] = [trans1t, str(newMS)]

	# make a transition matrix for all_MS

	Trans = np.zeros((len(all_MS),len(all_MS)))
	# we have an implicit ordering of states, which is given by the value of all_MS
	for i in all_MS:
		if Trans0s.has_key(i):
			foo = Trans0s[i]
			row = all_MS[foo[1]]; row=row[0]
			col = all_MS[i]; col=col[0]
			Trans[row,col] += foo[0]
		if Trans0t.has_key(i):
			foo = Trans0t[i]
			row = all_MS[foo[1]]; row=row[0]
			col = all_MS[i]; col=col[0]
			Trans[row,col] += foo[0]
		if Trans1s.has_key(i):
			foo = Trans1s[i]
			row = all_MS[foo[1]]; row=row[0]
			col = all_MS[i]; col=col[0]
			Trans[row,col] += foo[0]
		if Trans1t.has_key(i):
			foo = Trans1t[i]
			row = all_MS[foo[1]]; row=row[0]
			col = all_MS[i]; col=col[0]
			Trans[row,col] += foo[0]

	# Calculate the prob. dist. of reverse time causal states

	#D,V = sp.sparse.linalg.eigs(Trans, k=2)
	D, V = LA.eig(Trans)
	V = V.T
	piR = V[near(D, 1.0)]
	piR = np.abs(piR)
	piR = piR/np.sum(piR)
	# get rid of super low probability states
	piR[np.abs(piR)<1e-3] = 0 # arbitrary cutoff
	piR = piR/np.sum(piR)

	# From piR and all_MS, construct p

	p = np.zeros((np.count_nonzero(pi),np.count_nonzero(piR)))
	count = 0
	for i in all_MS:
		foo = all_MS[i]
		py = piR[0,foo[0]]
		pxgy = foo[1][0]
		if py>0:
			p[:,count] = py*pxgy.T
			count += 1

	# Now get the PIB curve

	m,n = np.shape(p)
	pX = np.sum(p,1)
	pY = np.sum(p,0)
	pYgX = np.dot(np.diag(1/pX),p)
	uYgX = np.log(pYgX)

	# Lagrange multiplier stepsize
	b=np.linspace(betamax,betamin,numsteps)

	# initial conditions
	pRgX = np.eye(m) # conditional of R given X
	pR = np.dot(pRgX,pX) # marginal of R
	pXgR = np.dot(np.dot(np.diag(1/pR),pRgX),np.diag(pX))
	pYgR = np.dot(pXgR,pYgX)

	# get the initial distortion matrix
	d = np.zeros((m,m))
	for i in xrange(m):
		for j in xrange(m):
			d[j,i]=np.nansum(pYgX[i,:]*np.log2(pYgX[i,:]/pYgR[j,:]))

	# the parametric rate and distortion curves
	R=np.zeros((1,numsteps))
	D=np.zeros((1,numsteps))

	# calculate E
	#E = -np.nansum(pY*np.log2(pY))+np.dot(pX,np.nansum(pYgX*np.log2(pYgX),1))

	# start the procedure!

	foo3, foo4 = np.meshgrid(np.log(pX),np.log(pX))

	for i in range(0,numsteps):
		for t in range(0,50): # arbitrary, fix this
			# calculate exp(beta*d)
			foo1, foo2 = np.meshgrid(np.log(pR),np.log(pR))
			uRgX = foo2-b[i]*d
			pRgX = np.exp(uRgX) #np.dot(np.diag(pR),np.exp(-b[i]*d))
			# calculate new partition function
			Z = np.sum(pRgX,0)
			# calculate new conditional distribution (with old marginal)
			foo1, foo2 = np.meshgrid(np.log(Z),np.log(Z))
			uRgX = uRgX - foo1
			pRgX = np.exp(uRgX)
			# calculate new marginal
			pR = np.dot(pRgX,pX)
			# calculate new p(y|R) from new conditional
			foo1, foo2 = np.meshgrid(np.log(pR),np.log(pR))
			uXgR = -foo2+uRgX+foo3
			pXgR = np.exp(uXgR)
			pYgR = np.dot(pXgR,pYgX)
			uYgR = np.log(pYgR)
			# calculate new distortion matrix
			for j in xrange(m):
				for k in xrange(m):
					d[k,j]=np.nansum(pYgX[j,:]*(uYgX[j,:]-uYgR[k,:]))
		# calculate new rate, I[R,X]
		R[0,i]=-np.nansum(pR*np.log2(pR))+np.dot(pX,np.nansum(pRgX*np.log2(pRgX),0))
		if R[0,i]<.001:
			break
		# calculate new I[R,Y]
		D[0,i]=-np.nansum(pY*np.log2(pY))+np.dot(pR,np.nansum(pYgR*np.log2(pYgR),1))

	return p, b, R, D