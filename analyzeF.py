import numpy as np
import pylab as pl
import os.path
from sklearn.linear_model import LinearRegression

# get distance and beta for all data
# first shrink RD curve so that origin is (0,0)
# instead of (0,min(D))
# and (max R, max D) is (1,1).
# want to make statement about distance


# might be useful to have histograms of how far away the methods fall from maximal accuracy.
# np.max(Accs)-acc_lstm, np.max(Accs)-acc_res, np.max(Accs)-acc_GLM

# through out any filenames with max R <1e-10.

dist_lstms_all = []
dist_res_all = []
dist_GLM_all = []

acc_lstms_all = []
acc_res_all = []
acc_GLM_all = []

for i in range(1,6):
	count = 1
	filename = 'EM_'+str(i+1)+'_'+str(count)+'_PRD.npz'
	while os.path.exists(filename):
		dat = np.load(filename)
		Rs = np.asarray(dat['Rs']); Accs = np.asarray(dat['Accs'])
		if np.max(Rs)>1e-5:
			#
			H_lstm = dat['H_lstm']; acc_lstm = dat['acc_lstm']
			H_res = dat['H_res']; acc_res = dat['acc_res']
			H_GLM = dat['H_GLM']; acc_GLM = dat['acc_GLM']
			# normalize
			H_lstm = np.asarray(H_lstm)/np.max(Rs); acc_lstm = np.asarray(acc_lstm)/np.max(Accs)
			H_res = np.asarray(H_res)/np.max(Rs); acc_res = np.asarray(acc_res)/np.max(Accs)
			H_GLM = np.asarray(H_GLM)/(np.max(Rs)-np.min(Rs)); acc_GLM = np.asarray(acc_GLM)/np.max(Accs)
			#
			Rs /= np.max(Rs); Accs /= np.max(Accs)
			#
			acc_lstms_all.append(1-acc_lstm)
			acc_res_all.append(1-acc_res)
			acc_GLM_all.append(1-acc_GLM)
			#
			dist_lstm = []
			for j in range(len(H_lstm)):
				foo = np.sqrt((Rs-H_lstm[j])**2+(Accs-acc_lstm[j])**2)
				dist_lstm.append(np.min(foo))
			dist_lstms_all.append(dist_lstm)
			#
			dist_res = []
			for j in range(len(H_res)):
				foo = np.sqrt((Rs-H_res[j])**2+(Accs-acc_res[j])**2)
				dist_res.append(np.min(foo))
			dist_res_all.append(dist_res)
			#
			dist_GLM = []
			for j in range(len(H_GLM)):
				foo = np.sqrt((Rs-H_GLM[j])**2+(Accs-acc_GLM[j])**2)
				dist_GLM.append(np.min(foo))
			dist_GLM_all.append(dist_GLM)
		count += 1
		filename = 'EM_'+str(i+1)+'_'+str(count)+'_PRD.npz'

min_dist_GLM = []
min_dist_res = []
min_dist_lstm = []
for i in range(len(dist_GLM_all)):
	min_dist_GLM.append(np.min(dist_GLM_all[i]))
	min_dist_res.append(np.min(dist_res_all[i]))
	min_dist_lstm.append(np.min(dist_lstms_all[i]))
#
min_acc_GLM = []
min_acc_res = []
min_acc_lstm = []
for i in range(len(acc_GLM_all)):
	min_acc_GLM.append(np.min(acc_GLM_all[i]))
	min_acc_res.append(np.min(acc_res_all[i]))
	min_acc_lstm.append(np.min(acc_lstms_all[i]))

pl.rc('text', usetex=True)
pl.rc('font', size=16)

foo = np.reshape(dist_lstms_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True)
#pl.hist(foo,20,label='LSTM')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='LSTM')
#
foo = np.reshape(dist_res_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True)
#pl.hist(foo,20,label='Reservoir')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='Reservoir')
#
foo = np.reshape(dist_GLM_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True)
#pl.hist(foo,20,label='GLM')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='GLM')
#
pl.xlabel('Normalized distance',size=20)
pl.ylabel('Frequency',size=20)
pl.legend(loc='best')
pl.yscale('log')
pl.savefig('Hist_dist_PRD.pdf',bbox_inches='tight')
pl.show()

foo = np.reshape(acc_lstms_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True,range=(0,1))
#pl.hist(foo,20,label='LSTM')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='LSTM')
#
foo = np.reshape(acc_res_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True,range=(0,1))
#pl.hist(foo,20,label='Reservoir')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='Reservoir')
#
foo = np.reshape(acc_GLM_all,-1)
foo = foo[~np.isnan(foo)]
hist, bin_edges = np.histogram(foo,20,density=True,range=(0,1))
#pl.hist(foo,20,label='GLM')
pl.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),hist,'-x',label='GLM')
pl.xlabel('Normalized predictive distortion',size=20)
pl.ylabel('Frequency',size=20)
pl.legend(loc='best')
pl.yscale('log')
pl.savefig('Hist_acc.pdf',bbox_inches='tight')
pl.show()

# look at Cmu, hmu correlations for v2

dist_lstms_all = []
dist_res_all = []
dist_GLM_all = []

acc_lstms_all = []
acc_res_all = []
acc_GLM_all = []

Cmus = []
hmus = []

for i in range(1,6):
	count = 1
	filename = 'EM_'+str(i+1)+'_'+str(count)+'_PRD_v4.npz'
	while os.path.exists(filename):
		dat = np.load(filename)
		Rs = np.asarray(dat['Rs']); Accs = np.asarray(dat['Accs'])
		if np.max(Rs)>1e-5:
			#
			Cmu = dat['Cmu']; hmu = dat['hmu']
			Cmus.append(Cmu); hmus.append(hmu)
			H_lstm = dat['H_lstm']; acc_lstm = dat['acc_lstm']
			H_res = dat['H_res']; acc_res = dat['acc_res']
			H_GLM = dat['H_GLM']; acc_GLM = dat['acc_GLM']
			# normalize
			H_lstm = np.asarray(H_lstm)/np.max(Rs); acc_lstm = np.asarray(acc_lstm)/np.max(Accs)
			H_res = np.asarray(H_res)/np.max(Rs); acc_res = np.asarray(acc_res)/np.max(Accs)
			H_GLM = np.asarray(H_GLM)/(np.max(Rs)-np.min(Rs)); acc_GLM = np.asarray(acc_GLM)/np.max(Accs)
			#
			Rs /= np.max(Rs); Accs /= np.max(Accs)
			#
			acc_lstms_all.append(1-acc_lstm)
			print(min(1-acc_lstm))
			acc_res_all.append(1-acc_res)
			acc_GLM_all.append(1-acc_GLM)
			#
			dist_lstm = []
			for j in range(len(H_lstm)):
				foo = np.sqrt((Rs-H_lstm[j])**2+(Accs-acc_lstm[j])**2)
				dist_lstm.append(np.min(foo))
			dist_lstms_all.append(dist_lstm)
			#
			dist_res = []
			for j in range(len(H_res)):
				foo = np.sqrt((Rs-H_res[j])**2+(Accs-acc_res[j])**2)
				dist_res.append(np.min(foo))
			dist_res_all.append(dist_res)
			#
			dist_GLM = []
			for j in range(len(H_GLM)):
				foo = np.sqrt((Rs-H_GLM[j])**2+(Accs-acc_GLM[j])**2)
				dist_GLM.append(np.min(foo))
			dist_GLM_all.append(dist_GLM)
		count += 1
		filename = 'EM_'+str(i+1)+'_'+str(count)+'_PRD_v4.npz'

#
min_dist_GLM = []
min_dist_res = []
min_dist_lstm = []
for i in range(len(dist_GLM_all)):
	min_dist_GLM.append(np.min(dist_GLM_all[i]))
	min_dist_res.append(np.min(dist_res_all[i]))
	min_dist_lstm.append(np.min(dist_lstms_all[i]))
#
min_acc_GLM = []
min_acc_res = []
min_acc_lstm = []
for i in range(len(acc_GLM_all)):
	min_acc_GLM.append(np.min(acc_GLM_all[i]))
	min_acc_res.append(np.min(acc_res_all[i]))
	min_acc_lstm.append(np.min(acc_lstms_all[i]))
#
X = [[Cmus[i],hmus[i]] for i in range(len(Cmus))]
#
reg = LinearRegression().fit(X, min_acc_GLM)
print(reg.score(X, min_acc_GLM))
print(reg.coef_)
reg = LinearRegression().fit(X, min_acc_res)
print(reg.score(X, min_acc_res))
print(reg.coef_)
reg = LinearRegression().fit(X, min_acc_lstm)
print(reg.score(X, min_acc_lstm))
print(reg.coef_)

