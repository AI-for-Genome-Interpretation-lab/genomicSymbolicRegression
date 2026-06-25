#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  readGenes.py
#  
#  Copyright 2022 eddiewrc <eddiewrc@alnilam>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import numpy as np
import os, pickle, math
import matplotlib.pyplot as plt
import torch as t
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from plinkEpistasis import plink2EpistasisPredict
from scipy.stats import pearsonr
import feyn
from feyn.plots import plot_model_response_auto
from feyn.plots.interactive import interactive_activation_flow
import pandas as pd
from sympy import pprint, simplify
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score

def readSNPMatrix(f, header = True):
	row = []
	cols = []
	data = []
	ifp = open(f,"rb")
	lines = ifp.readlines()
	ifp.close()
	if header:
		cols = lines.pop(0).decode('utf8').strip().split(",")[1:]
	else:
		cols = []
	for l in lines:
		tmp = l.decode('utf8').strip().split(",")
		if header:
			row.append(tmp[0])
			data.append(np.array(tmp[1:], dtype=np.int8))
		else:
			data.append(np.array(tmp[:], dtype=np.int8))
		#assert len(tmp[1:]) == len(cols)
	data = np.array(data).T
	print( len(row), len(cols), data.shape)
	#print("Found %d %d matrix"% (len(row), len(cols)))
	return row, cols, data

def readCausativeSNPs(f):
	ifp = open(f, "r")
	l = ifp.readlines()
	rnames = []
	num = []
	for i in l:
		rnames.append(i.strip())
		print(i.strip())
		num.append(int(i.strip()[4:])-1)
	return rnames, num

def readPhenotypes(f):
	ifp = open(f, "r")
	l = ifp.readlines()
	l.pop(0)
	phen = []
	corresp = []
	for i, tmp in enumerate(l):
		#print(tmp, i)
		p, c = tmp.strip().replace("sample_","").split(",")
		assert int(c) == i+1
		phen.append(float(p))
		corresp.append(c)
	return phen, corresp

def readEpistaticPairsSNPs(f):
	ifp = open(f, "r")
	l = ifp.readlines()
	epiPairs = []
	num = []
	i = 0
	while i < len(l):
		if "[P1epistasis]" in l[i]:
			break
		i+=1
	i+=1
	while i < len(l):
		if "[P1heritability]" in l[i]:
			break
		tmp = l[i].strip().split(" ")
		epiPairs.append((tmp[0], tmp[1]))
		i+=1
	print("Found %d epistatic pairs" % len(epiPairs))
	return epiPairs
	
def readSimulations(f):
	causativeNames, causative = readCausativeSNPs(f+"SNPs_causative.list.txt")
	phenotypes, corresp = readPhenotypes(f+"simulated_phenotype.csv")
	epiPairs = readEpistaticPairsSNPs(f+"usedpars.txt")
	for p in epiPairs:
		assert p[0] in causativeNames
		assert p[1] in causativeNames
	assert len(phenotypes) == len(corresp)
	return causative, phenotypes, corresp, epiPairs

def nameColumns(cols):
	r = []
	for i in cols:
		r.append("SNP_"+str(i))
	return r

def getFewerFeatsModel(l):
	l = sorted(l, key=lambda x:len(x.features), reverse=False)
	print(l[0].to_query_string())
	print(l[1].to_query_string())
	print(l[2].to_query_string())
	rae

def getScores(Y, Yp, modelName):
	r = pearsonr(Y, Yp)[0]
	r2 = r2_score(Y, Yp)
	print(modelName+" r: ", r)
	print(modelName+" R²: ",r2)
	s = MinMaxScaler()
	Y = s.fit_transform(Y.reshape(-1,1)).T
	s = MinMaxScaler()
	Yp = s.fit_transform(Yp.reshape(-1,1)).T
	#print(Y.shape, Yp.shape)
	#print(Y, Yp) 
	ndcg = ndcg_score(Y, Yp)
	print(modelName+" NDCG: ", ndcg)
	return modelName, r, r2, ndcg

def checkFeatCorr(causative, features, TRAIN, TEST, columns):
	print(causative, features)
	corr = {}
	for c in causative:
		cname = "SNP_"+str(columns.index(c))
		#corr[c] = []
		for f in features:
			if cname == f:
				continue
			corr[cname,f] = pearsonr(TRAIN.loc[:,cname], TRAIN.loc[:,f])[0]
		#for i in TRAIN.columns:
		#corr[c].append(pearsonr(TRAIN[c], TRAIN[i])[0])
		c+=1
	print(corr)
	return corr

def main(args):
	FOLDER = "results/run100_synthLarge/"
	RUN_NAME = "run100"
	CONTINUE = -1#720
	'''
	heritability = [0.3, 0.6, 0.9]
	coeffOfVariation = 0.1
	qtl = [2,10,50,100,200]
	epi = ["0", "0.5" ,"1"]
	dom = ["0", "0.5", "1"]
	numDecoys = [0,10,50,100,200,500,1000, 1800]
	numSamples = [50,100,500,1000,2000]
	EPOCHS = [10]'''

	heritability = [0.6, 0.3]
	coeffOfVariation = 0.1
	qtl = [2,4,6,8,10,12,14,16,18,20,24,26,30,34,38,40,46,50]
	epi = ["0","1"]
	dom = ["0","1"]
	numDecoys = [0,100,500,1000,1900]
	numSamples = [500,1000,2000]
	EPOCHS = [10]

	PERC_TRAIN = 0.7
	totIter = len(heritability) * len(qtl) * len(epi) * len(dom) * len(numDecoys) * len(numSamples) * len(EPOCHS)
	if CONTINUE > 0:
		results = pickle.load(open(FOLDER+RUN_NAME+"_%d.pickle"%CONTINUE, "rb"))
	else:
		results = {}
	s = 0
	for ep in EPOCHS:
		for h in heritability:
			for c in qtl:
				for E in epi:
					for D in dom:
						for d in numDecoys:
							for n in numSamples:
								if s < CONTINUE:
									s+=1
									continue
								print("########################ITER: %d/%d (%.3f)" % (s, totIter, 100*s/float(totIter)))
								results[(h, c, D, E, d, n, ep)] = mainExecute((h, D, E, d, n, PERC_TRAIN, ep, c))
								s+=1
								if s % 10 == 0:
									pickle.dump(results, open(FOLDER+RUN_NAME+"_%d.pickle"%s,"wb"))
	
	pickle.dump(results, open(FOLDER+RUN_NAME+"FINAL100.pickle","wb"))

def mainExecute(args):
	results = {}
	if len(args) <= 1:
		heritability = 0.3
		DOM = "0"
		EPI = "0"
		numDecoys = 1000
		numSamples = 1000
		PERC_TRAIN = 0.7
		EPOCHS = 10
		qtl = 2
	else:
		heritability = args[0]
		DOM = args[1]
		EPI = args[2]
		numDecoys = args[3]
		numSamples = args[4]
		PERC_TRAIN = args[5]
		EPOCHS = args[6]
		qtl = args[7]
	TRAIN_SAMPLES = int(numSamples * PERC_TRAIN)
	print("RUN ARGS *************************************************************")
	print(args, TRAIN_SAMPLES)
	f = "data/synthetic/synthPhenotypesLargeFewQTL/herit_ %1.1f numQTL %d _Dfract %s _EpiAddOv %s/" % (heritability, qtl, DOM, EPI)
	print(f)
	
	#try:
	causative, phenotypes, corresp, epiPairs = readSimulations(f)
	#except:
	#	print("ERROR: %s not found"% f)
	#	return results
	_, snpNames, SNPDATA = readSNPMatrix("data/synthetic/synthSNPsLarge.csv")
	print(causative)
	print("DATA SHAPE: ",SNPDATA.shape)
	results["simpheCausative"] = causative
	results["epiPairs"] = epiPairs
	columns = list(range(0,SNPDATA.shape[1]))
	for i in causative:
		columns.remove(i)
	assert len(columns) + len(causative) == SNPDATA.shape[1]
	columns = causative + np.random.choice(columns, (numDecoys)).tolist() #add the randomly chosen decoys
	np.random.shuffle(columns) #make sure causative snps are not always at the beginning
	assert len(columns) == numDecoys + len(causative)
	SNPDATA, phenotypes = shuffle(SNPDATA, phenotypes)#shuffle samples before splitting just in case
	SNPDATA = SNPDATA[:numSamples]
	SNPDATA = SNPDATA[:,columns]
	SNPDATA = SNPDATA.astype(np.float32)
	phenotypes = phenotypes[:numSamples]
	phenotypes = (np.array(phenotypes) * -1).reshape(-1,1)
	print("Pheno data: ", np.mean(phenotypes), np.var(phenotypes), phenotypes.shape)
	#plt.violinplot(phenotypes)
	#plt.show()
	#PREPROCESS THE DATA
	#scaler = StandardScaler()
	#SNPDATA = scaler.fit_transform(SNPDATA).astype(np.float32)
	#phenoScaler = StandardScaler()
	#phenotypes = phenoScaler.fit_transform(phenotypes).astype(np.float32)
	phenotypes = phenotypes.reshape(-1)
	print(SNPDATA.dtype, phenotypes.dtype)
	#raw

	X = SNPDATA[:TRAIN_SAMPLES,:]
	Y = phenotypes[:TRAIN_SAMPLES]
	x = SNPDATA[TRAIN_SAMPLES:,:]
	y = phenotypes[TRAIN_SAMPLES:]
	print(SNPDATA.shape, phenotypes.shape)
	print(X.dtype)
	#REGULAR MODELS
	lin = Ridge()
	lin.fit(X, Y)
	Yp = lin.predict(X)
	yp = lin.predict(x)
	#getScores(Y, Yp, "Ridge Train")
	results["Ridge Test"] = getScores(y, yp, "Ridge Test")
	results["ridgeParams"] = lin.coef_

	lin = Lasso(max_iter=2000)
	lin.fit(X, Y)
	Yp = lin.predict(X)
	yp = lin.predict(x)
	#getScores(Y, Yp, "Lasso Train")
	results["Lasso Test"] = getScores(y, yp, "Lasso Test")
	results["LassoParams"] = lin.coef_
	
	nn = MLPRegressor(activation="tanh", learning_rate="adaptive", max_iter=400, learning_rate_init=1e-2)
	nn.fit(X, Y)
	Yp = nn.predict(X)
	yp = nn.predict(x)
	#getScores(Y, Yp, "MLP Train")
	results["MLP Test"] = getScores(y, yp, "MLP Test")

	rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
	rf.fit(X, Y)
	yp = rf.predict(x)
	results["RF Test"] = getScores(y, yp, "RF Test")
	results["RFParams"] = rf.feature_importances_

	#START THE MAGIC###############################################
	{"train_input":X, "train_label":Y, "test_input":x, "test_label":y}
	TRAIN = pd.DataFrame(X)
	#TRAIN.columns = TRAIN.columns.astype(str)
	TRAIN.columns = nameColumns(TRAIN.columns)
	TRAIN["label"] = Y
	
	#print(TRAIN)
	ql = feyn.QLattice()
	models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=EPOCHS, max_complexity=99, function_names=["add", "multiply"], threads = 16)
	#models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=EPOCHS, max_complexity=2*len(causative), function_names=["add", "multiply"], threads = 16)
	TEST = pd.DataFrame(x)
	#TEST.columns = TEST.columns.astype(str)
	TEST.columns = nameColumns(TEST.columns)
	TEST["label"] = y
	best = models[0]
	Yp = best.predict(TRAIN)
	yp = best.predict(TEST)
	#getScores(Y, Yp, "FeynBIC Train")
	results["FeynBIC Test"] = getScores(y, yp, "FeynBIC Test")
	results["best"] = best
	print("FEATURES: ", best.features)
	results["FEATURES"] = best.features
	cpos = []
	for c in causative:
		cpos.append("SNP_"+ str(columns.index(c)))
	print("THE CAUSATIVE VARIANTS CORRESPOND TO: ", cpos)
	results["CAUSATIVE"] = cpos
	#TODO#results["corrDecoys"] = checkFeatCorr(causative, best.features, TRAIN, TEST, columns)
	#fewFeatModel = getFewerFeatsModel(models)
	#best.plot(TRAIN, TEST, filename="plot1")
	#interactive_activation_flow(best, TEST)
	#best.savefig("feyn-signal-plot.svg")
	#best.plot_flow(TEST, filename="feyn-signal-plot.svg")
	print(best.to_query_string())
	#sympy_model = best.sympify(signif=3)
	#final = simplify(simplify(simplify(sympy_model)))
	#pprint(sympy_model, use_unicode=True)
	#pprint(final, use_unicode=True)
	#sympy_model.as_expr()
	#plt.scatter(yp, y, alpha=0.3)
	#p = best.plot_residuals(data=TEST)
	#plt.show()
	#plot_model_response_auto(model=best, data=TRAIN)
	#plt.show()
	
	#plotComparison(x, yp, y)

	try:
		yp_plink, plink_pairs = plink2EpistasisPredict(X, Y, x, n_top_pairs=max(1, len(epiPairs)))
		results["PLINK2 Test"] = getScores(y, yp_plink, "PLINK2 Test")
		results["PLINK2Pairs"] = plink_pairs
		detected_snps = set()
		for pair in plink_pairs:
			detected_snps.add("SNP_%d" % pair['snp1'])
			detected_snps.add("SNP_%d" % pair['snp2'])
		results["PLINK2Features"] = list(detected_snps)
		epi_pairs_orig = []
		for pair in plink_pairs:
			name1 = "SNP_" + str(columns[pair['snp1']] + 1)
			name2 = "SNP_" + str(columns[pair['snp2']] + 1)
			epi_pairs_orig.append(tuple(sorted([name1, name2])))
		results["PLINK2EpiPairs"] = epi_pairs_orig
	except Exception as e:
		print("PLINK2 error:", e)
		results["PLINK2 Test"] = ("PLINK2 Test", np.nan, np.nan, np.nan)
		results["PLINK2Pairs"] = []
		results["PLINK2Features"] = []
		results["PLINK2EpiPairs"] = []

	return results
'''
MIGRATION_CODES_TO_FNAME_MAP = {
    1000: "exp:1",
    1001: "gaussian:1",
    1002: "inverse:1",
    1003: "linear:1",
    1004: "log:1",
    1005: "sqrt:1",
    1006: "squared:1",
    1007: "tanh:1",
    2000: "add:2",
    2001: "gaussian:2",
    2002: "multiply:2",
'''


def plotComparison(TEST, yp, y):
	l1 = TEST[:, 0].astype(np.int32)
	l2 = TEST[:, 1].astype(np.int32)
	realmat = {}
	predmat = {}

	assert len(l2) == len(l1) == len(y) == len(yp)
	for i, p in enumerate(y):
		if not (l1[i], l2[i]) in realmat:
			realmat[(l1[i], l2[i])] = []
		realmat[(l1[i], l2[i])].append(y[i])
	for i, p in enumerate(yp):
		if not (l1[i], l2[i]) in predmat:
			predmat[(l1[i], l2[i])] = []
		predmat[(l1[i], l2[i])].append(yp[i])

	#print(realmat)
	realmattmp = np.zeros((3,3))
	predmattmp = np.zeros((3,3))
	for i in realmat.items():
		realmattmp[i[0][0], i[0][1]] = np.mean(i[1])
	for i in predmat.items():
		predmattmp[i[0][0], i[0][1]] = np.mean(i[1])
	realmat = realmattmp
	predmat = predmattmp
	plotPheno(realmat, predmat)
	#plt.savefig("toyPhenotyps.png", dpi=400)


def plotPheno(realmat, predmat, flip=True, colors = ["red", "green", "blue"], markers=["s","^","o"], locus1=["xx", "xX", "XX"], locus2=["yy", "yY","YY"]):#["YY", "yY", "yy"]):
	if flip:
		flip = np.array([[0,0,1],[0,1,0],[1,0,0]])
		locus2 = ["YY", "yY", "yy"]
		print(realmat)
		realmat = np.matmul(realmat, flip)*-1
		predmat = np.matmul(predmat, flip)*-1
	plt.rcParams.update(plt.rcParamsDefault)
	fig, axs = plt.subplots(figsize=(8,7),nrows=2, ncols=2, layout="tight")
	for i in range(0, realmat.shape[0]):
		axs[0,0].plot([0,1,2], realmat[i,:].squeeze(), color=colors[i], label=locus2[i], marker=markers[i], alpha=0.7)
	axs[0,0].set_xticks([0,1,2], locus1)
	axs[0,0].set_ylabel("Phenotype value")
	axs[0,0].legend()
	axs[0,0].grid(alpha=0.4)
	for i in range(0, predmat.shape[0]):
		axs[1,0].plot([0,1,2],  predmat[i,:].squeeze(), color=colors[i], label=locus2[i], marker=markers[i], alpha=0.7, ls = "--")
	axs[1,0].set_xticks([0,1,2], locus1)
	axs[1,0].set_ylabel("Predicted phenotype")
	axs[1,0].legend()
	axs[1,0].grid(alpha=0.4)

	#plot fucking mesh
	x = []
	y = []
	for i in [0,1,2]:
		for j in [0,1,2]:
			x.append(i)
			y.append(j)
	X, Y = np.meshgrid(x,y)
	#mat = np.matmul(mat*-1, flip)
	Z = []
	Zpred = []
	i = 0
	while i < X.shape[0]:
		j = 0
		while j < Y.shape[1]:
			Z.append(realmat[X[i,j], Y[i,j]])
			Zpred.append(predmat[X[i,j], Y[i,j]])
			j+=1
		i+=1
	Z = np.array(Z).reshape(X.shape)
	Zpred = np.array(Z).reshape(X.shape)
	#im2 = axs[1].imshow(mat)
	im2 = axs[0,1].contourf(x, y, Z, levels=20, origin="lower")
	axs[0,1].set_xticks([0,1,2], locus1[::-1])
	axs[0,1].set_yticks([2,1,0], locus2)
	axs[0,1].invert_xaxis()
	fig.colorbar(im2, ax=axs[0,1], aspect=50)
	#plt.show()
	im2 = axs[1,1].contourf(x, y, Zpred, levels=20, origin="lower")
	axs[1,1].set_xticks([0,1,2], locus1[::-1])
	axs[1,1].set_yticks([2,1,0], locus2)
	axs[1,1].invert_xaxis()
	fig.colorbar(im2, ax=axs[1,1], aspect=50)
	plt.show()


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
