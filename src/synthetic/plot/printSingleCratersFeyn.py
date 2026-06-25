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
from scipy.stats import pearsonr, spearmanr
import feyn
from feyn.plots import plot_model_response_auto
from feyn.plots.interactive import interactive_activation_flow
import pandas as pd
from sympy import pprint, simplify
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score

# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)


def nameColumns(cols):
	r = []
	for i in cols:
		r.append("SNP_"+str(i))
	return r


def getScores(Y, Yp, modelName):
	r = pearsonr(Y, Yp)[0]
	r2 = r2_score(Y, Yp)
	spearman = spearmanr(Y, Yp)[0]
	print(modelName+" r: ", r)
	print(modelName+" R²: ",r2)
	print(modelName+" Spearman: ", spearman)
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
	if args[1] == "sigmoid":
		FOLDER = "results/run_CraterSigmoid/"
		RUN_NAME = "runCraterSigmoidPlot"
		DATASETS = pickle.load(open("data/synthetic/craterModel/dist_sigmoidCrater.pickle", "rb"))
	elif args[1] == "gaussian":
		FOLDER = "results/run_CraterGauss/"
		RUN_NAME = "runCraterGaussPlot"
		DATASETS = pickle.load(open("data/synthetic/craterModel/dist_gaussianCrater.pickle", "rb"))
	else:
		print("vaffa")
		return
	CONTINUE = -1#720

	samples = [100,500,1000,2000,3000,4000]
	qtl = [2,4,8,16,32,50,100]
	EPOCHS = 25
	PERC_TRAIN = 0.7
	if CONTINUE > 0:
		results = pickle.load(open(FOLDER+RUN_NAME+"_%d.pickle"%CONTINUE, "rb"))
	else:
		results = {}
	a = 0
	#s, d, q, hammingDist, fitness, snps, reference, referenceSeq, causative, "euclideanDist_gaussian", SNP_file)
	#print("########################ITER: %d/%d (%.3f)" % (s, totIter, 100*s/float(totIter)))
	totIter = len(qtl) * len(samples)
	for ds in DATASETS:
		s = ds[0]
		q = ds[2]
		#if s == 4000:
		a += 1
		#if q != 16 or s != 3000:
		#	continue
		results[(s, q)] = mainExecute(s, q, PERC_TRAIN, ds, EPOCHS, args[1])
		print("########################ITER: %d/%d (%.3f)" % (a, totIter, 100*s/float(totIter)))
		if a % 10 == 0:
			pickle.dump(results, open(FOLDER+RUN_NAME+"_%d.pickle"%a,"wb"))
	
	pickle.dump(results, open(FOLDER+RUN_NAME+"FINAL.pickle","wb"))

def mainExecute(s, q, PERC_TRAIN, ds, EPOCHS, func):
	results = {}
	#0  1  2   3             4        5    6          7               8          9                   
	#s, d, q, hammingDist, fitness, snps, reference, referenceSeq, causative, "euclideanDist_gaussian", SNP_file)
	phenotypes = ds[4]
	SNPDATA = ds[5]
	numSamples = SNPDATA.shape[0]
	numVars = SNPDATA.shape[1]
	TRAIN_SAMPLES = int(numSamples * PERC_TRAIN)
	print("RUN ARGS *************************************************************")
	print(SNPDATA.dtype, phenotypes.dtype)
	print(SNPDATA.shape, phenotypes.shape)
	#raw
	genoDistTest = ds[3][TRAIN_SAMPLES:]
	X = SNPDATA[:TRAIN_SAMPLES,:]
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	Y = phenotypes[:TRAIN_SAMPLES]
	x = SNPDATA[TRAIN_SAMPLES:,:]
	x = scaler.transform(x)
	y = phenotypes[TRAIN_SAMPLES:]
	print(SNPDATA.shape, phenotypes.shape, ds[3].shape)
	print(X.dtype)
	#REGULAR MODELS
	lin = Ridge()
	lin.fit(X, Y)
	Yp = lin.predict(X)
	yp = lin.predict(x)
	#getScores(Y, Yp, "Ridge Train")
	results["ridgePreds"] = yp
	results["Ridge Test"] = getScores(y, yp, "Ridge Test")
	results["ridgeParams"] = lin.coef_

	lin = Lasso(max_iter=2000)
	lin.fit(X, Y)
	Yp = lin.predict(X)
	yp = lin.predict(x)
	#getScores(Y, Yp, "Lasso Train")
	results["Lasso Test"] = getScores(y, yp, "Lasso Test")
	results["LassoParams"] = lin.coef_
	results["LassoPreds"] = yp
	
	nn = MLPRegressor(activation="tanh", learning_rate="adaptive", max_iter=400, learning_rate_init=1e-2)
	nn.fit(X, Y)
	Yp = nn.predict(X)
	yp = nn.predict(x)
	#getScores(Y, Yp, "MLP Train")
	results["MLP Test"] = getScores(y, yp, "MLP Test")
	results["MLPPreds"] = yp

	rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
	rf.fit(X, Y)
	yp = rf.predict(x)
	results["RF Test"] = getScores(y, yp, "RF Test")
	results["RFPreds"] = yp

	#START THE MAGIC###############################################
	{"train_input":X, "train_label":Y, "test_input":x, "test_label":y}
	TRAIN = pd.DataFrame(X)
	#TRAIN.columns = TRAIN.columns.astype(str)
	TRAIN.columns = nameColumns(TRAIN.columns)
	TRAIN["label"] = Y
	
	#print(TRAIN)
	ql = feyn.QLattice()
	models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=EPOCHS, max_complexity=2*numVars, function_names=["add", "multiply"], threads = 16, criterion='bic')
	TEST = pd.DataFrame(x)
	#TEST.columns = TEST.columns.astype(str)
	TEST.columns = nameColumns(TEST.columns)
	TEST["label"] = y
	best = models[0]
	Yp = best.predict(TRAIN)
	yp = best.predict(TEST)
	#getScores(Y, Yp, "FeynBIC Train")
	results["FeynBIC Test"] = getScores(y, yp, "FeynBIC Test")
	results["bestFeynBic"] = best
	print("FEATURES: ", best.features)
	results["FEATURESFeynBic"] = best.features
	cpos = []
	for c in list(range(0,numVars)):
		cpos.append("SNP_"+ str(c))
	print("THE CAUSATIVE VARIANTS CORRESPOND TO: ", cpos)
	print(best.to_query_string())
	results["CAUSATIVEFeynBic"] = cpos

	x1, y1, ystd = sortPairs(genoDistTest, yp)
	results["feynBicLinePlot"] = (x1, y1, ystd)
	
	ql = feyn.QLattice()
	if func == "gaussian":
		models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=EPOCHS, max_complexity=2*numVars, function_names=["add", "multiply", "gaussian"], threads = 16, criterion='bic')
	elif func == "sigmoid":	
		models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=EPOCHS, max_complexity=2*numVars, function_names=["add", "multiply", "exp"], threads = 16, criterion='bic')
	else:
		raise Exception("fck")
	#TEST = pd.DataFrame(x)
	#TEST.columns = TEST.columns.astype(str)
	#TEST.columns = nameColumns(TEST.columns)
	#TEST["label"] = y
	best = models[0]
	Yp = best.predict(TRAIN)
	yp = best.predict(TEST)
	#getScores(Y, Yp, "FeynBIC Train")
	print(len(y), len(yp))
	results["FeynGauss Test"] = getScores(y, yp, "FeynGauss Test")
	results["bestFeynGauss"] = best
	print("FEATURES: ", best.features)
	results["FEATURESFeynGauss"] = best.features
	cpos = []
	for c in list(range(0,numVars)):
		cpos.append("SNP_"+ str(c))
	print("THE CAUSATIVE VARIANTS CORRESPOND TO: ", cpos)
	print(best.to_query_string())
	results["CAUSATIVEFeynGauss"] = cpos
	x1, y1, ystd1 = sortPairs(genoDistTest, yp)
	results["feynGaussLinePlot"] = (x1, y1, ystd1)
	
	hammingDist = ds[3]
	plt.rcParams.update(plt.rcParamsDefault)
	x, y, ystd = sortPairs(genoDistTest, y)
	plt.plot(x, y, color="black")
	results["labelLinePlot"] = (x, y, ystd)

	ALPHA = 0.1
	x, y, ystd = sortPairs(genoDistTest, results["ridgePreds"])
	results["ridgeLinePlot"] = (x, y, ystd)
	plt.plot(x, y, color="red", label="Ridge")
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="red")
	#plt.plot(tmp[:,0],tmp[:,1], color="red")
	x, y, ystd = sortPairs(genoDistTest, results["LassoPreds"])
	results["LassoLinePlot"] = (x, y, ystd)
	plt.plot(x, y, color="darkred", label="Lasso")
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="darkred")
	x, y, ystd = sortPairs(genoDistTest, results["MLPPreds"])
	results["MLPLinePlot"] = (x, y, ystd)
	plt.plot(x, y, color="blue", alpha=0.5, label="MLP")
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="blue")
	x, y, ystd = sortPairs(genoDistTest, results["RFPreds"])
	results["RFLinePlot"] = (x, y, ystd)
	plt.plot(x, y, color="orange", alpha=0.5, label="RF")
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="orange")

	x, y, ystd = results["feynBicLinePlot"]
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="lightgreen")
	plt.plot(x, y, color="lightgreen", label="Feyn(+*)")
	x, y, ystd = results["feynGaussLinePlot"]
	plt.fill_between(x, y - ystd*0.5, y + ystd*0.5, alpha=ALPHA, color="darkgreen")
	plt.plot(x, y, color="darkgreen", label="Feyn(+*N)")
	if func == "sigmoid":
		plt.title("Sigmoid landscape, %d samples, %d QTL"%(s,q))
	elif func == "gaussian":
		plt.title("Gaussian landscape, %d samples, %d QTL"%(s,q))
	else:
		raise Exception("fck")
	plt.ylabel("Phenotype value (N(d)))")
	plt.xlabel("Genotype distance value d")
	plt.legend()
	plt.tight_layout()
	#plt.scatter(genoDistTest, yp, color="green", alpha=ALPHA)
	#plt.scatter(genoDistTest, results["ridgePreds"], color="red", alpha=ALPHA)
	#plt.scatter(genoDistTest, results["LassoPreds"], color="darkred", alpha=ALPHA)
	#plt.scatter(genoDistTest, results["MLPPreds"], color="blue", alpha=ALPHA)
	#plt.plot(genoDistTest, results["ridgePreds"], color="red")
	#plt.plot(genoDistTest, results["LassoPreds"], color="darkred")
	#plt.plot(genoDistTest, results["MLPPreds"], color="blue")
	if func == "sigmoid":
		plt.savefig(os.path.join(OUTDIR, "linePlotSigmoidS%dQ%d.png"%(s,q)), dpi=400)
	elif func == "gaussian":	
		plt.savefig(os.path.join(OUTDIR, "linePlotGaussianS%dQ%d.png"%(s,q)), dpi=400)
	else:
		raise Exception("fck")
	plt.clf()
	#plt.savefig("linePlotGaussS%dQ%d.png"%(s,q), dpi=400)
	#plt.show()
	return results

def sortPairs(geno, y):
	tmp = np.stack((geno, y), axis=1)
	sorted_indices = np.argsort(tmp[:, 0])
	sorted_concatenated = tmp[sorted_indices]
	x = []
	y = []
	ystd = []
	for i in np.unique(sorted_concatenated[:,0]):
		x.append(i)
		selection = sorted_concatenated[sorted_concatenated[:,0]==i][:,1:]
		y.append(np.mean(selection))
		ystd.append(np.std(selection))
	return x, y, np.array(ystd)


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
