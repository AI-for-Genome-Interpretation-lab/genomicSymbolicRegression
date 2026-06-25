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
from scipy.stats import pearsonr, spearmanr
import feyn
from feyn.plots import plot_model_response_auto
from feyn.plots.interactive import interactive_activation_flow
import pandas as pd
from sympy import pprint, simplify
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score


def standardize(x):
	a = np.array(x)
	mu = np.mean(a)
	std = np.std(a)
	x =  ((a - mu) / std)
	assert abs(np.mean(x)) < 1e-4 
	assert abs(np.std(x) - 1) < 1e-4
	return x.tolist()

def readData(f): 
	ifp = open(f, "r")
	lines = ifp.readlines()
	lines.pop(0)
	x = []
	y = []
	for l in lines:
		tmp = l.strip().split("\t")
		seq = tmp[0]
		fitness = float(tmp[-1])
		x.append(seq)
		y.append(fitness)
	print("Found %d samples" % len(x))
	return x, y

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

def expandFeatures(X):
# Define the 20 standard amino acids
	amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
	aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
	# Initialize an array to hold the one-hot encoded vectors
	r = []
	# Fill the matrix
	for s in X:
		#print(s)
		one_hot_matrix = np.zeros((len(X[0])* len(amino_acids)), dtype=int)
		for i, aa in enumerate(s):
			#print(aa)
			one_hot_matrix[i*len(amino_acids) + aa_to_index[aa]] = 1
			#print(sum(one_hot_matrix))
		r.append(one_hot_matrix.tolist())
			#input()
	return r

def splitPositions(X):
	r = []
	for s in X:
		tmp = []
		for aa in s:
			tmp.append(aa)
		r.append(tmp)
	return r

def main(args):
	results = {}
	PERC_TRAIN = 0.7
	numVars = 4
	dataX, dataY = readData("data/elife/elife-16965-supp1-v4.csv")
	numSamples = len(dataX)
	TRAIN_SAMPLES = int(numSamples * PERC_TRAIN)
	dataX, dataY = shuffle(dataX, dataY)
	#dataY = standardize(dataY)
	#print(min(dataY), max(dataY))
	X = dataX[:TRAIN_SAMPLES]
	Y = np.array(dataY[:TRAIN_SAMPLES])
	x = dataX[TRAIN_SAMPLES:]
	y = np.array(dataY[TRAIN_SAMPLES:])
	#REGULAR MODELS
	expandedX = expandFeatures(X)
	expandedx = expandFeatures(x)
	#print(X[0])
	#print(expandedX[0])
	lin = Ridge()
	lin.fit(expandedX, Y)
	Yp = lin.predict(expandedX)
	yp = lin.predict(expandedx)
	#getScores(Y, Yp, "Ridge Train")
	results["ridgePreds"] = yp
	results["Ridge Test"] = getScores(y, yp, "Ridge Test")
	results["ridgeParams"] = lin.coef_

	lin = Lasso(max_iter=2000)
	lin.fit(expandedX, Y)
	Yp = lin.predict(expandedX)
	yp = lin.predict(expandedx)
	#getScores(Y, Yp, "Lasso Train")
	results["Lasso Test"] = getScores(y, yp, "Lasso Test")
	results["LassoParams"] = lin.coef_
	results["LassoPreds"] = yp
	if True:
		nn = MLPRegressor(activation="tanh", learning_rate="adaptive", max_iter=400, learning_rate_init=1e-2)
		nn.fit(expandedX, Y)
		Yp = nn.predict(expandedX)
		yp = nn.predict(expandedx)
		#getScores(Y, Yp, "MLP Train")
		results["MLP Test"] = getScores(y, yp, "MLP Test")
		results["MLPPreds"] = yp
	#START THE MAGIC###############################################
	{"train_input":X, "train_label":Y, "test_input":x, "test_label":y}
	TRAIN = pd.DataFrame(splitPositions(X))
	TRAIN.columns = ["p0", "p1","p2","p3"]
	TRAIN["label"] = Y
	print(TRAIN)
	ql = feyn.QLattice()
	#models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=20, max_complexity=2*numVars-1, threads = 16, criterion='bic')
	#models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=40, max_complexity=numVars*2-1, function_names=["add", "multiply"], threads = 16, criterion='bic', keep_num_models_final=1000, query_string="? * ? * ?")
	models = ql.auto_run(TRAIN, output_name="label", kind="regression", n_epochs=40, max_complexity=numVars*2-1, function_names=["add", "multiply"], threads = 16, criterion='bic', keep_num_models_final=1000)
	TEST = pd.DataFrame(splitPositions(x))
	print(TEST)
	#TEST.columns = TEST.columns.astype(str)
	TEST.columns = TRAIN.columns[:-1]
	TEST["label"] = y
	best = models[0]
	results["best20"] = models[:]
	print(len(models))
	Yp = best.predict(TRAIN)
	yp = best.predict(TEST)
	#getScores(Y, Yp, "FeynBIC Train")
	results["TEST"] = TEST
	retults["TRAIN"] = TRAIN
	results["FeynBIC Test"] = getScores(y, yp, "FeynBIC Test")
	results["bestFeynBic"] = best
	print("FEATURES: ", best.features)
	results["FEATURESFeynBic"] = best.features
	print(best.to_query_string())
	for i in best.features:
		print(best.get_parameters(i))
	pickle.dump(results, open("results/elife/elifeResults_7vars_ADDMUL.pickle", "wb"))
	#plt.show()
	#return results



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



if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
