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

# All output files go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)
import matplotlib.pyplot as plt
import torch as t
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from scipy.stats import pearsonr, spearmanr
import feyn
from feyn.tools import get_model_parameters
from feyn.plots import plot_model_response_auto
from feyn.plots.interactive import interactive_activation_flow
import pandas as pd
from sympy import *
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score
from collections import defaultdict

def parseAtoms(l):
	r = []
	for s in l:
		r.append(str(s)[:2])
	return r

def computeDerivatives(eq):
	partialD = {}
	totS = len(eq.free_symbols)
	for s in eq.free_symbols:
		partialD[s] = diff(eq, s)
	mixedD = {}
	ratio = {}
	i = 0
	for s1 in eq.free_symbols:
		for s2 in eq.free_symbols:
			if str(s2)[:2] == str(s1)[:2]:
				continue
			i+=1
			perc = 100.0*i/(totS*totS-1)/2
			if i % 10 == 0:
				print(perc)
			#print(s1, s2)
			t = sorted([str(s1), str(s2)])
			if tuple(t) not in mixedD:
				mixedD[tuple(t)] = diff(eq, s1, s2)
				#ratio[tuple(t)] = mixedD[tuple(t)] / (partialD[s1] * partialD[s2] + 1e-8))
	print("MIXED")
	for t in mixedD.items():
		print(t)
		print("ATOMS: ",t[1].atoms(Symbol))
	#print("RATIOS")
	#for t in ratio.items():
	#	print(t)
	return partialD, mixedD, ratio
	
	

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

def collect_multiplied_pairs(expr):
	# Ensure the expression is in a form where we can easily analyze products
	expr = simplify(expand(expr))
	print("fatto")	
	# Get the list of symbols in the expression
	symbols = expr.free_symbols
	print(symbols)
	# Initialize an empty set to store unique pairs of multiplied symbols
	pairs = set()
	
	# Loop through each term in the expanded expression
	for term in expr.as_ordered_terms():
		# Get the factors of the term
		factors = Mul.make_args(term)
		
		# Extract only the symbols from the factors
		symbol_factors = [f for f in factors if isinstance(f, Symbol)]
		
		# If there are at least two symbols, add their pairs to the set
		if len(symbol_factors) > 1:
			for i in range(len(symbol_factors)):
				for j in range(i + 1, len(symbol_factors)):
					# Add the pair as a frozenset to ensure uniqueness (order doesn't matter)
					pairs.add(frozenset([symbol_factors[i], symbol_factors[j]]))
	
	# Convert the set of frozensets to a list of tuples for easier reading
	pairs_list = [tuple(pair) for pair in pairs]
	
	return pairs_list

def filterNames(l):
	r = set()
	for p in l:
		tmp = tuple(sorted([str(p[0]).split("_")[0], str(p[1]).split("_")[0]]))
		if tmp[0] != tmp[1]:
			r.add(tmp)
	return list(r)

def countPairs(l):
	hist = {}
	for p in l:
		if not p in hist:
			hist[p] = 1
		hist[p] += 1
	return hist

def rename_variables(expr):
	for s in ["p0","p1","p2","p3"]:
		expr = expr.replace(
			lambda sym: sym.is_Symbol and sym.name.startswith(s),
			lambda sym: Symbol(s))

	return expr


def mainSym(args):
	#DATA = pickle.load(open("elife/elifeResults_20epochs_8numVar_addmulFUnc.pickle", "rb"))
	DATA = pickle.load(open("results/elife/elifeResults_7vars_ADDMUL.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults_20epochs_2x20numvar_addMulFunc.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults_20epochs_allFuncs_numVars8.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults8numVars_50epochs.pickle", "rb"))
	print(DATA.keys())
	models = DATA["best20"]
	print(len(models))
	results = pickle.load(open("results/elife/derivativesResults.pickle", "rb"))

	symIntDB = {}
	for m in results.items():
		model = m[0]
		symRes = m[1]["symbolic"]
		catRes = m[1]["categorical"]
		
		for t in symRes[1].items(): #guardo nelle mixed
			pair = sorted([t[0][0][:2], t[0][1][:2]])
			expr = rename_variables(t[1])
			#expr =  expand(expr).replace(lambda t: t.is_Number, lambda t: 1)
			if tuple(pair) not in symIntDB:
				symIntDB[tuple(pair)] = {"atoms":[], "expr":[]} 
			if type(expr) != int and expr.free_symbols:
				print(m[0].sympify().expand(), pair, expr.expand())#, ",", expr - expr.as_coefficients_dict()[1])
				symIntDB[tuple(pair)]["atoms"] += parseAtoms(expr.atoms(Symbol))
				symIntDB[tuple(pair)]["expr"].append(expr)
	
	#pickle.dump(symIntDB, open("results/elife/symIntPlotdata.pickle", "wb"))

def mainCat(args):
	#DATA = pickle.load(open("elife/elifeResults_20epochs_8numVar_addmulFUnc.pickle", "rb"))
	DATA = pickle.load(open("results/elife/elifeResults_7vars_ADDMUL.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults_20epochs_2x20numvar_addMulFunc.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults_20epochs_allFuncs_numVars8.pickle", "rb"))
	#DATA = pickle.load(open("elife/elifeResults8numVars_50epochs.pickle", "rb"))
	print(DATA.keys())
	models = DATA["best20"]
	print(len(models))
	results = pickle.load(open("results/elife/derivativesResults.pickle", "rb"))

	catIntDB = {}
	for m in results.items():
		model = m[0]
		symRes = m[1]["symbolic"]
		catRes = m[1]["categorical"]
		
		for t in catRes[1].items(): #guardo nelle mixed
			pair = sorted([t[0][0][:2], t[0][1][:2]])
			#expr = rename_variables(t[1])
			#expr =  expand(expr).replace(lambda t: t.is_Number, lambda t: 1)
			expr = expand(t[1])
			if tuple(pair) not in catIntDB:
				catIntDB[tuple(pair)] = [] 
			if type(expr) != int and expr.free_symbols:
				#print(pair, expr - expr.as_coefficients_dict()[1])
				#catIntDB[tuple(pair)]["atoms"] += parseAtoms(expr.atoms(Symbol))
				#print(expr)
				#input() 
				catIntDB[tuple(pair)].append(expr.as_coefficients_dict())
	finalDic = {}
	for i in catIntDB.items():
		finalDic[i[0]] = averageDic(merge_dicts(i[1]))

	pickle.dump(finalDic, open("results/elife/catIntPlotdata.pickle", "wb"))

def averageDic(d):
	r = {}
	for i in d.items():
		r[i[0]] = np.mean(i[1])
	return r

def merge_dicts(dict_list):
	# Use a defaultdict to store lists of values for each key
	merged_dict = defaultdict(list)
	
	# Iterate through each dictionary in the list
	for d in dict_list:
		for key, value in d.items():
			# Append the value to the list for the corresponding key
			merged_dict[key].append(value)
	
	return dict(merged_dict)  # Convert back to a regular dictionary if desired

'''
	for c, i in enumerate(models):
		i.savefig(os.path.join(OUTDIR, "prova"+str(c)+".svg"))
		print(i.to_query_string())
		s = i.sympify(signif=1)
		SpartialD, SmixedD, Sratio = computeDerivatives(s)
		print("ALL CAT")
		s = i.sympify(signif=1, symbolic_cat=False)
		CatPartialD, CatMixedD, Catratio = computeDerivatives(s)
		results[i] = {"symbolic":(SpartialD, SmixedD, Sratio), "categorical":(CatPartialD, CatMixedD, Catratio)}
'''




if __name__ == '__main__':
	import sys
	sys.exit(mainSym(sys.argv))
	#sys.exit(mainCat(sys.argv))
