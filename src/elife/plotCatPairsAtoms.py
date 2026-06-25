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

def countAtoms(l):
	c = []
	for a in ["p0","p1", "p2", "p3"]:
		c.append(l.count(a))
	return c

def countInts(l):
	r = []
	if len(l) == 0:
		return {}
	for s in l:
		s = str(s).split("+")
		#print(s)
		for i in s:
			if "*" in i and i.count("p") == i.count("*")+1:
				r.append(i.strip())
	counts = {}
	for i in r:
		if i not in counts:
			counts[i] = 0
		counts[i] += 1
	return counts

def generateAAsPositions():
	aa = "ACDEFGHIKLMNPQRSTVWY"
	pos = ["p0","p1", "p2", "p3"]
	l = []
	for p in pos:
		for a in aa:
			l.append(p+"_"+a)
	return l


def main(args):
	symIntDB = pickle.load(open("results/elife/catIntPlotdata.pickle", "rb"))
	#print(symIntDB)
	x = []
	ya = []
	yi = []
	AApos = generateAAsPositions()
	ht = np.zeros((len(symIntDB.keys()), len(AApos))) * np.nan
	for c, T in enumerate([("p0","p1"),("p0","p2"),("p0","p3"),("p1","p2"),("p1","p3"),("p2","p3")]):
		t = (T, symIntDB[T])
		print(t[0], t[1])
		for i in t[1].items():
			if str(i[0]) in AApos:
				ht[c, AApos.index(str(i[0]))] = i[1]
	

	#plt.subplots(layout="constrained")
	#plt.figure(figsize=(10,6))
	plt.rcParams.update(plt.rcParamsDefault)
	fig, axs = plt.subplots(figsize=(17,4),nrows=1, ncols=1, layout="constrained")
	im1 = axs.matshow(ht, cmap="seismic", vmin=-3e-4, vmax=3e-4)
	axs.set_yticks([0,1,2,3,4,5], [("p0","p1"),("p0","p2"),("p0","p3"),("p1","p2"),("p1","p3"),("p2","p3")])
	axs.set_xticks(range(0,len(AApos)),AApos, rotation=45)
	cb1 = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.005, pad=0.01)
	axs.set_xlabel("Aminoacids at each epistatic position")
	axs.set_ylabel("Second derivatives pairs")
	plt.savefig(os.path.join(OUTDIR, "analyticalDerivativesAA.png"), dpi=400)
	plt.show()

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
