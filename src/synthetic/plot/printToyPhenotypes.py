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
import os
import matplotlib.pyplot as plt

# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

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
		
def readSimulations(f):
	causativeNames, causative = readCausativeSNPs(f+"SNPs_causative.list.txt")
	phenotypes, corresp = readPhenotypes(f+"simulated_phenotype.csv")
	assert len(phenotypes) == len(corresp)
	return causative, phenotypes, corresp

def main(args):
	causative, phenotypes, corresp = readSimulations("data/synthetic/toyPhenotypes/numQTL 2 _Dfract 1 _EpiAddOv 1/")	
	print(causative)
	_, snpNames, SNPDATA = readSNPMatrix("data/synthetic/toy.csv")
	#_, snpNames, SNPDATA = readSNPMatrix("SNPs_filteredMAF0.csv")
	l1 = SNPDATA[:, causative[0]]
	l2 = SNPDATA[:, causative[1]]
	tmpmat = {}
	mat = np.zeros((3,3))
	assert len(l2) == len(l1) == len(phenotypes)
	for i, p in enumerate(phenotypes):
		if not (l1[i], l2[i]) in tmpmat:
			tmpmat[(l1[i], l2[i])] = []
		tmpmat[(l1[i], l2[i])].append(phenotypes[i])
	for i in tmpmat.items():
		mat[i[0][0], i[0][1]] = np.mean(i[1])

	samplesPheno = [[],[],[]]
	assert len(l2) == len(l1) == len(phenotypes)
	for i, p in enumerate(phenotypes):
		samplesPheno[0].append(l1[i])
		samplesPheno[1].append(l2[i])
		samplesPheno[2].append(phenotypes[i])
	plotPheno(mat)
		
	return 0

def mainPlot(args):
	c2 = 0
	fig, axs = plt.subplots(figsize=(20,8), nrows=3, ncols=6, layout="tight")
	dom = [1, 0, -1,]
	epi = [-1, 0, 1]

	for di, d in enumerate(dom):
		for ei, e in enumerate(epi):
			f = "data/synthetic/toyPhenotypes/numQTL 2 _Dfract %d _EpiAddOv %d/" % (d, e)
			print(f)
			ei = ei * 2
			axs[di,ei].set_title("Dominance: %d, Epistasis: %d"%(d,e))
			causative, phenotypes, corresp = readSimulations(f)
			_, snpNames, SNPDATA = readSNPMatrix("data/synthetic/toy.csv")
			l1 = SNPDATA[:, causative[0]]
			l2 = SNPDATA[:, causative[1]]
			tmpmat = {}
			mat = np.zeros((3,3))
			assert len(l2) == len(l1) == len(phenotypes)
			for i, p in enumerate(phenotypes):
				if not (l1[i], l2[i]) in tmpmat:
					tmpmat[(l1[i], l2[i])] = []
				tmpmat[(l1[i], l2[i])].append(phenotypes[i])
			for i in tmpmat.items():
				mat[i[0][0], i[0][1]] = np.mean(i[1])
			plotPheno(mat, axs, di, ei, fig)
			print(d, ei)
	plt.savefig(os.path.join(OUTDIR, "toyPhenotyps.png"), dpi=400)
	plt.show()


def plotPheno(mat, axs, c1, c2, fig, flip=True, colors = ["red", "green", "blue"], markers=["s","^","o"], locus1=["xx", "xX", "XX"], locus2=["yy", "yY","YY"]):#["YY", "yY", "yy"]):
	if flip:
		flip = np.array([[0,0,1],[0,1,0],[1,0,0]])
		locus2 = ["YY", "yY", "yy"]
		print(mat)
		mat = np.matmul(mat, flip)
	print(mat)
	#fig, axs = plt.subplots(figsize=(8,3),nrows=1, ncols=2, layout=None)
	for i in range(0, mat.shape[0]):
		axs[c1,c2+0].plot([0,1,2], mat[i,:].squeeze(), color=colors[i], label=locus2[i], marker=markers[i], alpha=0.7)
	axs[c1,c2+0].set_xticks([0,1,2], locus1)
	if c2 == 0:
		axs[c1,c2+0].set_ylabel("Phenotype value")
	axs[c1,c2+0].legend()
	axs[c1,c2+0].grid(alpha=0.4)
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
	i = 0
	while i < X.shape[0]:
		j = 0
		while j < Y.shape[1]:
			Z.append(mat[X[i,j], Y[i,j]])
			j+=1
		i+=1
	Z = np.array(Z).reshape(X.shape)
	#im2 = axs[1].imshow(mat)
	im2 = axs[c1,c2+1].contourf(x, y, Z, levels=20, origin="lower")
	axs[c1,c2+1].set_xticks([0,1,2], locus1[::-1])
	axs[c1,c2+1].set_yticks([2,1,0], locus2)
	axs[c1,c2+1].invert_xaxis()
	fig.colorbar(im2, ax=axs[c1,c2+1], aspect=50)
	#plt.show()
	
if __name__ == '__main__':
	import sys
	sys.exit(mainPlot(sys.argv))
