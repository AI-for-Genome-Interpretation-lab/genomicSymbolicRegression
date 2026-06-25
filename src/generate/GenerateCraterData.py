#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  GeneratePhenoDatasetsSimple.py
#  
#  Copyright 2023 eddiewrc <eddiewrc@alnilam>
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
import os, sys
import numpy as np

def main(args):
	NUMSAMPLES = 5000
	NUMSNPS = 5000
	snps = np.random.randint(0,3,(NUMSNPS, NUMSAMPLES))
	ofp = open("synthCraterSNPs.csv", "w")
	i = 0
	ofp.write("\"\"")
	while i < snps.shape[1]:
		ofp.write(",\"sample_%d\""%(i+1))
		i+=1
	ofp.write("\n")
	i = 0
	while i < snps.shape[0]:
		j = 0
		ofp.write("\"SNP_%d\""%(i+1))
		while j < snps.shape[1]:
			ofp.write(",%d"%(snps[i,j]))
			j+=1
		ofp.write("\n")
		i+=1
	ofp.close()

	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
