#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mulBarchart.py
#  
#  Copyright 2025 eddiewrc <eddiewrc@alnilam>
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


import os
# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)


def main(args):
	import matplotlib.pyplot as plt

	# Data
	data = [
		(('p2', 'p3'), 26),
		(('p0', 'p3'), 26),
		(('p0', 'p2'), 24),
		(('p1', 'p3'), 3),
		(('p0', 'p1'), 2)
	]

	# Extract labels and values
	labels = [', '.join(pair) for pair, _ in data]
	values = [value for _, value in data]

	# Plot
	plt.figure(figsize=(5, 5))
	plt.bar(labels, values)

	# Add labels and title
	plt.xlabel("Variable Pairs", fontsize=12)
	plt.ylabel("Counts", fontsize=12)
	plt.title("Number of times loci are joined\nby MUL operator", fontsize=14)
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTDIR, "mulcounts.png"), dpi=400)
	# Show plot
	plt.show()
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
