
import os, sys

def main(args):
	folder = args[1]
	for f in os.listdir(folder):
		if len(os.listdir(folder+"/"+f)) != 4:
			print("WRONG PHENO: "+ folder+f)


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
