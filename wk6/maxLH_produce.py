#!/usr/local/bin/python3

import healpy as H
import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Palatino']})
	
	RUNS = 10000
	
	Nbins = 100
	mubg = 1000.0
	musig = 200.0 # 10.0
	
	filename = 'Nbins' + str(Nbins) + '_mubg' + str(mubg) + '_musig' + str(musig) + '.dat'
	
	Ntot = Nbins*mubg + musig
	Nsig = musig
	
	S = []
	B = []
	TS = []
	
	for i in range(0,Nbins) :
		B.append((Ntot-Nsig)/(1.*Nbins))
		if i==0 :
			S.append(Nsig)
		else :
			S.append(0.0)
			
	for run in range(0,RUNS) :
	
		print(run)
		
		bin = []
		totevents = 0
		for i in range(0,Nbins) :
			temp = np.random.poisson(lam=S[i]+B[i])
			bin.append(temp)
			totevents += temp
		
		temp = 0.0
		if bin[0] > 0.0 :
			temp += 2.*bin[0]*np.log((Nbins*1./(totevents*1.))*bin[0])
		if totevents-bin[0] > 0.0 :
			temp += 2.*(totevents-bin[0]*1.)*np.log((Nbins*1./(totevents*1.))*(totevents-bin[0])/(Nbins-1.0))
			
		TS.append(temp)
	
	pickle.dump(TS,open(filename, "bw" ))	
		