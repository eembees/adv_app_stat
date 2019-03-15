#!/usr/local/bin/python3

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle
from scipy.special import erf

if __name__ == "__main__":
	
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Palatino']})

	RUNS = 100000
	
	TS01 = np.array(pickle.load(open('Nbins100_mubg1000.0_musig0.0.dat', "br" )))
	TS02 = np.array(pickle.load(open('Nbins100_mubg10.0_musig0.0.dat', "br" )))
	TS03 = np.array(pickle.load(open('Nbins100_mubg0.1_musig0.0.dat', "br" )))
	
	TS1 = np.array(pickle.load(open('Nbins100_mubg1000.0_musig100.0.dat', "br" )))
	TS2 = np.array(pickle.load(open('Nbins100_mubg1000.0_musig200.0.dat', "br" )))
	
	NTS = 200
	delta = 0.5

	fig = figure(figsize=(8, 6))
	ax = fig.add_subplot(1,1,1)
	
	title(r'simulation ($10^5$ samples)',fontsize=16,y=1.05)
	
	xlabel(r'test statistic $\lambda$',fontsize=16)
	ylabel(r'number of samples',fontsize=16)
	
	ax.tick_params(axis='both',which='both',direction='in')
	
	for tick in ax.xaxis.get_major_ticks() :
		tick.label.set_fontsize(16) 
   	
	for tick in ax.yaxis.get_major_ticks() :
		tick.label.set_fontsize(16)  

	xlim([0,25])
	ylim([1e-1,1e5])
	
	hist(TS01,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="red",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=0$ / $\mu^*_{\rm bg}=1000$')
	#hist(TS02,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="green",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=0$ / $\mu^*_{\rm bg}=10$')
	#hist(TS03,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="blue",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=0$ / $\mu^*_{\rm bg}=0.1$')
	
	TS = []
	chi2 = []
	for i in range(0,NTS) :
		TS.append((i+0.5)*delta)
		chi2.append(1*RUNS*(erf(np.sqrt((i+1)*delta/2.))-erf(np.sqrt((i)*delta/2.))))

	plot(TS,chi2,color="black",drawstyle='steps-mid',label=r'$\chi_1$ distribution')
	
	leg = plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.,fancybox=False,framealpha=0.0,frameon=True,numpoints=1, scatterpoints = 1,handlelength=1)
	for t in leg.get_texts() :
		t.set_fontsize(14)
	
	savefig("background1.pdf",bbox_inches = 'tight');
	close()
	
	fig = figure(figsize=(8, 6))
	ax = fig.add_subplot(1,1,1)
	
	title(r'simulation ($10^5$ samples)',fontsize=16,y=1.05)
	
	xlabel(r'test statistic $\lambda$',fontsize=16)
	ylabel(r'number of samples',fontsize=16)
	
	ax.tick_params(axis='both',which='both',direction='in')
	
	for tick in ax.xaxis.get_major_ticks() :
		tick.label.set_fontsize(16) 
   	
	for tick in ax.yaxis.get_major_ticks() :
		tick.label.set_fontsize(16)  

	xlim([0,100])
	ylim([1e-1,1e5])
	
	hist(TS01,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="red",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=0$ / $\mu^*_{\rm bg}=1000$')
	hist(TS1,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="green",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=100$ / $\mu^*_{\rm bg}=1000$')
	hist(TS2,bins=np.arange(0,NTS*delta,delta),log=True,alpha=0.8,color="blue",label=r'$N_{\rm bins}=100$ / $\mu_{\rm sig}=200$ / $\mu^*_{\rm bg}=1000$')
	
	TS = []
	chi2 = []
	for i in range(0,NTS) :
		TS.append((i+0.5)*delta)
		chi2.append(1*RUNS*(erf(np.sqrt((i+1)*delta/2.))-erf(np.sqrt((i)*delta/2.))))
	
	plot(TS,chi2,color="black",drawstyle='steps-mid',label=r'$\chi_1$ distribution')
	
	med1 = np.median(TS1)
	med2 = np.median(TS2)
	
	plot(np.array([med1,med1]),np.array([0.01,1e6]),color="black",linestyle="dotted")
	plot(np.array([med2,med2]),np.array([0.01,1e6]),color="black",linestyle="dotted")
	
	leg = plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.,fancybox=False,framealpha=0.0,frameon=True,numpoints=1, scatterpoints = 1,handlelength=1)
	for t in leg.get_texts() :
		t.set_fontsize(14)
	
	savefig("signal.pdf",bbox_inches = 'tight');
	close()

		
		