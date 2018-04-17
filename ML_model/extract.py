import numpy as np
import matplotlib.pyplot as plt
import time

def extract(location):
	print 'Extract Data from ', location
	pulseFile = open(location, 'r')
	pulse = [line.split(',')
		for line in pulseFile.read().split('\r\n')[:-1]]
	Ts = 1.0/len(pulse[0])
	pulseFile.close()
	pulse = np.array(pulse)
	time, pulse = pulse[:,0], pulse[:,1:]
	pulse = np.array([[np.float64(i) for i in row] for row in pulse])
	time = [np.array(pt.split(':')).astype(float) for pt in time]
	time = [pt[0]*60*60+pt[1]*60+pt[2] for pt in time]
	time = np.array(time)-time[0]
	pulse = pulse.reshape(1,-1)[0,:]
	return time, pulse, Ts





# Instantaneous Frequency
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert

def insFreq(Ts, signal, method='hilbert'):
	if method == 'hilbert':
		anal = hilbert(signal)
		amp, phase = np.abs(anal), np.angle(anal)
		freq = np.concatenate((np.array([0]),np.diff(np.abs(phase))))/Ts
		return freq/np.pi/2
	if method == 'DQ':
		t = np.array(range(len(signal)))*Ts
		amp = np.abs(signal)
		upper = upperEnv(t, amp)
		norm = signal/upper
		phase = np.arctan(np.sqrt(1-norm**2)/norm)
		for i,term in enumerate(phase):
			if term != term:
				if i==0: phase[i] = 0
				else: phase[i] = phase[i-1]
		freq = np.concatenate(
			(np.array([0]),
			np.diff(np.abs(phase))/Ts
		))/np.pi/2
		return freq
def upperEnv(t, signal):
	def maximum(i,signal):
		if i==0 or i==len(signal)-1: return True
		return signal[i]>signal[i-1] and signal[i]>signal[i+1]
	upper = []
	upperT = []
	for i in xrange(len(signal)):
		if maximum(i,signal):
			upper.append(signal[i])
			upperT.append(t[i])
	upper = CubicSpline(upperT,upper)(t)
	return upper
	
def lowerEnv(t, signal):
	def minimum(i,signal):
		if i==0 or i==len(signal)-1: return True
		return signal[i]<signal[i-1] and signal[i]<signal[i+1]
	lower = []
	lowerT = []
	for i in xrange(len(signal)):
		if minimum(i,signal):
			lower.append(signal[i])
			lowerT.append(t[i])
	lower = CubicSpline(lowerT,lower)(t)
	return lower

def clamp(t, signal):
	upper = upperEnv(t,signal)
	lower = lowerEnv(t,signal)
	avg = upper+lower/2
	return signal-avg

def IMF(t, signal):
	for i in xrange(4):
		nxt = clamp(t,signal)
		#SD = np.sum(((signal-nxt)**2)/(signal**2))
		#if SD<0.25: break
		signal = nxt
	return signal

def EMD(t, signal):
	imfs = []
	for i in xrange(20):
		slope = np.diff(signal)
		if np.all(slope>0) or np.all(slope<0): break
		nxt = IMF(t,signal)
		imfs.append(nxt)
		signal -= nxt
	imfs.append(signal)
	return np.array(imfs)
			


#xR = np.array(range(int(10/Ts)))*Ts
#y = np.cos(2*np.pi*200*xR)
#plt.plot(xR,y)
#plt.plot(xR,insFreq(Ts,y,method = 'DQ'))
#print insFreq(Ts,y,method = 'DQ')
#plt.show()



def package(x,pulse, Ts, mode, emotion, window = 10.0):
	windowNum = int(window/Ts+0.5)
	print 'wNum',windowNum
	mat = []
	tar = []
	if(mode == 'hilbert' or mode=='DQ'):
		print 'Down Sampling...'
		dsNum = 20
		Ts,pulse = dSample(dsNum,Ts,pulse)
		print 'Energy Analyzing...'
		for i,sig in enumerate(pulse):
			t = np.array(range(len(sig)))*Ts
			imfList = EMD(t,sig)
			eList = lEnergy(Ts,imfList)
			mainIdx = np.argsort(eList[:-1])[-1:]
			pulse[i] = np.sum(imfList[mainIdx], axis = 0)

		print 'Instanatneous Frequency Computing...'
		
		for i in xrange(len(pulse)):
			term = insFreq(Ts,pulse[i], method = mode)
			term = term[:len(term)/windowNum*windowNum]
			modNorm = len(term)/windowNum
			for j in xrange(modNorm):
				k = term[j::modNorm]
				mat.append(k)
				tar.append(emotion)

		return np.array(mat),np.array(tar)

	elif mode=='fourier':
		for i in xrange(len(pulse)/windowNum):
			periodFreq = np.fft.fft(pulse[i*windowNum:(i+1)*windowNum])
			mat.append(periodFreq)
			tar.append(emotion)
		return np.abs(np.array(mat)),np.array(tar)
	elif mode=='naive':
		for i in xrange(len(pulse)/windowNum):
			period = pulse[i*windowNum:(i+1)*windowNum]
			period = np.array(period)
			tar.append(emotion)
			mat.append(period)
		return np.array(mat), np.array(tar)
	else: print 'error'

def dSample(r,Ts,signal):
	result = []
	n = len(signal)/r*r
	for i in xrange(r):
		result.append(signal[i:n:r])
	return Ts*r,np.array(result)

def lEnergy(Ts,signals):
	def energy(Ts,signal):
		return np.sum(signal**2)*Ts
	return np.array([energy(Ts,comp) for comp in signals])

'''
time, pulse, Ts = extract('sub018_baseline_ECG.csv')
#Configuration
dsNum = 20
Ts,pulse = dSample(dsNum,Ts,pulse)
for sig in pulse:
	t = np.array(range(len(sig)))*Ts
	imfList = EMD(t,sig)
	eList = lEnergy(Ts,imfList)
	mainIdx = np.argsort(eList[:-1])[-1:]
	sig = np.sum(imfList[mainIdx], axis = 0)
	sigIF = insFreq(Ts,sig)
	plt.plot(t,np.abs(np.fft.fft(sig)))
	plt.show()
'''

#print Ts
#N = len(pulse)
#x = np.array(range(N))*Ts
#freq = insFreq(Ts,IMF(x,pulse),method = 'hilbert')
#w = np.array(range(N))/(N*Ts)
#plt.plot(w[0:10000],np.abs(freq[0:10000]))
#plt.show()
#print 'File Output...'
#np.savetxt( 'test.csv',package(x, pulse, Ts, 'DQ',1) )


'''
freq = np.fft.fft(pulse)
plt.figure(ampFig.number)
print np.array(range(N))/(N*Ts)
plt.plot(np.array(range(N))/(N*Ts),np.absolute(freq))
plt.show()
'''
