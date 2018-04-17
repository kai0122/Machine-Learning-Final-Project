from extract import *

# Traquil: 0
# Excited: 1
# Fear: 2

def matrify(fileName, target, mode):
	mat = []
	tarOut = []
	for i, location in enumerate(fileName):
		time,pulse,Ts = extract(location)
		N = len(pulse)
		x = np.array(range(N))*Ts
		fList,tList = package(x,pulse,Ts,mode,target[i],1.0)
		for i,feature in enumerate(fList):
			mat.append(feature)
			tarOut.append(tList[i])

	mat = np.array(mat)
	tarOut = np.array(tarOut)
	#mat = np.concatenate(mat)
	return mat, tarOut
'''
fileName = [
	'sub017_baseline_ECG.csv',
	'sub017_film 2_ECG.csv',
	'sub017_film 1_ECG.csv'
]
target = [0,1,2]

print matrify(fileName, target, 'naive')
'''
