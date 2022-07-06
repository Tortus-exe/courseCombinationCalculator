"""
8:00	0
8:30	1
9:00	2
9:30	3
10:00	4
10:30	5
11:00	6
11:30	7
12:00	8
12:30	9
13:00	10
13:30	11
14:00	12
14:30	13
15:00	14
15:30	15
16:00	16
16:30	17
17:00	18
17:30	19
18:00	20
18:30	21
19:00	22
19:30	23
"""

import numpy as np
import itertools
import functools
import statistics
from copy import deepcopy
import re

DEVWEIGHT = 5	# the larger, the worse
FREEWEIGHT = -5	# the more, the better
RLEWEIGHT = 4	# the longer, the worse
BREAKWEIGHT = 50 # the more, the worse
PENALIZEDBREAKSWEIGHT = 90

PENALIZEDBREAKLENGTHS = [2,3,4,5]
					#	 T1					  T2
#COURSE DICTS: FORMAT -> [M] [T] [W] [TH] [F] [M] [T] [W] [TH] [F]
STTindexDict = {
#	"COAL": [[4,5],[9,10,11,15,16,17,18],[4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[4,5,16,17,18,19],[],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17,18,19,20,21],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5]],
#	"COAT": [[4,5],[3,4,5,6,9,10,11,15,16,17,18],[4,5,14,15,16,17],[15,16,17,18,19,20,21,22],[4,5,16,17,18,19],[12,13,14,15],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17,18,19,20,21],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COCA": [[],[3,4,5,6,9,10,11,12,13,14,15,16,17,18],[],[6,7,8,9,12,13,14,15,16,17,18,19,20,21,22],[16,17,18,19],[],[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18],[4,5,6,7,14,15,16,17,18,19,20,21],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
#	"CODE": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[16,17,18,19],[12,13,14,15],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17,18,19,20,21],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
#	"COED": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[16,17,18,19],[],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17,18,19,20,21],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COLA": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5],[6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22],[16,17,18,19],[12,13,14,15],[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18],[14,15,16,17,18,19,20,21],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
#	"COPS": [[4,5],[9,10,11,15,16,17,18],[4,5],[6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22],[4,5,16,17,18,19],[12,13,14,15],[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18],[14,15,16,17,18,19,20,21],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
#	"COPY": [[],[3,4,5,6,9,10,11,12,13,14,15,16,17,18],[14,15,16,17],[12,13,14,15,16,17,18,19,20,21,22],[16,17,18,19],[12,13,14,15],[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18],[14,15,16,17,18,19,20,21],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
	"COZY": [[3,4,5],[3,4,5,6,9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[15,16,17,18,19,20,21,22],[16,17,18,19],[],[0,1,2,3,4,5,12,13,14,15,16,17,18],[4,5,6,7,14,15,16,17,18,19,20,21],[0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18],[]]
}
""" 
#exclude half-times
STTindexDict = {
	"COAL": [[4,5],[9,10,11,15,16,17,18],[4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[4,5],[],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COAT": [[4,5],[3,4,5,6,9,10,11,15,16,17,18],[4,5,14,15,16,17],[15,16,17,18,19,20,21,22],[4,5],[12,13,14,15],[12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COCA": [[],[3,4,5,6,9,10,11,12,13,14,15,16,17,18],[],[6,7,8,9,12,13,14,15,16,17,18,19,20,21,22],[],[],[0,1,2,3,4,5,12,13,14,15,16,17,18],[4,5,6,7,14,15,16,17],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
	"CODE": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[],[12,13,14,15],[12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COED": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[11,12,13,14,15,16,17,18,19,20,21,22],[],[],[6,7,8,9,12,13,14,15,16,17,18],[0,1,2,3,4,5,14,15,16,17],[12,13,14,15,16,17,18],[0,1,2,3,4,5]],
	"COLA": [[3,4,5],[9,10,11,15,16,17,18],[3,4,5],[6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22],[],[12,13,14,15],[0,1,2,3,4,5,12,13,14,15,16,17,18],[14,15,16,17],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
	"COPS": [[4,5],[9,10,11,15,16,17,18],[4,5],[6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22],[4,5],[12,13,14,15],[0,1,2,3,4,5,12,13,14,15,16,17,18],[14,15,16,17],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
	"COPY": [[],[3,4,5,6,9,10,11,12,13,14,15,16,17,18],[14,15,16,17],[12,13,14,15,16,17,18,19,20,21,22],[],[12,13,14,15],[0,1,2,3,4,5,12,13,14,15,16,17,18],[14,15,16,17],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]],
	"COZY": [[3,4,5],[3,4,5,6,9,10,11,15,16,17,18],[3,4,5,14,15,16,17],[15,16,17,18,19,20,21,22],[],[],[0,1,2,3,4,5,12,13,14,15,16,17,18],[4,5,6,7,14,15,16,17],[0,1,2,3,4,5,12,13,14,15,16,17,18],[]]
}
"""
M220indexDict = {
#	"M220_101": [[8,9],[],[8,9],[],[8,9],[],[],[],[],[]],
#	"M220_103": [[4,5],[],[4,5],[],[4,5],[],[],[],[],[]],
	"M220_104": [[14,15],[],[14,15],[],[14,15],[],[],[],[],[]],
	"M220_105": [[12,13],[],[12,13],[],[12,13],[],[],[],[],[]],
#	"M220_107": [[],[3,4,5],[],[3,4,5],[],[],[],[],[],[]],
#	"M220_108": [[],[9,10,11],[],[9,10,11],[],[],[],[],[],[]],
	#M220_201 = []
#	"M220_203": [[],[],[],[],[],[10,11],[],[10,11],[],[10,11]],
#	"M220_204": [[],[],[],[],[],[4,5],[],[4,5],[],[4,5]]
}
M253indexDict = {
	"M253": [[6,7],[],[6,7],[],[6,7],[],[],[],[],[]]
}
M256indexDict = {
	"M256_101": [[8,9],[],[8,9],[],[8,9],[],[],[],[],[]],
#	"M256_102": [[8,9],[],[8,9],[],[8,9],[],[],[],[],[]],
#	"M256_103": [[8,9],[],[8,9],[],[8,9],[],[],[],[],[]],
	"M256_201": [[],[],[],[],[],[],[12,13,14],[],[12,13,14],[]],
	"M256_202": [[],[],[],[],[],[10,11],[],[10,11],[],[10,11]],
	"M256_203": [[],[],[],[],[],[2,3],[],[2,3],[],[2,3]]
}
CPSC221indexDict = {
#	"CPSC221_101": [[12,13],[],[12,13],[],[12,13],[],[],[],[],[]],
#	"CPSC221_102": [[12,13],[],[12,13],[],[12,13],[],[],[],[],[]],
	"CPSC221_201": [[],[],[],[],[],[10,11],[],[10,11],[],[10,11]],
	"CPSC221_202": [[],[],[],[],[],[16,17],[],[16,17],[],[16,17]],
#	"CPSC221_203": [[],[],[],[],[],[8,9],[],[8,9],[],[8,9]]
}
CPSC221LBindexDict = {
#	"CPSC221_L1A": [[],[6,7,8,9],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1B": [[],[],[12,13,14,15],[],[],[],[],[],[],[]],
#	"CPSC221_L1C": [[18,19,20,21],[],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1E": [[],[20,21,22,23],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1F": [[],[16,17,18,19],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1H": [[],[],[18,19,20,21],[],[],[],[],[],[],[]],
#	"CPSC221_L1J": [[2,3,4,5],[],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1P": [[2,3,4,5],[],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1K": [[],[],[2,3,4,5],[],[],[],[],[],[],[]],
#	"CPSC221_L1N": [[],[2,3,4,5],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1R": [[],[12,13,14,15],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1S": [[],[16,17,18,19],[],[],[],[],[],[],[],[]],
#	"CPSC221_L1T": [[],[],[18,19,20,21],[],[],[],[],[],[],[]],
#	"CPSC221_L2A": [[],[],[],[],[],[],[10,11,12,13],[],[],[]],
	"CPSC221_L2B": [[],[],[],[],[],[],[],[12,13,14,15],[],[]],
#	"CPSC221_L2J": [[],[],[],[],[],[],[],[12,13,14,15],[],[]],
#	"CPSC221_L2C": [[],[],[],[],[],[],[14,15,16,17],[],[],[]],
#	"CPSC221_L2D": [[],[],[],[],[],[4,5,6,7],[],[],[],[]],
#	"CPSC221_L2E": [[],[],[],[],[],[12,13,14,15],[],[],[],[]],
#	"CPSC221_L2K": [[],[],[],[],[],[12,13,14,15],[],[],[],[]],
#	"CPSC221_L2F": [[],[],[],[],[],[],[],[4,5,6,7],[],[]],
#	"CPSC221_L2G": [[],[],[],[],[],[],[6,7,8,9],[],[],[]],
#	"CPSC221_L2H": [[],[],[],[],[],[],[],[],[2,3,4,5],[]],
	"CPSC221_L2L": [[],[],[],[],[],[],[2,3,4,5],[],[],[]],
#	"CPSC221_L2Q": [[],[],[],[],[],[],[2,3,4,5],[],[],[]],
#	"CPSC221_L2M": [[],[],[],[],[],[],[14,15,16,17],[],[],[]],
	"CPSC221_L2P": [[],[],[],[],[],[],[],[18,19,20,21],[],[]],
	"CPSC221_L2R": [[],[],[],[],[],[],[18,19,20,21],[],[],[]],
#	"CPSC221_L2S": [[],[],[],[],[],[18,19,20,21],[],[],[],[]],
	"CPSC221_L2T": [[],[],[],[],[],[],[],[],[2,3,4,5],[]],
	"CPSC221_L2U": [[],[],[],[],[],[],[],[],[18,19,20,21],[]],
#	"CPSC221_L2V": [[],[],[],[],[],[],[],[],[6,7,8,9],[]],
	"CPSC221_L2Y": [[],[],[],[],[],[],[],[],[14,15,16,17],[]],
	"CPSC221_L2Z": [[],[],[],[],[],[],[],[],[18,19,20,21],[]]
}
CPSCwhitelistRegexMatch = (
	"COAL:.*:M256_202:CPSC221_203:CPSC221_L2[DEHKLQST]|"
	"COAT:.*:M256_202:CPSC221_203:CPSC221_L2[DEHKLQST]|"
	"COCA:.*:M256_203:CPSC221_201:CPSC221_L2[DEKSV]|"
	"CODE:.*:M256_202:CPSC221_203:CPSC221_L2[DEHKLQST]|"
	"COED:.*:M256_202:CPSC221_203:CPSC221_L2[DEHKLQST]|"
	"COLA:.*:M256_203:CPSC221_201:CPSC221_L2[DEFKS]|"
	"COPS:.*:M256_203:CPSC221_201:CPSC221_L2[DEFKS]|"
	"COPY:.*:M256_203:CPSC221_201:CPSC221_L2[DEFKS]|"
	"COZY:.*:M256_203:CPSC221_201:CPSC221_L2[DEFKS]")
# ARRAY OF ALL DICTS TO CONSIDER
AllDicts = [STTindexDict, M220indexDict, M253indexDict, M256indexDict, CPSC221indexDict]




#HELPER FUNCTIONS
def genFromIndex(index):
	#turn an array of times into a timetable
	return [[int(a in index[x]) for a in base[x]] for x in range(len(base))]
def orArray(array1, array2):
	#∨/
	return [[int(array1[x][a] or array2[x][a]) for a in range(len(array1[x]))] for x in range(len(array1))]
def andArray(array1, array2):
	#∧/
	return [[int(array1[x][a] and array2[x][a]) for a in range(len(array1[x]))] for x in range(len(array1))]
def sumReduce(array):
	#+/
	return [sum(x) for x in array]
def printFormatted(array):
	for x in range(len(array[0])):
		string = ""
		for a in range(len(array)):
			string += array[a][x] * "#####" + (not array[a][x]) * "_____"
		print(string)
	return
def sumLong(arrayOfArrays):
	#+/each array in arrayOfArrays
	return np.sum(arrayOfArrays, axis=0).tolist()
def getRunLengthRanking(timeTable):
	#return the sum of the RLEs of each day in the timetable
	return sum(list(map(lambda x: len([(k, sum(1 for i in g)) for k,g in itertools.groupby(x)]), timeTable)))
def getAvgLengthBreaks(timeTable):
	rle = list(map(lambda x: [(k, sum(1 for i in g)) for k,g in itertools.groupby(x)], timeTable))
	for day in rle:
		if(day[-1][0] == 0):
			day.pop()
		if(len(day) == 0):
			continue
		if(day[0][0] == 0):
			day.pop(0)
	return (sum([sum([run[1] for run in day if run[0] == 0]) for day in rle]), sum([sum([1 for run in day if (run[0] == 0 and run[1] in PENALIZEDBREAKLENGTHS)]) for day in rle]))
def score(deviance, freeDays, RLE, lengthOfBreaks, numPenalizedBreaks):
	return sum([DEVWEIGHT*deviance, FREEWEIGHT*freeDays, RLEWEIGHT*RLE, BREAKWEIGHT*lengthOfBreaks, PENALIZEDBREAKSWEIGHT*numPenalizedBreaks])
base = [[i for i in range(24)] for _ in range(10)]
IDArray = [[0 for _ in range(24)] for _ in range(10)]

#MAINLOOP

AllDictsAsTuple = []
for x in AllDicts:
	currentArray = []
	for k,v in x.items():
		currentArray.append( (k, genFromIndex(v)) )
	AllDictsAsTuple.append(deepcopy(currentArray))

statsDict = {}
schedDict = {}
counter = 0
for combination in itertools.product(*AllDictsAsTuple):
	names = [a[0] for a in combination]
	timetables = [a[1] for a in combination]
	if not any([(max(x)>1) for x in (sumLong(timetables))]):
		#arrays are compatible if here
		#print(list(STTindexDict.keys())[s] + " and " + list(M220indexDict.keys())[m] + " are compatible!")
		combined = functools.reduce(orArray, timetables, IDArray)
		deviation = statistics.stdev(sumReduce([x for x in combined if x!=0]))
		numFreeDays = len([x for x in sumReduce(combined) if x == 0])
		runLengthRanking = getRunLengthRanking(combined)
		lengthOfBreaksInBetween = getAvgLengthBreaks(combined)[0]
		numBreaksInPenalizedBreakLengths = getAvgLengthBreaks(combined)[1]
		statsDict[":".join(names)] = [deviation, numFreeDays, runLengthRanking, lengthOfBreaksInBetween, numBreaksInPenalizedBreakLengths]
		schedDict[":".join(names)] = combined
	counter += 1
	print("counter: " + str(counter), end='\r')

maxDev = max([x[0] for x in list(statsDict.values())])
maxFree = max([x[1] for x in list(statsDict.values())])
maxRLE = max([x[2] for x in list(statsDict.values())])
maxLengthOfBreaks = max([x[3] for x in list(statsDict.values())])
maxNumPenalizedBreakLengths = max([x[4] for x in list(statsDict.values())])
if(maxFree == 0): 
	maxFree = 1


finalDict = {}
for k,v in statsDict.items():
	#if(re.compile(CPSCwhitelistRegexMatch).match(k)):
	finalDict[k] = score(v[0]/maxDev, v[1]/maxFree, v[2]/maxRLE, v[3]/maxLengthOfBreaks, v[4]/maxNumPenalizedBreakLengths)
#lower = better -> sort forwards so the larger items appear last
sortedFinal = sorted(finalDict.items(), key=lambda item: item[1])[0:10]

for item in sortedFinal:
	print(item[0])
	printFormatted(schedDict[item[0]])
	print("")