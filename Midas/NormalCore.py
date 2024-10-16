from numba import i4
from numba.core.types import string
from numba.experimental import jitclass
from numpy import inf, zeros
from numpy.random import randint

# region @jitclass
@jitclass({
	'nameAlg': string,
	'ts': i4,
	'row': i4,
	'col': i4,
	'param': i4[:],
	'current': i4[:],
	'total': i4[:],
})
# endregion
class NormalCore:
	'''
	TO DO: add .process_dataset() method [like the one present in Custom]
	TO DO: add .plot_hash_table() method [like the one present in Custom]
	'''
	
	def __init__(self, row: int, col: int):
		self.nameAlg = 'MIDAS'
		self.ts: int = 1
		self.row = row
		self.col = col
		self.param = randint(1, 1 << 16, 2 * row).astype(i4)
		self.current = zeros(row * col, i4)
		self.total = zeros(row * col, i4)

	@staticmethod
	def ChiSquaredTest(a: float, s: float, t: float) -> float:
		return 0 if s == 0 or t - 1 == 0 or t == 0 else pow((a - s / t) * t, 2) / (s * (t - 1))
		#return (0, 0) if s == 0 or t == 0 else (a, s/t)

	def Call(self, src: int, dst: int, ts: int) -> float:
		if self.ts < ts:
			#print(self.current) #SUPER BEWARE!!!! YOU CAN FREEZE JUPYTER IF YOU RUN THIS CODE ON A FULL DATASET!!!
			self.current *= 0
			self.ts = ts
		minCurrent = minTotal = inf
		for i in range(self.row):
			i = i * self.col + ((src + 347 * dst) * self.param[i] + self.param[i + self.row]) % self.col
			self.current[i] += 1
			self.total[i] += 1
			minCurrent = min(minCurrent, self.current[i])
			minTotal = min(minTotal, self.total[i])
		return self.ChiSquaredTest(minCurrent, minTotal, ts)
