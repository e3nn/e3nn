import os

# simple logger class that creates a log file and mirrors all inputs to write to the file and stdout
class logger(object):
	def __init__(self, basepath, timestamp):
		os.makedirs('{:s}/logs'.format(basepath), exist_ok=True)
		self.logfile = '{:s}/logs/{:s}.log'.format(basepath, timestamp)

	def write(self, string, print_bool=True):
		''' append string to log file and optionally print out '''
		if print_bool:
			print(string)
		with open(self.logfile, 'a') as f:
			f.write('\n'+string)

	# def prepend(self, string): # for convenience to prepend final results 
	# 	with open(self.logfile, 'r') as original:
	# 		data = original.read()
	# 	with open(self.logfile, 'w') as modified:
	# 		modified.write(string + '\n' + data)