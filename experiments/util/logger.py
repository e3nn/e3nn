import os

# simple logger class that creates a log file and mirrors all inputs to write to the file and stdout
class logger(object):
	def __init__(self, experiment, network):
		path = 'networks/{:s}/{:s}/logs'.format(experiment, network)
		os.makedirs(path, exist_ok=True)
		from time import gmtime, strftime
		self.logfile = '{:s}/{:s}_{:s}.log'.format(path, network, strftime("%Y-%m-%d_%H:%M:%S", gmtime()))

	def write(self, string, print_bool=True):
		if print_bool:
			print(string)
		with open(self.logfile, 'a') as f:
			f.write('\n'+string)

	# def prepend(self, string): # for convenience to prepend final results 
	# 	with open(self.logfile, 'r') as original:
	# 		data = original.read()
	# 	with open(self.logfile, 'w') as modified:
	# 		modified.write(string + '\n' + data)