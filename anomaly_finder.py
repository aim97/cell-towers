import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


class AnomalyFinder:
	def __init__(self, calls):
		self.__calls = calls
		self.__calls['timestamp'] = pd.to_datetime(self.__calls['timestamp'], )
		self.__alpha = 0.05

	def find_freq_anomalies(self):
		"""""
		this function should point out the days during wish user had made strangely large number
		or small number of calls
		we compute the number of calls for each day
		we compute mean and std for the number of calls per day
		then we compute significance value for each day
		if the significance level is less than 0.05 then we consider that day as an anomaly
		if not it's considered as a normal value
		:return: two data frame 
		1 - data frame for accepted days
		2 - data frame for anomalous days 	
		"""""
		# user_id = self.__calls.user_id[0]
		day = self.__calls.timestamp.apply(lambda x: x.date())
		self.__calls['day'] = pd.Series(day, self.__calls.index)
		day_freq = self.__calls.day.value_counts()

		# compute normal distribution parameters for current user number of calls per day
		cut_limit = int(day_freq.shape[0] * 0.2)
		mean_calls_count = day_freq[cut_limit:-cut_limit].mean()
		std_calls_count = day_freq[cut_limit:-cut_limit].std()

		# compute significance level for all days
		significance_levels = stats.norm.sf(np.abs((day_freq.values - mean_calls_count) / std_calls_count))

		# create a data frame for the data for all days
		df = pd.DataFrame({
			'day': pd.Series(day_freq.index),
			'freq': pd.Series(day_freq.values),
			'sig_lvl': pd.Series(significance_levels)
		})
		# sort the data frame based on day
		df = df.sort_values(['day'], ascending=[1])

		# display
		fig, ax = plt.subplots()
		days = plt.bar(np.arange(1, df.shape[0] + 1), df.freq)
		colors = np.array(["r", "b"])[np.where(df.sig_lvl < self.__alpha, 0, 1)]
		for color, day in zip(colors, days):
			day.set_facecolor(color)
		ax.set_ylabel('number of calls')
		ax.set_xlabel('days')
		ax.set_title('user %d calls' % self.__calls.user_id.unique()[0])
		plt.show()
		fig.savefig("user %d bar chart.png" % self.__calls.user_id.unique()[0])
		print(df)
		return df[df.sig_lvl > 0.05], df[df.sig_lvl < 0.05]


if __name__ == '__main__':
	# sys.stdout = open("out", "w")
	# calls = pd.read_csv("MOBILE_CALLS.CSV")
	calls = pd.read_csv("highest30.csv")
	calls['timestamp'] = pd.to_datetime(calls['timestamp'], )

	user = calls.user_id.unique()[0]
	calls = calls[calls.user_id == user]
	a = AnomalyFinder(calls)
	a.find_freq_anomalies()
