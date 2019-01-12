import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster as cluster
import sys


getDayName = lambda x: pd.Timestamp(x).day_name()


def getTotalSecs(x):
	x = pd.Timestamp(x)
	t = x.hour
	t = t * 60 + x.minute
	t = t * 60 + x.second
	return t


weekDays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday']
colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

t1 = pd.Timestamp("2013-01-13 23:59:59")
oneDay = lambda x: x < t1

if __name__ == '__main__':
	sys.stdout = open("out1", "w")
	calls = pd.read_csv("highest30.csv")
	calls['timestamp'] = pd.to_datetime(calls['timestamp'], )
	towers = pd.read_excel("towers.xlsx")

	users = calls.user_id.unique()
	calls = calls[np.array(calls.timestamp.apply(oneDay))]
	# add another column to the date frame for time in secs
	secs = calls.timestamp.apply(getTotalSecs)
	calls['secs'] = pd.Series(secs, calls.index)
	user_anomalies = {}
	for i in users:
		userCalls = calls[calls.user_id == i]
		user_anomalies[i] = {}

		clusters = cluster.DBSCAN(eps = 2000, min_samples=8).fit(
			np.array(userCalls.secs).reshape(-1, 1)
		).labels_
		user_anomalies[i] = userCalls[clusters == -1]

		yy = [1] * userCalls.shape[0]
		clusterColors = colors[clusters + 1]
		fig, ax = plt.subplots()
		ax.set_title("user %d calls" %i)
		ax.set_xlabel('calling time (hours)')
		ax.set_ylabel('y = 1')
		plt.scatter(userCalls.secs/3600.0, yy, c = clusterColors)
		plt.show()
		fig.savefig("user%d.png" %i);
