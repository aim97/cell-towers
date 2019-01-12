import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn.cluster as cluster
import sys

t1 = pd.Timestamp("09:00:00") # 9:00 AM time to start work
t2 = pd.Timestamp("18:00:00") # 6:00 PM end of work time
# this function returns if the call happened within work time or not
# expected input : string ex : "2013-01-07 00:00:00"
# expected output : True > within work time, False > not within work time
isWork = lambda x: pd.Timestamp(x).time() > t1.time() and pd.Timestamp(x).time() < t2


if __name__ == '__main__':
	sys.stdout = open("out", "w")
	calls = pd.read_csv("highest30.csv")
	towers = pd.read_excel("towers.xlsx")

	users = calls.user_id.unique()

	user_towers_freq = {}
	user_calls_count = {}
	user_towers_locations = {}
	user_significant_towers_freq = {}
	user_significant_towers_locations = {}
	location_clusters = {}
	user_common_locations = {}
	user_calls_count = calls.user_id.value_counts()

	for i in users:
		h = calls[calls.user_id == i].tower_id.value_counts()
		user_towers_freq[i] = h
		user_towers_locations = towers.loc[towers.ID.isin(h.index)]

		user_significant_towers_freq[i] = h[h > sum(h) * 0.025]
		user_significant_towers_locations[i] = towers.loc[towers.ID.isin(user_significant_towers_freq[i].index)]

		location_clusters[i] = cluster.DBSCAN(eps = 1000, min_samples=1).fit(
			user_significant_towers_locations[i][['X coordinate ','Y coordinate']]
		).labels_

		user_clusters = np.unique(location_clusters[i])
		user_common_locations[i] = []
		total_number_of_calls = user_calls_count[i]
		for j in user_clusters:
			cluster_towers = user_significant_towers_locations[i].iloc[user_clusters == j]
			# print(cluster_towers)
			ids = list(cluster_towers['ID'])
			cx = list(cluster_towers['X coordinate '])
			cy = list(cluster_towers['Y coordinate'])
			cluster_center = [0, 0]
			for k in range(len(ids)):
				# print(user_significant_towers_freq[i])
				tower_calls = user_significant_towers_freq[i][ids[k]]
				cluster_center[0] += (tower_calls / total_number_of_calls) * cx[k]
				cluster_center[1] += (tower_calls / total_number_of_calls) * cy[k]
			user_common_locations[i].append(cluster_center)

	print("========================================================")
	for i in users:
		print("> user %d : " %i)
		print("> total number of calls : %d" %user_calls_count[i])
		print("> total number of towers used : %d" %user_towers_freq[i].shape[0])
		print("> number of significant towers : %d" %user_significant_towers_freq[i].shape[0])
		print("> significant towers : \n{}".format(user_significant_towers_freq[i]))
		print("> clusters labels : {}".format(location_clusters[i]))
		print("> number of common location for user : %d" %len(user_common_locations[i]))
		print("> common locations for user : ")
		for j in user_common_locations[i]:
			print(j)
		print("--------------------------------------------------")

"""
user_towers_freq : map (id, tower_freq_array)
	for each user id in this map : we obtain a series of towers used by this user along with number 
	of calls done from each tower

user_towers_locations : map (id, towers locations)
	for each user id in in this map : we obtain a dataframe for location of towers this user used

user_significant_towers_freq : map (id, tower_freq_array)
	this is a subset of user_towers_freq containing only towers with frequency higher than 1% of user calls

user_significant_towers_locations : map (id, towers locations)
	this is a subset of user_towers_locations contianing only towers with frequency higher than 1% of user calls

"""