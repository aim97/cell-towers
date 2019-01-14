"""
for each user we should provide the following data 

1 - for the common places task
DONE 1.1 list of locations where the user is most likely to be in
        each location is represented as (x, y) coordinates for the user
        and a variance for how much certain i'm about this estimate
        and probability of being in that cluster

DONE 1.2 list of common towers used by this user
        more specifically towers that hold more than 1% of total number of calls.
        for each tower we should know his ID, (x, y) coordinates and number of calls done by this
        user on this tower
        (assuming 0.01 as the value used for significance test)

DONE 1.3 list of clusters for the locations
        for each cluster we should provide a list of Tower IDs in that cluster

DONE 1.4 a graph representing the clusters formed for the user

    1.5 a contour line graph for the presence of user in various locations using seaborn

2 - home and work for each user
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import pandas as pd
import sklearn.cluster as cluster


# import sys


class UserInfo:
	common_towers: pd.DataFrame
	__colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

	def __init__(self, user_id):
		self.id = user_id
		self.calls_count = 0
		self.home = Location()
		self.work = Location()
		self.locations = {}
		self.common_towers = 0
		self.clusters = ()

	@staticmethod
	def __gaussian(c, p, var):
		d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2
		return np.exp(-d / (2 * var)) / np.sqrt(2 * np.pi * var)

	def get_probability(self, x, y):
		p = 0
		for _, location in self.locations.items():
			p += self.__gaussian((location.x, location.y), (x, y), location.spread)
		return p

	def disp(self):
		my_colors = UserInfo.__colors[self.clusters[1]]

		fig, ax = plt.subplots()
		ax.set_title("towers for user %d" % self.id)
		ax.set_xlabel('X-axis')
		ax.set_ylabel('Y-axis')

		# plt.gca().set_aspect('equal', adjustable='box')
		plt.scatter(list(self.common_towers['X coordinate ']), list(self.common_towers['Y coordinate']), c=my_colors)

		for c, location in self.locations.items():
			plt.scatter(location.x, location.y, c=self.__colors[c], marker='*')
		# circle = plt.Circle((location.x, location.y), location.spread, color=colors.to_rgba(self.__colors[cluster], 0.5))
		# ax.add_artist(circle)
		plt.text(self.home.x, self.home.y, "HOME", fontsize=10)
		plt.text(self.work.x, self.work.y, "WORK", fontsize=10)
		plt.show()
		fig.savefig('user%dLocations.png' % self.id)

	def __str__(self):
		distances = []
		for i, locationI in self.locations.items():
			for j, locationJ in self.locations.items():
				if i <= j:
					continue
				distances.append(
					np.sqrt((locationI.x - locationJ.x) ** 2 + (locationI.y - locationJ.y) ** 2)
				)
		distances = np.array(distances)

		s = ""
		s += "=" * 30 + "\n"
		s += "> user %d : \n" % self.id
		s += "> total number of calls : %d\n" % self.calls_count
		s += "--------------------------------------------------\n"
		s += "> number of significant towers : %d\n" % self.common_towers.shape[0]
		s += "significant towers data :\n"
		s += "maximum distance between towers : %f meters\n" % distances.max()
		s += "minimum distance between towers : %f meters\n" % distances.min()
		s += str(self.common_towers) + "\n"
		s += "--------------------------------------------------\n"
		s += "user towers clusters : \n"
		for c, towers in self.clusters[0].items():
			s += "{} : {}\n".format(c, towers)
		s += "--------------------------------------------------\n"
		s += "> number of common location for user : %d\n" % len(self.locations)
		s += "> common locations for user : \n"
		for locationID, locationData in self.locations.items():
			s += "Location %d : \n" % locationID
			s += str(locationData) + "\n"
		s += "=" * 30 + "\n"
		return s


class Location:
	def __init__(self, x=0, y=0, s=0, p=0):
		self.x = x
		self.y = y
		self.spread = s
		self.probability = p

	def __str__(self):
		s = ""
		s += "coordinates : (%f, %f)\n" % (self.x, self.y)
		s += "location spread : %f\n" % self.spread
		s += "probability : %f\n" % self.probability
		return s


class Solver:
	# the time ranges specified here are made as best estimates
	# work time was from 10AM to 4PM to make sure the user should be in work during this period
	__work_time_start = pd.Timestamp("10:00:00")
	__work_time_end = pd.Timestamp("16:00:00")

	# user should be at home until 7 AM  and should return home by 8 PM
	__home_time_start = pd.Timestamp("20:00:00")
	__home_time_end = pd.Timestamp("08:00:00")

	__weekDays = np.array(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday'])
	__colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

	def __init__(self):
		# read towers data set
		self.__towers = pd.read_excel("towers.xlsx")

		# significance level
		self.alpha = 0.025

	@staticmethod
	def __is_work(x):
		return Solver.__work_time_start.time() < pd.Timestamp(x).time() < Solver.__work_time_end.time()

	@staticmethod
	def __is_home(x):
		return Solver.__home_time_start.time() < pd.Timestamp(x).time() or pd.Timestamp(
			x).time() < Solver.__home_time_end.time()

	@staticmethod
	def get_common_places(clusters, towers):
		"""""
		this function should return a list of common places the user is supposed to be in
		for each place we are supposed to provide 
		1 - (x, y) coordinates for this place
		2 - the spread of this place
		3 - how often does he visit this place
		:type clusters: dictionary
		:param clusters: Dictionary for clusters made by the clustering algorithm
		:param towers: DataFrame for significant cell towers used by this user 
		:return: list of locations
		"""""
		total_calls = np.sum(towers.freq)
		locations = {}
		for cluster_label, my_ids in clusters.items():
			locations[cluster_label] = Location()
			cluster_towers = towers[towers.ID.isin(my_ids)]

			xx = list(cluster_towers['X coordinate '])
			yy = list(cluster_towers['Y coordinate'])
			frq = list(cluster_towers['freq'])
			cluster_calls_count = np.sum(cluster_towers.freq)
			for i in range(len(xx)):
				locations[cluster_label].x += (frq[i] * 1.0 / cluster_calls_count) * xx[i]
				locations[cluster_label].y += (frq[i] * 1.0 / cluster_calls_count) * yy[i]
				locations[cluster_label].spread += total_calls / (2000.0 * frq[i])
			locations[cluster_label].spread = 1 / locations[cluster_label].spread
			locations[cluster_label].probability = cluster_calls_count / total_calls
		return locations

	def get_common_towers(self, user_calls):
		"""""
		:description: this function should return a dataframe for towers commonly used by this user
		:param: a DataFrame for calls made by one user with columns user_id, timestamp, tower_id
			user_id : the id of the user making the call
			timestamp : the time when this call took place
			tower_id : the id of the tower serving this call
		:return: a DataFrame for significant towers used by this user with columns
			ID : tower id
			'X coordinate ' : the x coordinate of this tower
			'Y coordinate' : the y coordinate of this tower
			freq : the number of times this user used this tower
		P.S. significant tower : is a tower serving number of calls above 1% of user calls 
		"""""
		used_towers = user_calls.tower_id.value_counts()  # get the towers used along with their freq
		used_towers = used_towers[used_towers > self.alpha * np.sum(used_towers)]  # select only significant towers
		used_towers_locations = self.__towers.loc[self.__towers.ID.isin(used_towers.index)]  # self towers with given id
		used_towers = pd.DataFrame(
			{'ID': used_towers.index, 'freq': used_towers})  # turn used_towers into dataframe to make merging easeir
		return pd.merge(used_towers, used_towers_locations, on='ID')

	@staticmethod
	def get_location_clusters(common_towers):
		"""""
		this function should return the clusters for locations of user
		:param: DataFrame for common towers used by this user
		:return: a dictionary with keys as clusters labels, and value of each key is a
			list containing IDs for towers of this cluster  
		"""""
		towers_clusters = cluster.DBSCAN(eps=1000, min_samples=1).fit(
			common_towers[['X coordinate ', 'Y coordinate']]
		).labels_
		t = towers_clusters.copy()
		towers_clusters = np.array(towers_clusters)
		clusters_labels = np.unique(towers_clusters)
		clusters = {}
		for label in clusters_labels:
			clusters[label] = list(common_towers.iloc[towers_clusters == label].ID)
		return clusters, t

	def get_hm(self, user_calls, user_info):
		if len(user_info.locations) == 1:
			return user_info.locations[0], Location(-1, -1, -1, 0)
		else:
			user_calls = user_calls[user_calls.tower_id.isin(user_info.common_towers.ID)]
			user_calls['isHOME'] = pd.Series(user_calls.timestamp.apply(self.__is_home), index=user_calls.index)
			user_calls['isWORK'] = pd.Series(user_calls.timestamp.apply(self.__is_work), index=user_calls.index)
			cluster_home = {}
			cluster_work = {}
			max_home = max_work = -1
			home_id = work_id = -1
			home_call_count = work_call_count = 0
			for c, towers in user_info.clusters[0].items():
				cluster_calls = user_calls[user_calls.tower_id.isin(towers)]
				cluster_home[c] = np.sum(cluster_calls.isHOME) / cluster_calls.shape[0]
				cluster_work[c] = np.sum(cluster_calls.isWORK) / cluster_calls.shape[0]
				print("Cluster %d > H : %f, W :%f\n" % (c, cluster_home[c], cluster_work[c]))
				s = cluster_home[c] + cluster_work[c]
				cluster_home[c] = cluster_home[c] / s
				cluster_work[c] = cluster_work[c] / s
				if max_home < cluster_home[c] and home_call_count / cluster_calls.shape[0] < 1.5:
					max_home, home_id, home_call_count = cluster_home[c], c, cluster_calls.shape[0]
				elif max_work < cluster_work[c] and work_call_count / cluster_calls.shape[0] < 1.5:
					max_work, work_id, work_call_count = cluster_work[c], c, cluster_calls.shape[0]
			return user_info.locations[home_id], user_info.locations[work_id]

	def get_user_info(self, user_calls):
		"""""
		:param user_calls: Object (DataFrame) of calls made by a certain user
		:return: user info extracted from user calls
		"""""
		user_id = user_calls.user_id.unique()[0]
		user_info = UserInfo(user_id)
		user_info.calls_count = user_calls.shape[0]
		user_info.common_towers = self.get_common_towers(user_calls)
		user_info.clusters = self.get_location_clusters(user_info.common_towers)
		user_info.locations = self.get_common_places(user_info.clusters[0], user_info.common_towers)
		user_info.home, user_info.work = self.get_hm(user_calls, user_info)
		return user_info


if __name__ == '__main__':
	# sys.stdout = open("out", "w")
	# calls = pd.read_csv("MOBILE_CALLS.CSV")
	calls = pd.read_csv("highest30.csv")
	calls['timestamp'] = pd.to_datetime(calls['timestamp'], )

	# user = calls.user_id.unique()[0]
	# calls = calls[calls.user_id == user]
	a = Solver()
	info = a.get_user_info(calls)
	print(info)
	info.disp()
