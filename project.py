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
import matplotlib.colors as colors
import pandas as pd
import sklearn.cluster as cluster
import sys


class UserInfo:
    __colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

    def __init__(self, user_id):
        self.id = user_id
        self.calls_count = 0
        self.home = (0, 0)
        self.work = (0, 0)
        self.locations = {}
        self.common_towers = 0
        self.clusters = ()

    def __gaussian(self, c, p, var):
        d = (c[0] - p[0])**2 + (c[1] - p[1])**2
        return np.exp(-d/(2 * var)) / np.sqrt(2 * np.pi * var)

    def getProbability(self, x, y):
        p = 0
        for _, location in self.locations.items():
            p += self.__gaussian((location.x, location.y), (x, y), location.spread)
        return p

    def disp(self):
        my_colors = UserInfo.__colors[self.clusters[1]]

        fig, ax = plt.subplots()
        ax.set_title("towers for user %d" %self.id)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(list(self.common_towers['X coordinate ']), list(self.common_towers['Y coordinate']), c = my_colors)

        for cluster, location in self.locations.items():
            plt.scatter(location.x, location.y, c = self.__colors[cluster], marker='*')
            # circle = plt.Circle((location.x, location.y), location.spread, color=colors.to_rgba(self.__colors[cluster], 0.5))
            # ax.add_artist(circle)
        plt.text(self.home.x, self.home.y, "HOME", fontsize=10)
        plt.text(self.work.x, self.work.y, "WORK", fontsize=10)
        plt.show()
        fig.savefig('user%dLocations.png' %self.id)

    def __str__(self):
        l = []
        for i,locationI in self.locations.items():
            for j,locationJ in self.locations.items():
                if i <= j: continue
                l.append(
                    np.sqrt((locationI.x - locationJ.x)**2 + (locationI.y - locationJ.y)**2)
                )
        l = np.array(l)

        s = ""
        s += "=" * 30 + "\n"
        s += "> user %d : \n" % self.id
        s += "> total number of calls : %d\n" % self.calls_count
        s += "--------------------------------------------------\n"
        s += "> number of significant towers : %d\n" % self.common_towers.shape[0]
        s += "significant towers data :\n"
        s += "maximum distance between towers : %f meters\n" %l.max()
        s += "minimum distance between towers : %f meters\n" %l.min()
        s += str(self.common_towers) + "\n"
        s += "--------------------------------------------------\n"
        s += "user towers clusters : \n"
        for cluster, towers in self.clusters[0].items():
            s += "{} : {}\n".format(cluster, towers)
        s += "--------------------------------------------------\n"
        s += "> number of common location for user : %d\n" % len(self.locations)
        s += "> common locations for user : \n"
        for locationID, locationData in self.locations.items():
            s += "Location %d : \n" %locationID
            s += str(locationData) + "\n"
        s += "=" * 30 + "\n"
        return s


class Location:
    def __init__(self, x = 0, y = 0, s = 0, p = 0):
        self.x = x
        self.y = y
        self.spread = s
        self.probability = p

    def __str__(self):
        s = ""
        s += "coordinates : (%f, %f)\n" %(self.x, self.y)
        s += "location spread : %f\n" %self.spread
        s += "probability : %f\n" %self.probability
        return s


class Solver:
    def __init__(self):
        # the time ranges specified here are made as best estimates
        # work time was from 10AM to 4PM to make sure the user should be in work during this period
        self.__work_time_start = pd.Timestamp("10:00:00")
        self.__work_time_end = pd.Timestamp("16:00:00")

        # user should be at home until 7 AM  and should return home by 8 PM
        self.__home_time_start = pd.Timestamp("20:00:00")
        self.__home_time_end = pd.Timestamp("08:00:00")

        self.__weekDays = np.array(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday'])
        self.__colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'])

        # read towers data set
        self.__towers = pd.read_excel("towers.xlsx")

        # significance level
        self.alpha = 0.025

    def __is_work(self, x):
        return self.__work_time_start.time() < pd.Timestamp(x).time() < self.__work_time_end.time()

    def __is_home(self, x):
        return self.__home_time_start.time() < pd.Timestamp(x).time() or \
            pd.Timestamp(x).time() < self.__home_time_end.time()

    def get_common_places(self, clusters, towers):
        """""
        this function should return a list of common places the user is supposed to be in
        for each place we are supposed to provide 
        1 - (x, y) coordinates for this place
        2 - the spread of this place
        3 - how often does he visit this place
        :type clusters: dictionary
        :param calls : the calls of the user to be investigated as Dataframe with columns id, timestamp, tower_id
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

    def get_common_towers(self, calls):
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
        used_towers = calls.tower_id.value_counts() # get the towers used along with their freq
        used_towers = used_towers[used_towers > self.alpha * np.sum(used_towers)] # select only significant towers
        used_towers_locations = self.__towers.loc[self.__towers.ID.isin(used_towers.index)] # self towers with given id
        used_towers = pd.DataFrame({'ID': used_towers.index, 'freq': used_towers}) # turn used_towers into dataframe to make merging easeir
        return pd.merge(used_towers, used_towers_locations, on='ID')

    def get_location_clusters(self, common_towers):
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

    def get_HW(self, calls, usesinfo):
        if len(usesinfo.locations) == 1:
            return usesinfo.locations[0], Location(-1, -1, -1, 0)
        else:
            calls = calls[calls.tower_id.isin(usesinfo.common_towers.ID)]
            calls['isHOME'] = pd.Series(calls.timestamp.apply(self.__is_home) ,index=calls.index)
            calls['isWORK'] = pd.Series(calls.timestamp.apply(self.__is_work), index=calls.index)
            cluster_home = {}
            cluster_work = {}
            max_home = max_work = -1
            home_id = work_id = -1
            home_call_count = work_call_count = 0
            for cluster, towers in usesinfo.clusters[0].items():
                cluster_calls = calls[calls.tower_id.isin(towers)]
                cluster_home[cluster] = np.sum(cluster_calls.isHOME) / cluster_calls.shape[0]
                cluster_work[cluster] = np.sum(cluster_calls.isWORK) / cluster_calls.shape[0]
                print("H : %f, W :%f\n" %(cluster_home[cluster], cluster_work[cluster]))
                s = cluster_home[cluster] + cluster_work[cluster]
                cluster_home[cluster] = cluster_home[cluster] / s
                cluster_work[cluster] = cluster_work[cluster] / s
                if max_home < cluster_home[cluster] and home_call_count / cluster_calls.shape[0] < 1.5:
                    max_home, home_id, home_call_count = cluster_home[cluster], cluster, cluster_calls.shape[0]
                elif max_work < cluster_work[cluster] and work_call_count / cluster_calls.shape[0] < 1.5:
                    max_work, work_id, work_call_count = cluster_work[cluster], cluster, cluster_calls.shape[0]
            print(cluster_home)
            print(cluster_work)
            return usesinfo.locations[home_id], usesinfo.locations[work_id]

    def get_user_info(self, calls):
        """""
        :param calls: Object (DataFrame) of calls made by a certain user
        :return: user info extracted from user calls
        """""
        user_id = calls.user_id.unique()[0]
        user_info = UserInfo(user_id)
        user_info.calls_count = calls.shape[0]
        user_info.common_towers = self.get_common_towers(calls)
        user_info.clusters = self.get_location_clusters(user_info.common_towers)
        user_info.locations = self.get_common_places(user_info.clusters[0], user_info.common_towers)
        user_info.home, user_info.work = self.get_HW(calls, user_info)
        return user_info

if __name__ == '__main__':
    # sys.stdout = open("out", "w")
    # calls = pd.read_csv("MOBILE_CALLS.CSV")
    calls = pd.read_csv("highest30.csv")
    calls['timestamp'] = pd.to_datetime(calls['timestamp'], )

    user = calls.user_id.unique()[0]
    calls = calls[calls.user_id == user]
    a = Solver()
    info = a.get_user_info(calls)
    print(info)
    info.disp()
