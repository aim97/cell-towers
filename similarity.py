import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

##Read Input
user1_id = 1
user2_id = 25
user1_data = pd.DataFrame(pd.read_csv("UsersData/%d.csv"%user1_id, names = ['Time', 'ID']));
user2_data = pd.DataFrame(pd.read_csv("UsersData/%d.csv"%user2_id, names = ['Time', 'ID']));

##Convert DateTime into Time
user1_data['Time'] = pd.to_datetime(user1_data['Time'], format='%Y-%m-%d %H:%M:%S')
user1_data['Time'] = pd.Series([val.hour for val in user1_data['Time']])

user2_data['Time'] = pd.to_datetime(user2_data['Time'], format='%Y-%m-%d %H:%M:%S')
user2_data['Time'] = pd.Series([val.hour for val in user2_data['Time']])


ar = np.zeros(24);
user1_calls = user1_data.shape[0];
user2_calls = user2_data.shape[0];


##Plot the normalized Frequency
fig, axes = plt.subplots(2, 1)
axes[0].set_title("User - %d"%user1_id)
axes[1].set_title("User - %d"%user2_id)

user1_data = user1_data.groupby('Time').count()/user1_calls
user2_data = user2_data.groupby('Time').count()/user2_calls

user1_data = user1_data.reindex(np.arange(0, 24), fill_value = 0)
user2_data = user2_data.reindex(np.arange(0, 24), fill_value = 0)

user1_data.plot(kind = 'bar', ax = axes[0], label = 'User10')
user2_data.plot(kind = 'bar', ax = axes[1])

plt.show();

##Calculate the Correlation And print it

user1_data = np.array(user1_data).reshape(user1_data.shape[0])
user2_data = np.array(user2_data).reshape(user2_data.shape[0])

print("The Users %d, %d are %.1f%s similar."%(
    user1_id,
    user2_id,
    np.corrcoef(user1_data, user2_data)[0][1]*100,
    "%"));
