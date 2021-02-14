import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from numpy import linalg as LA

# Reading in our csv file
df = pd.read_csv(sys.argv[1], header=None)

# Set k equal to the second command line argument, the basis of k-means functions.
k = int(sys.argv[2])

# Create an array that will correspond to df's index with it's k grouping (0-k)
kGrouping = [-1 for x in range(len(df))]

# Conversion of data from a pandas df to a numpy array, 1 line, pure magic
npa = df.to_numpy()


# This is python magic, 25 lines of c++ code in 1. it zips centersX and centersY
# to a tuple, which gets turned into a list which gets turned into an array.
centersX = df[0].sample(n = k) # this takes n samples from the x column which allows our initial points to stay within bounds
centersY = df[1].sample(n = k) # same, but with y values
centers = np.array(list(zip(centersX, centersY)))

# This is the kMeans function, when called it finds the norm of every point and every center using a for loop and norm from linalg
# it then uses arg min to return the index of the center that is closest to it. It then is saved as a complimentary (meaning every index i in
# kGrouping corresponds to the ith element of NPA, the numpy array of data) array so we can print into groups.
def kMeans(k, df):
    for i in range(len(df)):
        kGrouping[i] = np.argmin([LA.norm(npa[i]-centers[j]) for j in range(k)]) # magic python magic
        # print(kGrouping[i])
        # ^ That's a useful debugging statement, unlike the entirety of GDB...


# We call this first to get our groupings based off the random values we pulled as centers.
kMeans(k, df)

# In python, this is literally an array of colors (b is blue, g is green, etc.) we save it as an array because we will print
# to the plot by grouping. This allows it to be a for loop and we set the color = kGrouping = i
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
for i in range(k):
    # print(kGrouping)
    pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
    # print(pointsInGroup)
    # plt.show()
    plt.scatter(pointsInGroup[:,0], pointsInGroup[:,1], c=colors[i])
    # ^^^^^ This line will randomly through bugs, but if you run again it'll go away
    # it may take like 2-4 retrys, but it'll work ¯\_(ツ)_/¯

# for plotting
plt.scatter(centers[:, 0], centers[:, 1], c='k')
plt.show()


# for keeping track of initial centers and seeing how they change
print("These are the centers for first iteration: ")
print(centers)

# another critical aspect of kMeans is updating your centers based on the mean x and y elements, aka the center of the grouping
# we call this function in our for loop after every time we call kMeans to get our new centers.
def updateCenters(x): # x is the centers array, which you'd know if we declared our types in this commie language
    for i in range(k):
        pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
        x[i] = [np.mean(pointsInGroup[:, 0]), np.mean(pointsInGroup[:, 1])]

# debugging statement
# print(kGrouping)
# print(centers)


# I arbitrarily chose 100 iterations because I don't feel like waiting hours in between compilations.
# In reality, we would set a tolerance, then check the difference between the last array of centers and the new array of centers
# and wait for the change to be within our tolerance.

# You're doing 2 nested for loops here so O(n^2) is best case, don't go too crazy with csv size lol
for i in range(100):
    updateCenters(centers)
    kMeans(k, df)
    # print(kGrouping)
    if i < 5:
        print(centers)
    if i is 50:
        print("Halfway through iterations the centers are") # this is pretty useless, they center real quick
        print(centers)
    # plt.scatter(centers[:, 0], centers[:, 1], c='k')

# print(pointsInGroup)
print("New centers after 100 iterations")
print(centers)


# printing the final plot
for i in range(k):
    # print(kGrouping)
    pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
    # print(pointsInGroup)
    plt.scatter(pointsInGroup[:, 0], pointsInGroup[:, 1], c=colors[i])

plt.scatter(centers[:, 0], centers[:, 1], c='k')
plt.show()
