import matplotlib.pyplot as plt
import numpy as np


def make_diagram(values,errorValues):
    pyplot.bar(range(len(values)), values, color = 'skyblue')
    pyplot.errorbar(range(len(values)), values, yerr = errorValues,
        fmt = 'none', capsize = 5, ecolor = 'red', elinewidth = 2, capthick = 8)
    pyplot.show()

values = [ 2.12939059144, 5.18038106673]
errorValues = [0.316765488319, 1.14742375256]
#make_diagram(values,errorValues)

data = [[2.12939059144, 25., 50., 20.],
  [5.18038106673, 23., 51., 17.]]
 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [2.12939059144, 7.73409599066, 18.2390090823]
bars2 = [5.18038106673, 18.6004477906, 33.4215455985]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1',yerr = [0.316765488,1.51398837062 ,5.10537391075])
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2', yerr = [1.14742375256,5.12328967415,17.4016412777])
 
# Add xticks on the middle of the group bars
plt.xlabel('Datasets with patter 1 - 6 instances', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1)-1)], ['A', 'B', 'C'])
 
# Create legend & Show graphic
plt.legend()
plt.show()
