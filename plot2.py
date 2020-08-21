import matplotlib.pyplot as plt
import numpy as np

 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [4.34628116395,20.2935066342, 4.91764410734]
bars2 = [5.69397014626,  28.5061495733 , 5.50142632484]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='KMP based',yerr = [3.71023907727, 16.9987315186 ,4.48730113657])
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='cut off', yerr = [4.50065840614,30.4456094384,5.12704743784])
 
# Add xticks on the middle of the group bars
#plt.xlabel('Datasets with patter 1 - 6 instances', fontweight='bold')
plt.ylabel('execution time in sec', fontweight='bold')
plt.xticks([(r-0.15) + barWidth for r in range(len(bars1))], ['A', 'B', 'C'])
 
# Create legend & Show graphic
plt.legend()
#plt.show()
plt.plot()
plt.savefig('/home/fatemeh/Bureau/Stage/figures/pattern2_noleg.png',dpi = 300)

