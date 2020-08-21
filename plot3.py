import matplotlib.pyplot as plt
import numpy as np

 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [17.0231158519, 93.7575270994, 20.9415031481]
bars2 = [18.116855967,  92.6099369577,  22.5121266103]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='KMP based',yerr = [13.594967958, 70.2665871287 ,20.3726521514])
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='cut off', yerr = [ 18.3561276306,69.2787080428 ,22.2110192154])
 
# Add xticks on the middle of the group bars
#plt.xlabel('Datasets with patter 1 - 6 instances', fontweight='bold')
plt.ylabel('execution time in sec', fontweight='bold')
plt.xticks([(r-0.15) + barWidth for r in range(len(bars1))], ['A', 'B', 'C'])
 
# Create legend & Show graphic
plt.legend()
#plt.show()
plt.plot()
plt.savefig('/home/fatemeh/Bureau/Stage/figures/pattern3_noleg.png',dpi = 300)

