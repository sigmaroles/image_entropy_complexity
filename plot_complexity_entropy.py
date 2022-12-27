from traceback import print_last
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


csvfname = "outputs_personal.csv"
pngname = "plot_movScreens.png"
printLabel = False

cedf = pd.read_csv(csvfname)
cedf.head()

x = cedf['entropy']
y = cedf['complexity']

reg_result = stats.linregress(x, y)
xline = np.linspace(x.min(),x.max(),1000)
# y = mx + c
yline = reg_result.slope*xline + reg_result.intercept

labels = cedf['filename']
#labels = [s1[s1.find("(")+1:s1.find(")")] for s1 in labels] # only numbers, discard "UL" and ".png"

f = plt.figure(figsize=(12,11))
ax = f.gca()
ax.scatter(x, y)
ax.plot(xline,yline,color='orange',linewidth=0.9)

if printLabel:
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]), xytext = (5,1), textcoords="offset points") # https://stackoverflow.com/a/60786569

ax.set_ylabel("Complexity")
ax.set_xlabel("Entropy")

plt.tight_layout()
f.savefig(pngname, facecolor="white")

print (reg_result)