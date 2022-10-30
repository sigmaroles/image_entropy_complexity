from PIL import Image
import numpy as np
import scipy.stats as stats
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


basedir = "./satImage_copy/"
csvfname = "outputs_satImage.csv"
dx = 2
dy = 2
printLabel = False
pngname = "plot_SatImage.png"


def nse(p,n):  # normalized shannon entropy
    #return (stats.entropy(p) / np.log(n)
    s1 = 0
    for prob1 in p:
        s1 = s1 + prob1*np.log(1/prob1)
    return s1 / np.log(n)   
    # this is equivalent to : 
    # stats.entropy(p) / np.log(n)

def D(p, u): # equation 3
    n = p.shape[0]
    S = stats.entropy
    return S((p+u)/2) - S(p)/2 - S(u)/2

if __name__=='__main__':
    filelist = os.listdir(basedir)
    filelist.sort()
    print (f"{len(filelist)} files found in {basedir}")

    n = np.math.factorial(dx*dy)

    with open(csvfname, "w") as ofh:
        s1 = "filename,complexity,entropy"
        ofh.write(s1+"\n")
        for fnn in filelist:
            # data preparation etc.
            filename = basedir+fnn
            img = Image.open(filename)
            idata = np.asarray(img)
            idata = np.uint32(np.mean(idata, axis=2)) # arithmetic mean, as done by the paper
            #img2 = Image.fromarray(idata)    
            COLMAX, ROWMAX = img.size # Get image size (width, height) i.e. column followed by row

            # ==== algorithm begins ====

            argsort_results = np.uint32(np.zeros([(ROWMAX-1)*(COLMAX-1), 4]))
            acounter = 0
            for i in np.arange(0, ROWMAX-1, 1):
                for j in np.arange(0, COLMAX-1, 1):
                    wflat = idata[i:i+dx, j:j+dy].flatten()  # find the "ordinal pattern" (1-dimensional)
                    argsort1 = np.uint32(np.argsort(wflat, kind='stable')) # in case of tie, preserve original order
                    argsort_results[acounter] = argsort1 # save the sort results for this iteration
                    acounter = acounter+1
            elements, counts = np.unique(argsort_results, axis=0, return_counts=True)
            p = counts / argsort_results.shape[0]  # probability distribution, divide counts by total
            h = nse(p,n) # normalized shannon entropy... equation 1
            
            # complexity calculation begins here
            Dstar = -0.5 * ( (n+1)/n*np.log(n+1) + np.log(n) - 2*np.log(2*n) ) # equation 4
            u = np.array([1/n]*elements.shape[0]) # uniform distribution
            C = (D(p,u) * h) / Dstar # Equation 2

            # ==== algoithm ends ====
            
            s2 = f"{fnn},{C:.10f},{h:.10f}"
            ofh.write(s2+"\n")
            print (f"{filename} done")

        # file loop ends

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

