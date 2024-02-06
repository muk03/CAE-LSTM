
# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Video Reader
def reader(path):
    out = cv2.VideoCapture(path)
    tsteps = []

    while 1 > 0:
        truth, data = out.read()
        if not truth:
            break
        
        grays = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grays, 128, 1, cv2.THRESH_BINARY)
        tsteps.append(binary)

    out.release()
    procData = np.array(tsteps)

    return procData


# %%

def PCAnalysis(videoData):
    # Perform PCA
    pObj = PCA()
    pObj.fit(videoData)

    # Get explained variance ratio
    eVR = pObj.explained_variance_ratio_
    cEVR = np.cumsum(eVR)

    return pObj, cEVR

def compress(videoData, N):
    p2 = PCA(n_components=N)
    viParse = videoData.reshape(16, -1)
    compressData = p2.fit_transform(viParse)
    return compressData, p2

def decompress (compressedData, pcaObject):
    decompressData = pcaObject.inverse_transform(compressedData)
    out = decompressData.reshape(16, 128, 128)
    return out


# %%



# and all its 16 frames - Most of the videos will have similar behavior, and can be evaluated by changing
# the range appropriately

video45 = []
video47 = []

video45 = reader(f'VIDEOS/fire_Chimney_video_{45}.mp4')
video47 = reader(f'VIDEOS/fire_Chimney_video_{47}.mp4')


p45 = []
cu45 = []

p47 = []
cu47 = []

pC1 = np.arange(1, 129, step = 1)

for i in range(len(video45)):
    pTemp45, cumulus45 = PCAnalysis(video45[i])
    p45.append(pTemp45)
    cu45.append(cumulus45)

    pTemp47, cumulus47 = PCAnalysis(video47[i])
    p47.append(pTemp47)
    cu47.append(cumulus47)

# %%

# Scree Plot production to visualise variance ratio 
    
fig1, axs1 = plt.subplots(1, 2, figsize = (15, 8))

for i in range(0, 16, 5):
    axs1[0].plot(pC1, cu45[i], 'x-', color='black', label = f'Frame Number {i+1}')
    axs1[1].plot(pC1, cu47[i], 'x-', color='black', label = f'Frame Number {i+1}')
    
axs1[0].set_title('Scree Plot - Video 45')
axs1[1].set_title('Scree Plot - Video 47')
axs1[0].set_xlabel('Number of Principal Components')
axs1[0].set_ylabel('Cumulative Explained Variance')
axs1[1].set_xlabel('Number of Principal Components')
axs1[1].set_ylabel('Cumulative Explained Variance')
    
axs1[0].grid(True)
axs1[1].grid(True)
axs1[0].axvline(16, linestyle = 'dashed', color = 'black', label = 'CAE - Spatial Dimensionality')
axs1[1].axvline(16, linestyle = 'dashed', color = 'black', label = 'CAE - Spatial Dimensionality')
axs1[0].legend(loc='lower right')
axs1[1].legend(loc='lower right')


# %%

video45PCA, p45o = compress(video45, 10)
video47PCA, p47o = compress(video47, 10)

decomp45 = decompress(video45PCA, p45o)
decomp47 = decompress(video47PCA, p47o)

# %%

fig1, axs1 = plt.subplots(2, 2, figsize = (12, 10))

axs1[0, 0].imshow(video45[7], cmap= 'gray')
axs1[0, 1].imshow(decomp45[7], cmap= 'gray')

axs1[0, 0].set_title('Video 45, Frame 8 - Actual Frame')
axs1[0, 1].set_title('Video 45, Frame 8 - PCA Frame')

axs1[1, 0].imshow(video47[7], cmap= 'gray')
axs1[1, 1].imshow(decomp47[7], cmap= 'gray')

axs1[1, 0].set_title('Video 47, Frame 8 - Actual Frame')
axs1[1, 1].set_title('Video 47, Frame 8 - PCA Frame')


# %%
