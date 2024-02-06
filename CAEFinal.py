# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# %%

# Reader Function and consequently reading in the necessary data
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

# Model that contains 10 layers, 5 convolution and 5 pooling layers 
# to convolve and compress

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        # Encoder
        self.Conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.MaxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, return_indices=True) # 1/2

        self.Conv2 = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.MaxPool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, return_indices=True)  # 1/4

        self.Conv3 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.MaxPool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, return_indices=True)  # 1/8

        self.Conv4 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.MaxPool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, return_indices=True)  # 1/16

        self.Conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                    nn.ReLU())
        self.MaxPool5 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, return_indices=True)  # 1/32

        self.Conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        #Decoder
        self.Conv7 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.MaxUnPool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0) # 1/16 (reverse of encoder)

        self.Conv8 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.MaxUnPool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 1/8

        self.Conv9 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU())
        self.MaxUnPool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0) # 1/4

        self.Conv10 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU())
        self.MaxUnPool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 1/2

        self.Conv11 = nn.Sequential(nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU())
        self.MaxUnPool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)  # 1

        self.Conv12 = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        #encoding
        c1 = self.Conv1(x)
        mp1, i1 = self.MaxPool1(c1)

        c2 = self.Conv2(mp1)
        mp2, i2 = self.MaxPool2(c2)

        c3 = self.Conv3(mp2)
        mp3, i3 = self.MaxPool3(c3)

        c4 = self.Conv4(mp3)
        mp4, i4 = self.MaxPool4(c4)

        c5 = self.Conv5(mp4)
        mp5, i5 = self.MaxPool5(c5)

        c6 = self.Conv6(mp5)

        # decoding
        c7 = self.Conv7(c6)
        mup1 = self.MaxUnPool1(c7, i5, output_size=c5.size())

        c8 = self.Conv8(mup1)
        mup2 = self.MaxUnPool2(c8, i4, output_size=c4.size())

        c9 = self.Conv9(mup2)
        mup3 = self.MaxUnPool3(c9, i3, output_size=c3.size())

        c10 = self.Conv10(mup3)
        mup4 = self.MaxUnPool4(c10, i2, output_size=c2.size())

        c11 = self.Conv11(mup4)
        mup5 = self.MaxUnPool5(c11, i1, output_size=c1.size())

        c12 = self.Conv12(mup5)

        return c12



# %%
    
# Importing all the data as required and creating necessary structures
trainStore = []
validStore = []

for i in range(33):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{i}.mp4')

    for j in range(len(tempViData)):
        trainStore.append(tempViData[j])

for k in range(33, 41):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{k}.mp4')
    for l in range(len(tempViData)):
        validStore.append(tempViData[l])


trainStore = np.array(trainStore)
validStore = np.array(validStore)

tensorTrain = torch.tensor(trainStore, dtype=torch.float32)
tensorValid = torch.tensor(validStore, dtype=torch.float32)

batch = 32
epochs = 100


loadTrain = DataLoader(TensorDataset(tensorTrain, tensorTrain), batch_size=batch, shuffle=True)
loadValid = DataLoader(TensorDataset(tensorValid, tensorValid), batch_size=batch, shuffle=False)



# %%
    
# Training the model
testCAE = CAE()
testCAE.train() # set to train

# initalise the weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)

testCAE.apply(weights_init)


# BCE as data has binary channels

criterion = nn.BCELoss()
optimizer = optim.Adam(testCAE.parameters(), lr=0.001)


best_val_loss = float('inf')
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

# %%

lossEvolution = []
epochNumber = []

# training loops
for epoch in range(epochs):

    cumuLoss = 0.0

    for data in loadTrain:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = testCAE.forward(inputs.unsqueeze(1))
        loss = criterion(outputs, inputs.unsqueeze(1))
        loss.backward()
        optimizer.step()

    scheduler.step()  # Update learning rate
    

    # validation set

    with torch.no_grad():

        testCAE.eval()

        for data in loadValid:
            inputs, _ = data
            outputs = testCAE.forward(inputs.unsqueeze(1))
            val_loss = criterion(outputs, inputs.unsqueeze(1))

            cumuLoss += val_loss.item()

        meanLoss = cumuLoss/(len(tensorValid)/batch)
        lossEvolution.append(meanLoss)
        epochNumber.append(epoch)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {meanLoss:.4f}')

    if meanLoss < best_val_loss:
        best_val_loss = meanLoss
        best_model_state = testCAE.state_dict()

torch.save(best_model_state, 'best_modelTemp.pth') # change file name each time to prevent overwriting

# %%

#plotting out the evoluation of the loss function BCE as the model gets trained
# This allows to clearly visualise convergence of the model on optimal
# parameters

lossEvolution = np.array(lossEvolution)

plt.figure(figsize = (10, 10), dpi = 900)
plt.plot(epochNumber, lossEvolution * 100, 'x-', color = 'black', 
         label ='BCE Loss Convergece ')

plt.title('Convergence of CAE on Optimal Parameters - BCE Loss Function')
plt.xlabel('Epoch Number')
plt.ylabel('Information Loss (%)')
plt.legend()
plt.grid()

#%%

# testing set
testStore = []
for i in range(41, 49):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{i}.mp4')

    for j in range(len(tempViData)):
        testStore.append(tempViData[j])

testStore = np.array(testStore)

# load the optimised model

testCAE = CAE()
testCAE.load_state_dict(torch.load('best_model.pth'))

# data structures

test_data = torch.tensor(testStore, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(test_data, test_data), batch_size=16, shuffle=False)

# %%


testCAE.eval()  # Set the model to evaluation mode
test_loss = 0.0

allOut = []
testLosses = []
videoID = []

i = 0.0

with torch.no_grad():

    # this loop essentially picks 16 frames which are just one video
    for data in test_loader:
        cumuTestLoss = 0.0

        inputs, _ = data
        outputs = testCAE(inputs.unsqueeze(1))
        test_loss += criterion(outputs, inputs.unsqueeze(1)).item()
        allOut.append(outputs.numpy())

        # adding losses across all frames from this video
        meanTestLoss = criterion(outputs, inputs.unsqueeze(1)).item()
        #calculating mean loss for thei video

        testLosses.append(meanTestLoss)

        videoID.append(i + 41)
        i += 1

average_test_loss = test_loss / len(test_loader)
print(f'Average Test Loss: {average_test_loss:.4f}')

# %%

# bar plot showing how the CAE performs against different videos in the testing dataset
testLosses = np.array(testLosses)


plt.figure(figsize=(10, 10))
plt.bar(videoID, testLosses * 100, 0.5, bottom=0, align='center', color = 'black')
plt.title('Data Loss Across Different Videos - Test Dataset')
plt.xlabel('Video Number/ID')
plt.ylabel('Information Loss (%)')
plt.grid()


# %%

for i in range(16): 
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].imshow(allOut[0][i][0], cmap='gray') 
    axs[1].imshow(testStore[i], cmap='gray')  
    plt.show()

# %%
videoWise = []
for i in range(41, 49):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{i}.mp4')
    videoWise.append(tempViData)

# %%
    

fig1, axs1 = plt.subplots(2, 2, figsize = (12, 10))

axs1[0, 0].imshow(videoWise[4][7], cmap= 'gray')
axs1[0, 1].imshow(allOut[4][7][0], cmap= 'gray')

axs1[0, 0].set_title('Video 45, Frame 8 - Highest Loss Video - Actual Frame')
axs1[0, 1].set_title('Video 45, Frame 8 - Highest Loss Video - CAE Frame')

axs1[1, 0].imshow(videoWise[6][7], cmap= 'gray')
axs1[1, 1].imshow(allOut[6][7][0], cmap= 'gray')

axs1[1, 0].set_title('Video 47, Frame 8 - Lowest Loss Video - Actual Frame')
axs1[1, 1].set_title('Video 47, Frame 8 - Lowest Loss Video - CAE Frame')


# %%
