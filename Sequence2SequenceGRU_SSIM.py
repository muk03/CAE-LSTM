# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import time
from pytorch_msssim import ms_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

# Read in the video as necessary
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

#GRU Class for sequence to sequence prediction
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size=16, hidden_size=16, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),  # 1/16
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),  # 1/8
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0), # 1/4
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2, padding=0), # 1/2
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2, padding=0), # 1
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
            )
        

    def forward(self, x):
        flatx = x.view(x.shape[0], x.shape[1], -1) # flatten the spatial dimensions for compatibility
        flatx = flatx.permute(1, 0, 2) # permute to correspond to batch_size = True
        
        lstmx, _ = self.rnn(flatx) # run through the GRU layer
        
        permutelstmx = lstmx.permute(1, 0, 2) # re configure to have shape [8, 16, 1089] back from [16, 8, 1089]
        permutelstmx = permutelstmx.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) # reconfigure to have shape [8, 16, 33, 33] for decoder
        
        decomp = self.decoder(permutelstmx)
        
        return decomp
    
    def testing(self, x):
        flatx = x.view(x.shape[0], x.shape[1], -1) # flatten the spatial dimensions for compatibility
        flatx = flatx.permute(1, 0, 2) # permute to correspond to batch_size = True
        
        lstmx, _ = self.rnn(flatx) # run through the GRU layer

        return lstmx

# General CAE architecture for compression and decompression 
    
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

    
    def compress(self, x):
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


        sizes = (c1.size(), c2.size(), c3.size(), c4.size(), c5.size())
        
        return c6, sizes, (i1, i2, i3, i4, i5)
    
    def decompress(self, x, indices, sizes):
        c7 = self.Conv7(x)
        mup1 = self.MaxUnPool1(c7, indices[4], output_size= sizes[4])

        c8 = self.Conv8(mup1)
        mup2 = self.MaxUnPool2(c8, indices[3], output_size=sizes[3])

        c9 = self.Conv9(mup2)
        mup3 = self.MaxUnPool3(c9, indices[2], output_size=sizes[2])

        c10 = self.Conv10(mup3)
        mup4 = self.MaxUnPool4(c10, indices[1], output_size=sizes[1])

        c11 = self.Conv11(mup4)
        mup5 = self.MaxUnPool5(c11, indices[0], output_size=sizes[0])

        c12 = self.Conv12(mup5)

        return c12

# %%

# Loading in training and validation data and creating necessary data structures
trainStore = []
validStore = []

for i in range(0, 32):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{i}.mp4')
    trainStore.append(tempViData)

for k in range(33, 41):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{k}.mp4')
    validStore.append(tempViData)


trainStore = np.array(trainStore)
validStore = np.array(validStore)


tensorTrain = torch.tensor(trainStore, dtype=torch.float32).to(device)
tensorValid = torch.tensor(validStore, dtype=torch.float32).to(device)

batch = 4

loadTrain = DataLoader(TensorDataset(tensorTrain, tensorTrain), batch_size=batch, shuffle=True)
loadValid = DataLoader(TensorDataset(tensorValid, tensorValid), batch_size=batch, shuffle=False)


# %%

predictor = GRU().to(device)

encoder = CAE()
encoder.load_state_dict(torch.load('best_model.pth'))
encoder.eval()

# Use MS-SSIM as the loss function
criterion = ms_ssim
critBCE = nn.MSELoss()

optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

best_val_loss = float('inf')
best_model_state = None

predictor.train()

epochs = 100

# %%
# Training the model where 12 frames are used as a basis to predict the remaining 4
meanLosses = []
epochNumber = []

start_time = time.time()

for epoch in range(epochs):
    cumuLoss = 0.0

    for data in loadTrain:
        video, _ = data
        allPred = torch.tensor([], dtype=torch.float32)
        allTargs = torch.tensor([], dtype=torch.float32)

        for i in range(len(video)):
            inputs = video[i].unsqueeze(1).to(device)

            basis = inputs[:12]
            targets = inputs[12:]
            optimizer.zero_grad()

            basisCompression, _, _ = encoder.compress(basis)
            targetsCompression, sizeT, indexT = encoder.compress(targets)

            outputs = predictor.forward(basisCompression)
            predictionFinal = outputs[:4]

            targetFinal = encoder.decompress(targetsCompression, indexT, sizeT)

            allPred = torch.cat((allPred, predictionFinal), dim=0)
            allTargs = torch.cat((allTargs, targetFinal), dim=0)

        # Calculate MS-SSIM
        ms_ssim_loss = 1 - criterion(allPred, allTargs, win_size=7)

        loss = ms_ssim_loss + critBCE(allPred, allTargs)

        loss.backward()
        optimizer.step()

    scheduler.step()

    with torch.no_grad():
        predictor.eval()

        allPred = torch.tensor([], dtype=torch.float32)
        allTargs = torch.tensor([], dtype=torch.float32)

        for data in loadValid:
            video, _ = data

            for i in range(len(video)):
                inputs = video[i].unsqueeze(1).to(device)

                basis = inputs[:12]
                targets = inputs[12:]

                basisCompression, _, _ = encoder.compress(basis)
                targetsCompression, sizeT, indexT = encoder.compress(targets)

                outputs = predictor.forward(basisCompression)
                predictionFinal = outputs[:4]

                targetFinal = encoder.decompress(targetsCompression, indexT, sizeT)

                allPred = torch.cat((allPred, predictionFinal), dim=0)
                allTargs = torch.cat((allTargs, targetFinal), dim=0)

            ms_ssim_loss_val = 1 - criterion(allPred, allTargs, win_size=7)

            val_loss = ms_ssim_loss_val + critBCE(allPred, allTargs)
            cumuLoss += val_loss

        meanVal_loss = cumuLoss / (len(tensorValid)/batch)

        meanLosses.append(meanVal_loss)
        epochNumber.append(epoch)

        if meanVal_loss < best_val_loss:
            best_val_loss = meanVal_loss
            best_model_state = predictor.state_dict()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Loss: {meanVal_loss:.4f}')


end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time for training of GRU Method: {elapsed_time} seconds")

torch.save(best_model_state, 'bestGRU_SSIM+MSE.pth')

# %%

#plotting out the evoluation of the loss function BCE as the model gets trained
# This allows to clearly visualise convergence of the model on optimal
# parameters

meanLosses = np.array(meanLosses)

plt.figure(figsize = (10, 10), dpi = 900)
plt.plot(epochNumber, meanLosses * 100, 'x-', color = 'black', 
         label ='SSIM Loss Convergece ')

plt.title('Convergence of CAE on Optimal Parameters - SSIM Loss Function')
plt.xlabel('Epoch Number')
plt.ylabel('Information Loss (%)')
plt.legend()
plt.grid()

# %%

np.savetxt('GRU_SSIM+MSE.txt', [meanLosses])

# %%

# Loading and evaluating testing data with different metrics

testStore = []
for i in range(41, 49):
    tempViData = reader(f'VIDEOS/fire_Chimney_video_{i}.mp4')
    testStore.append(tempViData)

testStore = np.array(testStore)

testGRU = GRU()

testGRU.load_state_dict(torch.load('bestGRU_SSIM+BCE.pth'))

tensorTest = torch.tensor(testStore, dtype=torch.float32).to(device)
loadTest = DataLoader(TensorDataset(tensorTest, tensorTest), batch_size=8, shuffle=False)

testGRU.eval() 

allOut = []
allTargs = []
videoWiseLosses = []
videoID = []

with torch.no_grad():

    allPred = torch.tensor([], dtype = torch.float32)
    allTarg = torch.tensor([], dtype = torch.float32)

    for data in loadTest:
        video, _ = data
        for i in range(len(video)):
            inputs = video[i].unsqueeze(1).to(device)
            basis = inputs[:12]
            targets = inputs[12:]
            
            basisCompressed, _, _ = encoder.compress(basis)
            prediction = testGRU.forward(basisCompressed)
            predictionFinal = prediction[:4]

            targetCompressed, sizeT, indexT = encoder.compress(targets)
            targetFinal = encoder.decompress(targetCompressed, indexT, sizeT)

            allPred = torch.cat((allPred, predictionFinal), dim = 0)
            allTarg = torch.cat((allTarg, targetFinal), dim = 0)

            allOut.append(predictionFinal.numpy())
            allTargs.append(targetFinal.numpy())


            individualLosses = 1 - criterion(predictionFinal, targetFinal, win_size=7) + critBCE(predictionFinal, targetFinal)
            videoWiseLosses.append(individualLosses)

            videoID.append(i + 41)

        test_loss = (1 - criterion(allPred, allTarg, win_size=7)) + critBCE(allPred, allTarg)

print(f'Average Test Loss: {test_loss:.4f}')

# %%

videoWiseLosses = np.array(videoWiseLosses)

# Video Wise Individual Losses BCE Metric
plt.figure(figsize = (10, 10))
plt.bar(videoID, videoWiseLosses, color = 'black')
plt.title('Video Wise Losses - MSE+SSIM Metric - GRU prediction')
plt.xlabel('Video ID')
plt.ylabel('Losses (%)')
plt.grid()

# %%

# Visualise
allOut = np.array(allOut)
allTargs = np.array(allTargs)


Trows = 2 * len(allOut)
Tcolumn = len(allOut[0])

fig_width = 16
fig_height = Trows * 4

fig, axs = plt.subplots(nrows=Trows, ncols=Tcolumn, figsize=(fig_width, fig_height))
    
for i in range(0, len(allOut) * 2, 2):
    for j in range(len(allOut[0])):
        axs[i, j].imshow(allOut[i//2][j][0], cmap = 'gray')
        axs[i+1, j].imshow(allTargs[i//2][j][0], cmap = 'gray')

        axs[i, j].set_title(f'Video {i//2 + 41}, Frame {j} - Highest Loss Set - Predicted')
        axs[i + 1, j].set_title(f'Video {i//2 + 41}, Frame {j} - Highest Loss Set - Target')

plt.show()

# %%

# Visualise
allOut = np.array(allOut)
allTargs = np.array(allTargs)


Trows = 2 
Tcolumn = len(allOut[0])

fig_width = 16
fig_height = Trows * 4

fig, axs = plt.subplots(nrows=Trows, ncols=Tcolumn, figsize=(fig_width, fig_height))
    

for j in range(len(allOut[0])):
    axs[0, j].imshow(allOut[3][j][0], cmap = 'gray')
    axs[1, j].imshow(allTargs[3][j][0], cmap = 'gray')

    axs[0, j].set_title(f'Video 44, Frame {j} - Predicted')
    axs[1, j].set_title(f'Video 44, Frame {j} - Target')

plt.show()

# %%





















