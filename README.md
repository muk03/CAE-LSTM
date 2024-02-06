# CAE-LSTM
An implementation of a Convolutional Auto-encoder trained on binary image sets, with a sequence to sequence predictor module. The sequence length needs to be edited manually.

# CAE, PCA, LSTM and GRU for Binary Image Processing and Prediction
## By Mukesh Dharanibalan | CID : 02107745 | email : md1021@ic.ac.uk
—————————————————
### Project Description:

This module contains separate implementations of the training and testing of a CAE module with 
10 layers in the encoder and decoder, comparisons with a PCA analysis of the same video,
an LSTM based Sequence to Sequence predictor and lastly a GRU Sequence to Sequence 
predictor.

This only works for binary videos due to the nature of implemented layers. The images are first compressed to 1/32 height and width of the videos. 

When training the CAE this is compared to the input to calculate loss and take a step in the optimiser.

When training the LSTM/GRU the 12 input frames are compressed, used to predict the next 4 and compared to the remaining 4 target frames to calculate the loss in batches of 3 and optimisation steps then taken. 

—————————————————
### Build Status (v1.0):

The current build v1.0.0 comes with:

-  CAEFinal.py : data loading, training and evaluation of a CAE
- PCA.py : data loading, and PCA analysis of a chosen video from the given data-set, and comparisons with the trained CAE model
- Sequence2SequenceLSTM.py : data loading, training and evaluation of a LSTM+Decoding model
- Sequence2SequenceGRU.py : data loading, training and evaluation of a GRU+Decoding model\

Models:
- best_model.pth : best CAE model
- bestGRU.pth : best GRU+Decoder Model (uses best CAE for encoding)
- bestLSTM_batch1.pth : best LSTM+Decoder (batch_size = 1) (uses best CAE for encoding)
- bestLSTM_batch4.pth : best LSTM+Decoder (batch_size = 4) (uses best CAE for encoding)


Further extension may be done by including gradient thresholds during sampling/prediction. when sampling this would prevent smearing and allow the binary features to be preserved more accurately. 

During prediction, setting gradient thresholds would mean that for binary frames that the sliding window/Kernel of the LSTM is able to skip over certain areas with high mean luminence, as they will not decrease, thereby reducing load and possibly increasing accuracy.

————————————————— 
### Code Style and Framework:

This project strictly follows standard PEP-8 formatting and convention. It requiresunderstanding of matplotlib, numpy, scikit, and pytorch libaries.

All functions/numerical methods are modulated and grouped based on their objectives and are self explanatory given ample knowledge of numerical methods.

————————————————— 
### How to use:

Simply running each labelled code on spyder, or vscode appropriately in a jupyter interactive will produce all the desired results:

during training, losses at each epoch are shown; during testing losses using different metrics are evaluated for the chosen dataset, and outputs are plotted for visual conformation against the target output.

PCA produces a scree-plot which clearly shows the threshold of the spatial-dimensionality of the CAE
thereby clearly showing significant data-loss  at that threshold when using PCA due to the larger number of features seen in these datasets.

CAE produces binary-image plots for CAE-processed data, and target (inputs itself) data.

LSTM, and GRU produce Self explanatory binary-image plots which can be used to evaluate method clearly.

Studies on computational time of processing/training are still in progress.

——————————————————
### Credits:

[1]
A. Dertat, “Applied Deep Learning - Part 3: Autoencoders,” Oct. 03, 2017. https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798.

[2]
D. Herkert, “Multivariate Time Series Forecasting with Deep Learning,” Jan. 07, 2022. https://towardsdatascience.com/multivariate-time-series-forecasting-with-deep-learning-3e7b3e2d2bcf.

[3]
M. Brems, “A One-Stop Shop for Principal Component Analysis,” Apr. 17, 2017. https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c.


### 
Libraries:

matplotlib.pyplot
numpy
torch
scikit

——————————————————
