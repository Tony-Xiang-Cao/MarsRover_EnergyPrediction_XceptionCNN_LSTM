# MarsRover_EnergyPrediction_XceptionCNN_LSTM
A Deep Learning Model for Estimating Mars Rover Power Consumption Based on Terrain Images

Final project of University of Toronto ROB501 Computer Vision for Robotics

This research project presents a deep learning approach to estimate the power consumptions of Mars Rover to driver over different type of terrains. The training dataset is based on the Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset (CPET), collected at Canadian Space Agency’s Mars Emulation Terrain (MET). https://starslab.ca/enav-planetary-dataset/ 

The deep learning model is trained on 12099 color images, collected from Rover onboard camera (Occam Omni Stereo) during 5 runs of the dataset, and tested on 2133 images in Run2 of the dataset. The deep learning model uses transfer learning from the Xception model to extract visual features, and then adds LSTM layers to capture sequence information and TimeDistributed Dense layers to add time dimension of output power data. The model achieved a mean absolute error (MAE) of 63.3 on testset, which deviates 24.5% from the average power ground truth in Run2 of 257.4 Watt.

