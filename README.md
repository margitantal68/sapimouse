# SapiMouse
SapiMouse - a new dataset for Mouse Dynamics (2020)

Repository for paper: SapiMouse: Mouse Dynamics-based User Authentication Using Deep Feature Learning

## SapiMouse dataset
* Data collection software: https://mousedynamicsdatalogger.netlify.app
* Raw data: https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.html
   * First session: S3 (3 minutes)
   * Second session: S1 (1 minute)
   * 120 subjects (92 male, 28 female)
   * log file lines: [timestamp, button, state, x, y]
   
## Raw features folder: input_csv_mouse
   * step 1: first order differences (absolute value) - |dx|,|dy|
   * step 2: segmentation into fixed-sized blocks (128)
 
## Authentication measurements
   * split the dataset into 2 subsets
   	* subset 1: subjects 1 .. 72 (60%)
   	* subset 2: subjects 73 ..120 (40%)
   	
   * step 1: feature learning (training a fully convolutional neural network, see TRAINED_MODELS) using subset 1
   	 
   * step 2: creating a One-class SVM model for each subject from subset 2
## Publications
   * Margit Antal, Norbert Fejer, Krisztian Buza (<b>2021</b>), SapiMouse: Mouse Dynamics-based User Authentication Using Deep Feature Learning, May 19-21, 2021, [LINK](https://ieeexplore.ieee.org/document/9465583).
   * M. Antal, K. Buza and N. Fejer (<b>2021</b>), "SapiAgent: A Bot Based on Deep Learning to Generate Human-Like Mouse Trajectories," in IEEE Access, vol. 9, pp. 124396-124408, 2021, doi: 10.1109/ACCESS.2021.3111098, IF: 3.367, [LINK](https://ieeexplore.ieee.org/document/9530664).
   
   
	
	
