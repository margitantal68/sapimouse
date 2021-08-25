# sapimouse
SapiMouse - a new dataset for Mouse Dynamics

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
   
   
	
	
