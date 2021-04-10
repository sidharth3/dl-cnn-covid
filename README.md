#### External Libraries used:
Pytorch (torch)
openCV (cv2) (for data augmentation only)
matplotlib (for plots)
PIL
skimage

Main Submission with plots and results - Final_Notebook.ipynb

NOTE: We have done some preprocessing which involves upsampling of the dataset. If in case the training
needs to be performed, the code on the notebook section 'file preparation' section. After running
this code (on a duplicate dataset directory) , the files on this directory must be copied over to the original dataset/train/normal/
and dataset/train/infected/covid/

#### To run the code:
1. to train the binary model (shown the notebook cell) run `!python ./train_binary.py --gpu --epochs 25 --batchsize 32 --upsample True --transform True --scheduler True --decay True`
2. To load the model (shown in the last cell of the notebook) - `!python ./train_binary.py --gpu --sample True` (this loads a saved model and uses it on the test set)
The saved model path = model_2021_03_21-07_34_47_AM.pt

Files:
- Item `model.py` - file with the neural net architecture class obect for binary and multi-class classifiers
- Item  `train_multiclass.py` - file with the train, test, validate functions for the multi-class classifier
- Item  `train_binary.py` - file with the the train, test, validate functions for the binary classifier
- Item  `plots.py` - file with the relevant functions for plotting functions
- Item  `lung_data_loader_with_transform.py` - the data loader object is defined here


