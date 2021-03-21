External Libraries used:
Pytorch (torch)
openCV (cv2) (for data augmentation only)
matplotlib (for plots)
PIL
skimage

To run the code:
1. to train the binary model (shown the notebook cell) run !python ./train_binary.py --gpu --epochs 25 --batchsize 32 --upsample True --transform True --scheduler True --decay True
2. To load the model (shown in the last cell of the notebook) - !python ./train_binary.py --gpu --sample True (this loads a saved model and uses it on the test set)
The saved model path = model_2021_03_21-07_34_47_AM.pt

Files:
model.py - file with the neural net architecture class obect for binary and multi-class classifiers
train_multiclass.py - file with the train, test, validate functions for the multi-class classifier
train_binary.py - file with the the train, test, validate functions for the binary classifier
plots.py - file with the relevant functions for plotting functions
lung_data_loader_with_transform.py - the data loader object is defined here


