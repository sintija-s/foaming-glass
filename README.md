# Foaming glass

Built using Python 3.10.

Create virtual environment:
```commandline
cd foaming-glass-master
python -m venv .venv
MacOS/Linux: source .venv/bin/activate
Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

## Task: Predict Apparent Density and Closed Porosity from experimental parameters

## Data
The data contains Foaming Glass precursors and their results. 
It should be put in the _data_ folder, and if the name is different from _data.xlsx_, please alter the F_NAME variable in main.py

The feature space consists of the following variables:
- Waterglass content
- N330
- K3PO4
- Mn3O4
- Drying {yes, no}
- Mixing {classical, additional}
- Furnace temperature
- Heating rate
- Foaming time

Target space consists of:
- Apparent density (needs to be minimized)
- Closed porosity (needs to be maximized)

>**Important**: when using the model for predictions, the input data needs to be preprocessed the same way the train data was. The scaler for the target space is saved in _model/target_scaler.save_.
## Model

The multi-layer perceptron is made using PyTorch Lightning. It consists of 2 hidden layers and one output layer, defined as follows:
``` 
MLP(
  (NN): Sequential(
    (0): Linear(in_features=9, out_features=6, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6, out_features=4, bias=True)
    (3): ReLU()
    (4): Linear(in_features=4, out_features=2, bias=True)
  )
  (loss): SmoothL1Loss()
)
```
Smooth L1 Loss is used as the loss function.
The NN is evaluated using Pearson Correlation Coefficient, 
averaged on 5 seeds.

Semi-deviation was chosen as uncertainty measure, calculated on the results accross 10 runs (10 different random seed intializations, both for the data and the model - utilizing the entire dataset). 
All 10 models are saved in _model/_ folder, with a _.ckpt_ extension.


### Steps to load the model and run it on new data:

```python
import numpy as np
import model

# Set the model name to whichever base model name you are using
model_name = "mlp_v1"
# load data, let's say x as numpy array
x = np.random.rand(15, 9)
predictions = model.predict_with_model(x, model_name)
```
The output of the prediction process is a numpy array that contains the averaged value of the predictions across all 10 trained models.

All predictions, their average and standard deviation can be found in _results/_ folder, for density and porosity respectively.

Trained models are saved in _lightning_logs/mlp_model/_. Tensorboard can be activated with the following command
```commandline
tensorboard --logdir=lightning_logs
```

