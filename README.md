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
The data is an Excel file (provided upon request) that contains precursors and their results. 
Feature space consists of the following variables:
- Waterglass content
- N330
- K3PO4
- Mn3O4
- Drying (yes, no)
- Mixing (classical, additional)
- Furnace temperature
- Heating rate
- Foaming time

Target space consists of:
- Apparent density (needs to be minimized)
- Closed porosity (need to be maximized)

>**Important**: when using the model for predictions, the input data needs to be preprocessed the same way the train data was. The scaler for the target space is saved in _model/target_scaler.save_.
## Methodology

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
  (mse): MeanSquaredError()
)
```
Smooth L1 Loss is used as the loss function.
The NN is evaluated using Pearson Correlation Coefficient, averaged on 5 seeds.
The final model, trained on all available data, is saved as model/mlp_v0.ckpt.

Steps to load the model and run it on new data:

```python
import model
import dataset
import numpy as np

# load the model from the last save checkpoint
mlp = model.MLP.load_from_checkpoint(checkpoint_path="model/mlp_v0.ckpt")
# load data, let's say x as numpy array
x = np.random.rand(15, 9)
predictions = model.predict_with_model(x, mlp)
```

Trained models are saved in lightning_logs/mlp_model. Tensorboard can be activated withthe following command
```commandline
tensorboard --logdir=lightning_logs
```

