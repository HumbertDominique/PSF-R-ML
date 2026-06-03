import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
# import keras
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

import TM26_unet as unet

from tqdm.auto import tqdm  # tqdm: merci Jonathan!

print("------------- Starting TM26_unet_run.py -------------")

outputDir = "output_ML/"
inputDir = "output_prep/"
history_path = outputDir + "/history.h5"

stabilisation_delay = 100

rng_SEED = 42

num_epochs = 5
early_stopping_patience = 15
best_val_loss = np.inf
epochs_no_improve = 0

GPU = True
MULTIGPU = True

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 25



print(torch.__version__)
# print(keras.__version__)
print(torch.cuda.is_available())

def pixel_adder(data, scale_factor = [1,1], final_shape = None):
    """
    Generates a phase screen base on a Phase Spectrum Density (PSD)
    Parameters:
        data (ndarray): source data
        scale_factor (floar): scalling factor. Can be > or < o
        final_shape (ndarray): Option to input directly the desired shape. final_shape/data.shape must be an integer or the final shape will be data.shape*(final_shape//data.shape).
    Returns (list):
        D_phi (ndarray): 2D array representing the phase structure
    TODO: Error check
    """

    datashape = data.shape

    if final_shape == None:
        iscale = scale_factor[0]
        jscale = scale_factor[1]
        image_scaled = np.zeros((int(scale_factor[0]*datashape[0]), int(scale_factor[1]*datashape[1])))
    else:
        iscale = int(final_shape[0]/datashape[0])
        jscale = int(final_shape[1]/datashape[1])
        image_scaled = np.zeros((int(final_shape[0]), int(final_shape[1])))
        
    for i in range(0, datashape[0]):
        for j in range(0, datashape[1]):
            iscaled = i*iscale
            jscaled = j*jscale
            image_scaled[int(iscaled):int(iscaled+iscale), int(jscaled):int(jscaled+jscale)] = data[i,j]
            # print(data[i,j])

    return image_scaled

print("------------- Loading data -------------")
# TODO: make size dynamic
image_size = (270, 270)


with h5py.File(inputDir+"ao_lightfield_train_1.h5", 'r') as temp:
    key = list(temp.keys())
    x_train = temp[key[0]][()][:]
with h5py.File(inputDir+"ao_lightfield_val_1.h5", 'r') as temp:
    key = list(temp.keys())
    x_val = temp[key[0]][()][:]

print(f"x_train shape: {x_train.shape}")
print(f"x_val shape: {x_val.shape}")

with h5py.File(inputDir+"ao_psf_train_1.h5", 'r') as temp:
    key = list(temp.keys())
    tempData = temp[key[0]][()][:]
    y_train = np.zeros((tempData.shape[0], 270, 270))
    for i in range(tempData.shape[0]):
        y_train[i] = pixel_adder(tempData[i], final_shape=[270, 270])
with h5py.File(inputDir+"ao_psf_val_1.h5", 'r') as temp:
    key = list(temp.keys())
    tempData = temp[key[0]][()][:]
    y_val = np.zeros((tempData.shape[0], 270, 270))
    for i in range(tempData.shape[0]):
        y_val[i] = pixel_adder(tempData[i], final_shape=[270, 270])

print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

plt.close(0)
plt.figure(0)
plt.imshow(y_train[0])
print("------------- Data Loaded -------------")


print("------------- Setting up training -------------")
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
model = unet.build_unet(input_shape=(270,270, 1), n_channels_out=1).to(device) # input shape is (width, height, channel) where channel is WFSx and WFSy

if torch.cuda.device_count() > 1 and MULTIGPU:
    model = torch.nn.DataParallel(model)

# TODO: à revoir
weight_path = outputDir+"{}_weights.best.pth".format("cxr_reg")


# Convert numpy arrays to torch tensors
# TODO: Evaluate pertinence of float32

x_train_tens = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
x_val_tens = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_train_tens = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tens = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

print(f"x_train_tens shape {x_train_tens.shape}")
print(f"y_train_tens shape {y_train_tens.shape}")
print(f"x_val_tens shape {x_val_tens.shape}")
print(f"y_val_tens shape {y_val_tens.shape}")

# Define batch size depending on device
batch_size = 32 if device.type == "cuda" else 12

train_loader = DataLoader(TensorDataset(x_train_tens, y_train_tens), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_tens, y_val_tens), batch_size=batch_size, shuffle=False)

for x_batch, y_batch in train_loader:
    print("x_batch shape:", x_batch.shape)  # should be [batch_size, 2, 270, 270]
    print("y_batch shape:", y_batch.shape)  # should be [batch_size, 1, 270, 270] or [batch_size, 270, 270]
    break

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.05, patience=3, cooldown=2, min_lr=1e-6)
criterion = unet.mse_loss

def run_epoch(model, loader, optimizer=None, phase="Train", epoch=0, num_epochs=1):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(
        loader,
        desc=f"{phase} {epoch + 1}/{num_epochs}",
        leave=False
    )

    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(targets, outputs)

            if is_training:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        acc = (preds == targets).float().mean().item()

        running_loss += loss.item() * inputs.size(0)
        running_acc += acc * inputs.size(0)

        # Update the loading bar with current batch values
        pbar.set_postfix(
            loss=f"{loss.item():.6f}",
            acc=f"{acc:.6f}"
        )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)

    return epoch_loss, epoch_acc

best_model_wts = copy.deepcopy(model.state_dict())

history = {
    "loss": [],
    "val_loss": [],
    "binary_accuracy": [],
    "val_binary_accuracy": [],
    "auc": [],
    "val_auc": []
}

print("------------- Training starting -------------")
for epoch in range(num_epochs):
    train_loss, train_acc = run_epoch(
        model,
        train_loader,
        optimizer=optimizer,
        phase="Train",
        epoch=epoch,
        num_epochs=num_epochs
    )

    val_loss, val_acc = run_epoch(
        model,
        val_loader,
        optimizer=None,
        phase="Val",
        epoch=epoch,
        num_epochs=num_epochs
    )

    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["binary_accuracy"].append(train_acc)
    history["val_binary_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{num_epochs} - "
        f"loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - "
        f"acc: {train_acc:.6f} - val_acc: {val_acc:.6f}"
    )

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), weight_path)
        print(f"Saved best model to {weight_path}")
    else:
        epochs_no_improve += 1

    # Stop training early if validation loss does not improve
    if epochs_no_improve >= early_stopping_patience:
        print("Early stopping triggered")
        break

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), outputDir + "/model.weights.pth")

with h5py.File(history_path, "w") as f:
    for key, values in history.items():
        f.create_dataset(key, data=values)
print(f"Saved history to {history_path}")


print("------------- Cleaning up GPU memory -------------")
del model
del optimizer
del scheduler
del x_train_tens
del x_val_tens
del y_train_tens
del y_val_tens
del train_loader
del val_loader
torch.cuda.empty_cache()
print("GPU memory freed")