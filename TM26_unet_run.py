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
import myfunctions

from tqdm.auto import tqdm  # tqdm: merci Jonathan!

print("------------- Starting TM26_unet_run.py -------------")

outputDir = "output_ML/"
inputDir = "output_prep/"
history_path = outputDir + "/history.h5"


num_epochs = 100
early_stopping_patience = 20
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


print("------------- Loading data -------------")
# TODO: make size dynamic
with h5py.File(inputDir+"ao_lightfield_train_1.h5", 'r') as temp:
    key = list(temp.keys())
    # print(key)
    x_train = temp[key[0]][()]
with h5py.File(inputDir+"ao_lightfield_val_1.h5", 'r') as temp:
    key = list(temp.keys())
    x_val = temp[key[0]][()]

image_size = (x_train.shape[1], x_val.shape[2])
print(image_size)

print(f"x_train shape: {x_train.shape}")
print(f"x_val shape: {x_val.shape}")

with h5py.File(inputDir+"ao_ipsf_train_1.h5", 'r') as temp:
    key = list(temp.keys())
    tempData = temp[key[0]][()][:]
    y_train = np.zeros((tempData.shape[0], 270, 270))
    for i in range(tempData.shape[0]):
        y_train[i] = myfunctions.pixel_adder(tempData[i], final_shape=[270, 270])
with h5py.File(inputDir+"ao_ipsf_val_1.h5", 'r') as temp:
    key = list(temp.keys())
    tempData = temp[key[0]][()][:]
    y_val = np.zeros((tempData.shape[0], 270, 270))
    for i in range(tempData.shape[0]):
        y_val[i] = myfunctions.pixel_adder(tempData[i], final_shape=[270, 270])

print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

plt.close(0)
plt.figure(0)
plt.imshow(x_train[0])

device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
model = unet.build_unet(input_shape=(*image_size, 1), n_channels_out=1).to(device) # input shape is (width, height, channel) where channel is WFSx and WFSy
print("------------- Data Loaded -------------")


print("------------- Setting up training -------------")
if torch.cuda.device_count() > 1 and MULTIGPU:
    model = torch.nn.DataParallel(model)

# TODO: à revoir
weight_path = outputDir+"{}_weights.best.pth".format("cxr_reg")


# Convert numpy arrays to torch tensors
# TODO: Evaluate pertinence of float32
x_train_tens = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)/x_train.max()
x_val_tens = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)/x_train.max()
y_train_tens = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)/y_train.max()
y_val_tens = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)/y_train.max()

print(f"x_train_tens shape {x_train_tens.shape}")
print(f"y_train_tens shape {y_train_tens.shape}")
print(f"x_val_tens shape {x_val_tens.shape}")
print(f"y_val_tens shape {y_val_tens.shape}")

with open(outputDir + "max_val.txt", "w") as f:
    f.write(f"x_train.max(): {x_train.max()}\n")
    f.write(f"y_train.max(): {y_train.max()}\n")

print("------------- Data Loaded -------------")


# Define batch size depending on device
batch_size = 64 if device.type == "cuda" else 12

train_loader = DataLoader(TensorDataset(x_train_tens, y_train_tens), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_tens, y_val_tens), batch_size=batch_size, shuffle=False)

for x_batch, y_batch in train_loader:
    print("x_batch shape:", x_batch.shape)  # should be [batch_size, 2, 80, 80]
    print("y_batch shape:", y_batch.shape)  # should be [batch_size, 1, 80, 80] or [batch_size, 80, 80]
    break

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
min_lr = 1e-7
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',          # val_loss should decrease
    factor=0.1,          # reduce LR by 10x when stuck
    patience=5,          # wait 5 epochs with no improvement
    threshold=1e-6,      # minimum change to count as improvement
    threshold_mode='rel',# relative threshold (default)
    cooldown=0,          # no cooldown (fine for most cases)
    min_lr=min_lr,      # lower bound on LR
    eps=1e-8,            # minimal LR change (default)
)
criterion = unet.mse_loss

with open(outputDir + "ML_setup.txt", "w") as f:
    f.write(f"lr: {optimizer.param_groups[0]['lr']}\n")
    f.write(f"momentum: {optimizer.param_groups[0]['momentum']}\n")
    f.write(f"weight_decay: {optimizer.param_groups[0]['weight_decay']}\n")
    f.write(f"mode: {scheduler.mode}\n")
    f.write(f"factor: {scheduler.factor}\n")
    f.write(f"patience: {scheduler.patience}\n")
    f.write(f"threshold: {scheduler.threshold}\n")
    f.write(f"threshold_mode: {scheduler.threshold_mode}\n")
    f.write(f"cooldown: {scheduler.cooldown}\n")
    f.write(f"min_lr: {min_lr}\n")
    f.write(f"eps: {scheduler.eps}\n")


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
    # history["binary_accuracy"].append(train_acc)      # useless
    # history["val_binary_accuracy"].append(val_acc)    # useless

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