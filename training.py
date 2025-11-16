import os
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

plots_size = 7
figsize = 8
fontsize = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self, num_categories): # Model architecture modified from documentations https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5) # [28, 28, 1] -> [24, 24, 8]
        self.pool1 = nn.MaxPool2d(2, 2) # [24, 24, 8] -> [12, 12, 8]
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3) # [12, 12, 8] -> [10, 10, 12]
        self.pool2 = nn.MaxPool2d(2, 2) # [10, 10, 12] -> [5, 5, 12]
        self.conv2_bn = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(5 * 5 * 12, 20*num_categories)
        self.dropout2 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(20*num_categories, num_categories) # As many outputs as there are categories

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = self.conv2_bn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def make_new_model(num_categories):
    model = NeuralNetwork(num_categories).to(device)
    return model


def load_model(filepath, num_categories):
    model = NeuralNetwork(num_categories).to(device)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    return model


def prepare_dataset_and_labels(batch_size):
    labels_dict = {}
    datas_array = []
    labels_array = []
    
    for i, fn in enumerate(os.listdir('processed_training_data')):
        data_tensor = torch.load(f"processed_training_data/{fn}") # shape is [N, 1, 28, 28], 1 is the color channels
        data_tensor = data_tensor / 255 # Normalize from 0-255 to 0-1
        
        
        label_tensor = torch.full((len(data_tensor),), i)
        
        datas_array.append(data_tensor)
        labels_array.append(label_tensor)
        
        # Get name of this file and add to a dictionary for later labelling
        under_score_index = fn.rfind('_')
        labels_dict[i] = fn[under_score_index+1:-3] # -3 removes the '.pt'
        
        print(f"Loaded {fn} - {data_tensor.shape}")

    X = torch.cat(datas_array, dim=0).to(device) # Combines data into one big tensor with shape like [index, pixels]
    y = torch.cat(labels_array, dim=0).to(device) # Combines labels into one big label tensor for each index with shape [index]

    my_dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(my_dataset, [0.95, 0.05])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    
    return train_dataset, val_dataset, train_dataloader, val_dataloader, labels_dict


def train_loop(dataloader, model, loss_fn, optimizer, batch_size): # Modified from pytorch's documentation
    size = len(dataloader.dataset)
    running_loss = 0
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss 
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % ((size//batch_size) // (10-1)) == 0: # there is (size//batch_size) batches in enumerate(dataloader)
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return running_loss / len(dataloader) # Average training loss


def test_loop(dataloader, model, loss_fn): # Modified from pytorch's documentation
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            predicted_label = torch.argmax(pred, dim=1)
            
            correct += torch.sum(predicted_label == y).item()

    test_loss /= num_batches
    accuracy = correct / size
    
    print(f"Test Error: \n Accuracy: {100*accuracy:>0.1f}%, Avg test loss: {test_loss:>8f}\n")
    return accuracy, test_loss


def train_and_save_model(model, num_epochs, learning_rate, momentum, batch_size, model_save_file_name, train_dataloader, val_dataloader, show_graph = True):
    # Training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    results = []
    for i in range(num_epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        avg_training_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        accuracy, test_loss = test_loop(val_dataloader, model, loss_fn)
        results.append((i, accuracy, test_loss, avg_training_loss))
    print("Done!")

    torch.save(model.state_dict(), model_save_file_name)
    
    
    # Performance graph
    if show_graph:
        epoch = [x[0] for x in results]
        accuracy = [x[1] for x in results]
        test_loss = [x[2] for x in results]
        avg_training_loss = [x[3] for x in results]

        _,(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

        ax1.plot(epoch, accuracy, color='lightcoral', label='Accuracy')
        ax1.legend()
        ax1.set_title("Accuracy vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")

        ax2.plot(epoch, test_loss, color='lightsteelblue', label='Validation Loss')
        ax2.plot(epoch, avg_training_loss, color='salmon', label='Training Loss')
        ax2.legend()
        ax2.set_title("Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")

        plt.tight_layout()
        plt.show()


def random_preview_train_dataset(train_dataset: torch.utils.data.Dataset, labels_dict):
    fig, axs = plt.subplots(plots_size, plots_size, figsize=(figsize, figsize))
    fig.tight_layout()
    for i in range(plots_size):
        for j in range(plots_size):
            random_index = random.randint(0, len(train_dataset))
            image, label = train_dataset[random_index]
            
            image = image.squeeze(0) # Removes the channel [-> 1 <-, 28, 28]
            axs[i, j].imshow(image.cpu())
            axs[i, j].set_title(labels_dict[label.item()], fontsize=fontsize)
            axs[i, j].axis("off")

    plt.subplots_adjust(right=0.95, top=0.9)
    plt.suptitle("Training Dataset Preview", fontsize=20)
    plt.show()


def random_preview_model(model, val_dataset, labels_dict, title):
    model.eval()
    
    fig, axs = plt.subplots(plots_size, plots_size, figsize=(figsize, figsize))
    fig.tight_layout()
    for i in range(plots_size):
        for j in range(plots_size):
            random_index = random.randint(0, len(val_dataset)-1)
            image, label = val_dataset[random_index]
            
            image = image.unsqueeze(0)
            
            prediction = torch.argmax(model(image)).item()
            
            color = 'green' if prediction == label.item() else 'red'
            
            image = image.squeeze(0) # Removes the channel [-> 1 <-, 1, 28, 28]
            image = image.squeeze(0) # Removes the channel [-> 1 <-, 28, 28]
            axs[i, j].imshow(image.cpu())
            
            axs[i, j].set_title(f"{labels_dict[prediction]}?\nA: {labels_dict[label.item()]}", fontsize=fontsize, color=color)
            axs[i, j].axis("off")

    plt.subplots_adjust(right=0.95, top=0.9)
    plt.suptitle(title, fontsize=20)
    plt.show()