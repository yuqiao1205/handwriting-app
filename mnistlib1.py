import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils
import matplotlib.pyplot as plt

#train_data=datasets.MNIST(root='./MNIST',train=True,download=True,transform=transforms.ToTensor())
batch_size=256
# convert the image to tensor,and normalize it
transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
# Create a random 1/10th size subset of the data
# one_tenth = len(train_data) // 10
# train_data, _ = torch.utils.data.random_split(train_data, [one_tenth, len(train_data) - one_tenth])


# download dataset
train_set = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)

# load dataset
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)

#test_data=datasets.MNIST(root='./MNIST',train=False,download=True,transform=transforms.ToTensor())
test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

# for X, y in test_loader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# design model
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
           #  64*4*4 is the size of the feature map before the fully connected layer
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        # After the feature map is obtained, it is "flattened" into a 1D tensor.
        # The .view method is used to reshape the feature map.
        # img.shape[0] represents the batch size, and -1 indicates that the remaining dimensions should be
        # automatically inferred to ensure the same number of elements.
        output = self.fc(feature.view(img.shape[0], -1))
        return output
model = MyNet()
print(model)
# output of the structure of the model
# MyNet(
#   (conv): Sequential(
#     (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
#     (4): ReLU()
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (fc): Sequential(
#     (0): Linear(in_features=1024, out_features=128, bias=True)
#     (1): ReLU()
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=128, out_features=64, bias=True)
#     (4): ReLU()
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=64, out_features=10, bias=True)
#   )
# )

# nn.CrossEntropyLoss combines the softmax activation and the cross-entropy loss into a single step.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Define Training CNN
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    # Initialize variables to track the total number of samples, correct predictions, and epoch loss
    total = 0
    correct = 0.0
    epoch_loss = 0.0
    # Iterate through batches in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        # Calculate the loss between the predicted values and the actual labels
        loss = loss_fn(pred, y)
        # Accumulate the loss for the entire epoch
        epoch_loss += loss.item()
        # Get the predicted class by finding the index of the maximum value along the second dimension
        predicted=pred.argmax(dim=1)
        # Update total number of samples and correct predictions
        total += y.size(0)
        correct += (predicted==y).sum().item()
        # Backpropagation: Zero the gradients, perform backward pass, and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress for every K batches (in this case, 1/10th of the dataset)
        L = len(dataloader)
        K = L // 10
        if batch % K == 0:
            accuracy = 100 * correct / total
            print(f"train loss:{loss.item():.3f}", end=" ")
            print(f"train accuracy:{accuracy:.2f}%")
    # Calculate the average loss and accuracy for the entire epoch
    epoch_loss /= len(dataloader)
    return epoch_loss, accuracy

# Testing CNN and calculate the accuracy and save the best model
def test(testloader,model,loss_fn):
    model.eval()
    # Initialize variables to track the test loss, correct predictions, and total samples
    test_loss=0
    correct=0
    total=0
    with torch.no_grad():
        # Iterate through batches in the testloader.(X is the image and y is the label)
        for X,y in testloader:
            # Compute predictions using the model
            pred=model(X)
            # Calculate the loss between the predicted values and the actual labels
            test_loss+=loss_fn(pred,y).item()
            # Get the predicted class by finding the index of the maximum value along the second dimension
            predicted=pred.argmax(dim=1)

            # Update total number of samples and correct predictions
            total+=y.size(0)
            correct+=(predicted==y).sum().item()

        print("test loss:{:.3f},test accuracy:{:.2f}%".format(test_loss/total,100*correct/total))

  # Calculate the average loss and accuracy for the entire test set
    test_loss /= len(testloader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

# Train the model for 10 epochs and save the best model with the highest accuracy  and plot the loss and accuracy curves
epochs = 10
max_accuracy = 0.0

lines = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_loader, model, loss_fn)

    if test_acc > max_accuracy:
        print("best model accuracy: ", test_acc, "previous best accuracy: ", max_accuracy)
        print("Save best model")
        max_accuracy = test_acc

        torch.save(model.state_dict(), 'best_lenet.pth')
    lines["train_loss"].append(train_loss)
    lines["train_acc"].append(train_acc)
    lines["test_loss"].append(test_loss)
    lines["test_acc"].append(test_acc)
print("Done!")

# Plot the loss and accuracy curves
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# we want 4 different colors so
for index, (key, line) in enumerate(lines.items()):
    color = plt.cm.tab10(index)
    ax = ax1 if key.endswith("loss") else ax2
    ax.plot(line, label=key, color=color)

# combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")

plt.xlim(0, 9)
plt.savefig("mnist_lenet.png")
plt.show()

#torch.save(model.state_dict(), 'model_lenet.pth')