import numpy as np
import torch

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils

batch_size = 256

# train_data=datasets.MNIST(root='./MNIST',train=True,download=True,transform=transforms.ToTensor())
# train_loader=DataLoader(train_data,batch_size= batch_size,shuffle=True)
# test_data=datasets.MNIST(root='./MNIST',train=False,download=True,transform=transforms.ToTensor())
# test_loader=DataLoader(test_data,batch_size= batch_size,shuffle=False)

# convert the image to tensor, and normalize it
transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])

# Create a random 1/10th size subset of the data
# one_tenth = len(train_data) // 10
# train_data, _ = torch.utils.data.random_split(train_data, [one_tenth, len(train_data) - one_tenth])


# download dataset
train_set = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)

# load dataset
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

# display shape of data
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
        output = self.fc(feature.view(img.shape[0], -1))
        return output
model = MyNet()

# load model
model.load_state_dict(torch.load('best_lenet.pth'))

# takes an image, processes it, passes it through our pretrained model, and returns the predicted class.
def infer(image: np.ndarray):
    image = torch.from_numpy(image).type(torch.float32)
    image = image.unsqueeze(0)
    # The reshaped image is then passed through a pre-trained deep learning model (referred to as model),
    # presumably a neural network, for inference. The model makes predictions based on the input image.
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Predictions on Test Data
if __name__ == "__main__":
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in range(20):
        X, y = test_set[i][0], test_set[i][1]
        X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False)
        with torch.no_grad():
            y_pred = model(X)
            predicted, actual = classes[torch.argmax(y_pred[0])], classes[y]
            print('Predicted: ', {predicted}, 'Actual: ', {actual})

    # Predicted:  {'7'} Actual:  {'7'}
    # Predicted:  {'2'} Actual:  {'2'}
    # Predicted:  {'1'} Actual:  {'1'}
    # Predicted:  {'0'} Actual:  {'0'}
    # Predicted:  {'4'} Actual:  {'4'}
    # Predicted:  {'1'} Actual:  {'1'}
    # Predicted:  {'4'} Actual:  {'4'}
    # Predicted:  {'9'} Actual:  {'9'}
    # Predicted:  {'5'} Actual:  {'5'}
    # Predicted:  {'9'} Actual:  {'9'}
    # Predicted:  {'0'} Actual:  {'0'}
    # Predicted:  {'6'} Actual:  {'6'}
    # Predicted:  {'9'} Actual:  {'9'}
    # Predicted:  {'0'} Actual:  {'0'}
    # Predicted:  {'1'} Actual:  {'1'}
    # Predicted:  {'5'} Actual:  {'5'}
    # Predicted:  {'9'} Actual:  {'9'}
    # Predicted:  {'7'} Actual:  {'7'}
    # Predicted:  {'3'} Actual:  {'3'}
    # Predicted:  {'4'} Actual:  {'4'}


    for i in range(10):
        image, label = test_set[i]
        prediction = infer(image.numpy())
        print(f"Prediction: {prediction}, Label: {label}")

    # Prediction: 7, Label: 7
    # Prediction: 2, Label: 2
    # Prediction: 1, Label: 1
    # Prediction: 0, Label: 0
    # Prediction: 4, Label: 4
    # Prediction: 1, Label: 1
    # Prediction: 4, Label: 4
    # Prediction: 9, Label: 9
    # Prediction: 5, Label: 5
    # Prediction: 9, Label: 9
    # Prediction: 2, Label: 2

    #test on first image in test set
    image, label = test_set[1]
    prediction = infer(image.numpy())
    print(f"Prediction: {prediction}, Label: {label}")

    #print out arrays of first 10 images in test set
    for i in range(2):
        image, label = train_set[i]
        print(f"Image {i}:")
        print((image.numpy() * 255).astype(np.uint8))
        print(f"Label: {label}")
        print()


