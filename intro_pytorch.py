import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    # Data in set is 28x28 image, associated with a label from 10 classes
    # Contains image inputs as 2D arrays and labels
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # This sets the location where the dataset is downloaded
    # It also retrives the images and lables for testing/training
    if (training == False):
        test_set=datasets.FashionMNIST("./data", train=False,transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size = 64)
        return loader

    else:
        train_set=datasets.FashionMNIST("./data",train=True,download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(), # This creates a 784 1D array
        nn.Linear(784, 128), # Hidden Layer 1
        nn.ReLU(), # Activation function for Hidden Layer 1
        nn.Linear(128, 64), # Hidden Layer 2
        nn.ReLU(), # Activation Function for Hidden Layer 2
        nn.Linear(64, 10) # Output (will use softmax later)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    # Set optimizer and set model to train
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    # Val represents the number of epochs
    val = 0
    # Outer loop iteratoes over epochs, inner loop iterates over batches of (images, labels)
    # pairs from the train_loader dataset
    for epoch in range(T):
        running_loss = 0.0
        correctAmount = 0
        correctSamples = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs (images, labels)
            images, labels = data
            # zero the parameter gradients, optimize
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            # Find the running loss and accuracy items needed for calculation after loop ends
            running_loss +=  loss.item()
            _, predicted = torch.max(outputs, 1)
            correctAmount += (predicted == labels).sum().item()
            correctSamples += labels.size(0)

        accuracy = 100 * correctAmount / correctSamples
        running_loss = running_loss / len(train_loader)
        tCorrectAmount = str(correctAmount)
        tCorrectSamples = str(correctSamples)
        tVal = str(val)
        print("Train Epoch: " + tVal + " Accuracy: " + tCorrectAmount + "/" + tCorrectSamples + "(", end="")
        print(f'{accuracy:.2f}%', end="") 
        print(") Loss: ", end="")
        print(f'{running_loss:.3f}')
        val = val + 1


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    # Set an optimizer, set model to evaluate mode
    # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()
    # disable tracking gradients during testing
    with torch.no_grad():
        correctAmount = 0
        correctSamples = 0
        running_loss = 0.0
        for data, labels in test_loader:
                # opt.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                # opt.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correctAmount += (predicted == labels).sum().item()
                correctSamples += labels.size(0)
        
        running_loss = running_loss / len(test_loader)
        accuracy = 100 * correctAmount / correctSamples
        if (show_loss == True):
            print("Average loss: ", end = "")
            print(f'{running_loss:.4f}')
        print("Accuracy: ", end = "")
        print(f'{accuracy:.2f}%')


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    # Attain the class labels and specified image to predict the label of
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot'] 
    image = test_images[index]
    # Attain the logits 
    outputs = model(image)
    # Convert output of final Dense layer into a probability
    prob = F.softmax(outputs, dim=1)
    prob_percentage = prob * 100
    # Find the top 3 probabilities from the output (and their indices)
    top_3, top_indx = torch.topk(prob, 3)
    # Access the labels in class_names list that correspond to the indices found
    topLabels = [class_names[i] for i in top_indx[0]]
    prob_top3 = top_3 * 100
    # Access the top3 probabilities in the top_3 list, print them with their corresponding labels
    prob1 = prob_top3[0][0].item()
    prob2 = prob_top3[0][1].item()
    prob3 = prob_top3[0][2].item()
    print(topLabels[0] + ": ", end = "")
    print(f'{prob1:.2f}%')
    print(topLabels[1] + ": ", end = "")
    print(f'{prob2:.2f}%')
    print(topLabels[2] + ": ", end = "")
    print(f'{prob3:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # This is a test for get_data_loader()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)
    # This is a test for build_model()
    model = build_model()
    print(model)
    # These are tests for train model and evaluate model
    criterion = nn.CrossEntropyLoss()
    T = 5
    train_model(model, train_loader, criterion, T)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    evaluate_model(model, test_loader, criterion, show_loss=False)
    # This is a test for predict label
    test_images = next(iter(test_loader))[0]
    index = 1
    predict_label(model, test_images, index)
