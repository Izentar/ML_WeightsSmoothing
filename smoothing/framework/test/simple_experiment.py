import torch
import torchvision
import torchvision.transforms as transforms
import os, sys
from os.path import expanduser


def test():
    

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(expanduser("~"), '.data'), train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(expanduser("~"), '.data'), train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    net = models.resnext50_32x4d()
    modelMetadata = dc.DefaultModel_Metadata(device='cuda:0', lossFuncDataDict={}, optimizerDataDict=None)
    net = dc.DefaultModelPredef(obj=net, modelMetadata=modelMetadata, name="OK")

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.getNNModelModule().parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        print(torch.cuda.memory_summary(device='cuda:0'))
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device='cuda:0'), labels.to(device='cuda:0')
            weights = net.getNNModelModule().state_dict()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.getNNModelModule()(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1))
            running_loss = 0.0
            print(torch.cuda.memory_summary(device='cuda:0'))
            sys.stdout.flush()

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.getNNModelModule().state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    outputs = net.getNNModelModule()(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network 
            outputs = net.getNNModelModule()(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data    
            outputs = net.getNNModelModule()(images)    
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                    accuracy))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)


if(__name__ == '__main__'):
    test()