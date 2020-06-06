import Net
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import utils.mnist_reader as reader

if __name__ == "__main__":
    train = torchvision.datasets.FashionMNIST('', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test = torchvision.datasets.FashionMNIST('', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    net = Net.Net()
    optimizer = optim.ASGD(net.parameters(), lr=0.01)

    for epoch in range(3):
        for data in trainset:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 784))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = net(X.view(-1, 784))
            # print(output)
            for idx, i in enumerate(output):
                # print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct / total, 3))