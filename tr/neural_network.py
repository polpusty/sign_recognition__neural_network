import torch
from torch import nn
from torch.utils import data
from torchvision import transforms

from tr.transforms import BytesToNdarray


class Network(nn.Module):
    """
    :type layers_list: list[nn.layers.*]
    :type classes_list: list[dict]
    :type size_input_image: (int, int)
    :type optimizer: nn.optimizers.Optimizer
    """

    def __init__(self, layers_list, classes_list, size_input_image, optimizer):
        super(Network, self).__init__()
        self.layers = nn.Sequential(*layers_list)
        self.classes_list = classes_list
        self.size_input_image = size_input_image
        self.optimizer = optimizer(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_data):
        return self.layers(input_data)

    def get_perfect_output_for_class(self, number_class):
        result = torch.zeros(len(self.classes_list))
        result[number_class] = 1.0
        return result.to(torch.long)

    def transform_input_data(self, input_data):
        transform = transforms.Compose([
            BytesToNdarray(self.size_input_image),
            transforms.ToTensor()
        ])
        preparing_data = []
        for image, class_code in input_data:
            preparing_data.append((transform(image),
                                   self.classes_list.index(class_code)))
        return preparing_data

    def fit(self, input_data, number_epochs, batch_len):
        input_data = self.transform_input_data(input_data)
        data_loader = data.DataLoader(input_data, batch_size=batch_len, shuffle=True, num_workers=4)
        for epoch in range(number_epochs):
            running_loss = 0.0
            for i, input_data in enumerate(data_loader, 0):
                inputs, labels = input_data
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2 == 1:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2))
                    running_loss = 0.0

    def test(self, input_data, batch_len):
        data_loader = data.DataLoader(input_data, batch_size=batch_len, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        with torch.no_grad():
            for input_data in data_loader:
                images, labels = input_data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
