import torch
import torch.nn as nn
import torchvision
from enum import Enum

from resnet18_1d import resnet18_1d


class InputType(Enum):
    HISTOGRAM = "histogram"
    IMAGES = "images"


class Concatenation(torch.nn.Module):
    def __init__(self):
        super(Concatenation, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], 1)



def build_embedding_network(
        input_type: InputType,
        embedding_dimension: int
):
    if input_type == InputType.IMAGES:
        embedding_network = torchvision.models.resnet18()
    elif input_type == InputType.HISTOGRAM:
        embedding_network = resnet18_1d()
    else:
        raise NotImplementedError(f"Embedding network for input type {input_type} not implemented")

    if input_type == InputType.IMAGES or input_type == InputType.HISTOGRAM:
        fc_in_features = embedding_network.fc.in_features
        embedding_network = nn.Sequential(*(list(embedding_network.children())[:-1]))
        embedding_network = nn.Sequential(
            embedding_network,
            torch.nn.Flatten(),
            torch.nn.Linear(fc_in_features, embedding_dimension)
        )

    return embedding_network


class SiameseNetwork(nn.Module):
    def __init__(self,
                 input_type: InputType,
                 embedding_dimension: int
                 ):
        super(SiameseNetwork, self).__init__()

        self.kwargs = {
            'input_type': input_type,
            'embedding_dimension': embedding_dimension
        }

        self.embedding = build_embedding_network(
            input_type,
            embedding_dimension
        )

        # initialize the weights
        self.embedding.apply(self.init_weights)

        # build head
        self.merge_layer = Concatenation()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dimension, embedding_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, 1),
            nn.Flatten()
        )
        self.sigmoid = nn.Sigmoid()

        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.embedding(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.merge_layer:
            outputs = self.merge_layer(output1, output2)

            outputs = self.fc.forward(outputs)

            outputs = self.sigmoid(outputs)

            return outputs.view(outputs.size()[0])

        return output1, output2

    @property
    def embedding_network(self):
        return self.embedding

    @property
    def args(self):
        return self.kwargs