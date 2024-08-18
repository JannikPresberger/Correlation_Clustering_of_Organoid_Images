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
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self,
                 input_type: InputType,
                 embedding_dimension: int,
                 return_penultimate: bool = False
                 ):
        super(SiameseNetwork, self).__init__()

        self.return_penultimate = return_penultimate

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

        self.fc = None
        self.merge_layer = None
        self.sigmoid = None

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

            penultimate = self.fc.forward(outputs)

            outputs = self.sigmoid(penultimate)

            if self.return_penultimate:
                return outputs.view(outputs.size()[0]), penultimate
            else:
                return outputs.view(outputs.size()[0])

        return output1, output2

    @property
    def embedding_network(self):
        return self.embedding

    @property
    def args(self):
        return self.kwargs