import numpy as np

import torchimport numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size=408, seed=1, fc1_unit=300, fc2_unit=220, action_dim=40):
        """
        Initialize parameters and build model.
        Params
        state_size (int): Dimension of each state (and action)
        seed (int): Random seed
        fc1_unit (int): number of nodes in first hidden layer
        fc2_unit (int): number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_dim)

    def forward(self, x):
        """ Build a network that maps state -> action values. """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNET_CNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=11,
                 filter_sizes=[3, 4, 5],
                 num_filters=[16, 32, 64],
                 #filter_sizes=[3,4],
                 #num_filters=[16,32],
                 num_classes=40,
                 dropout=0.5,
                 seed=1):
        """
        The constructor for CNN_NLP class.50

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False

            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used.
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`.
            n_classes (int): Number of classes.
            dropout (float): Dropout rate.
        """

        super(QNET_CNN, self).__init__()
        # Embedding layer
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.seed = torch.manual_seed(seed)

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        if len(x.shape) == 2:
            x_embed = x.float().unsqueeze(0)
        else:
            x_embed = x.float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)  # size (1, 11, 50)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list] #

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits.squeeze(0)

"""
def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=100,
                    filter_sizes=[3, 4, 5],
                    num_filters=[50, 50, 50],
                    num_classes=178,
                    dropout=0.3,
                    learning_rate=0.005):


    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNET_CNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=11,
                 filter_sizes=[3, 4, 5],
                 num_filters=[16, 32, 64],
                 #filter_sizes=[3,4],
                 #num_filters=[16,32],
                 num_classes=40,
                 dropout=0.5,
                 seed=1):
        """
        The constructor for CNN_NLP class.50

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False

            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used.
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`.
            n_classes (int): Number of classes.
            dropout (float): Dropout rate.
        """

        super(QNET_CNN, self).__init__()
        # Embedding layer
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.seed = torch.manual_seed(seed)

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        if len(x.shape) == 2:
            x_embed = x.float().unsqueeze(0)
        else:
            x_embed = x.float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)  # size (1, 11, 50)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list] #

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits.squeeze(0)

"""
def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=100,
                    filter_sizes=[3, 4, 5],
                    num_filters=[50, 50, 50],
                    num_classes=178,
                    dropout=0.3,
                    learning_rate=0.005):


    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer
"""
