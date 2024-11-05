import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from kmeans_gpu import kmeans


class my_model(nn.Module):
    def __init__(self, dims, n_clusters, name):
        super(my_model, self).__init__()
        self.name = name
        if self.name == "bat":
            self.layers1 = nn.Linear(dims[0], 1024)
            self.layers3 = nn.Linear(1024, dims[1])

            self.layers2 = nn.Linear(dims[0], 1024)
            self.layers4 = nn.Linear(1024, dims[1])
        else:
            self.layers1 = nn.Linear(dims[0], dims[1])
            self.layers2 = nn.Linear(dims[0], dims[1])

        self.layers = nn.Linear(dims[1], n_clusters)

    def forward(self, x, xw, n_clusters):
        if self.name == "bat":
            out1 = self.layers1(x)
            out1 = F.relu(self.layers3(out1))

            out2 = self.layers2(xw)
            out2 = F.relu(self.layers4(out2))

        else:
            out1 = self.layers1(x)
            out2 = self.layers2(xw)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        km_label, km_centers = kmeans(X=out2, num_clusters=n_clusters, distance="euclidean", device="cuda")
        km_label, km_centers = km_label.detach().numpy(), km_centers.detach().numpy()

        u1 = F.softmax(self.layers(out1), dim=1)
        u2 = F.softmax(self.layers(out2), dim=1)

        return out1, out2, km_label, km_centers, u1, u2
