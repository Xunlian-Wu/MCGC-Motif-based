import argparse


def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnnlayers', type=int, default=2, help="Number of gnn layers")
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--tao', type=str, default=0.1, help='tao')
    parser.add_argument('--lamda1', type=str, default=0.1, help='lamda1')
    parser.add_argument('--lamda2', type=str, default=0.1, help='lamda2')
    parser.add_argument('--seeds', type=str, default=[0], help='seeds')

    args = parser.parse_args()

    return args
