import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
import torch.nn.functional as F
import motifcluster.motifadjacency as motif
from params import cora_params


def train():
    args = cora_params()
    print("Using {} dataset".format(args.dataset))
    file = open("result/baseline.txt", "a+")
    print(args.dataset, file=file)
    file.close()

    X, y, A = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    Wm = motif.build_motif_adjacency_matrix(adj, "M3", "func", "mean")
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    Wm = Wm - sp.dia_matrix((Wm.diagonal()[np.newaxis, :], [0]), shape=Wm.shape)
    adj.eliminate_zeros()
    Wm.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    Wm_norm_s = preprocess_graph(Wm, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s_wm = sp.csr_matrix(features).toarray()

    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    pathw = "dataset/{}/{}_feat_smw_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
        sm_fea_s_wm = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        np.save(path, sm_fea_s, allow_pickle=True)
        for a in Wm_norm_s:
            sm_fea_s_wm = a.dot(sm_fea_s_wm)
        np.save(pathw, sm_fea_s_wm, allow_pickle=True)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    sm_fea_s_wm = torch.FloatTensor(sm_fea_s_wm)

    adj_wst = (adj + Wm + sp.eye(adj.shape[0])).toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in args.seeds:
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        model = my_model([features.shape[1]] + args.dims, args.cluster_num, args.dataset)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)
        inx = sm_fea_s.to(args.device)
        inxw = sm_fea_s_wm.to(args.device)
        target = torch.FloatTensor(adj_wst).to(args.device)
        print('Start Training...')

        for epoch in tqdm(range(args.epochs)):
            model.train()

            h1, h2, km_labels, km_centers, u1, u2 = model(inx, inxw, args.cluster_num)

            S = h1 @ h2.T

            centers = torch.FloatTensor(km_centers).to(args.device)
            label = torch.LongTensor(km_labels).to(args.device)

            loss1 = F.mse_loss(S, target)
            loss2 = NC_loss(args.cluster_num, h1, centers, label, h2, args.tao)
            loss3 = CC_loss(u1, u2, 0.5 * (h1 + h2), args.tao)

            loss = loss1 + args.lamda1 * loss2 + args.lamda2 * loss3
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()

                h1, h2, _, _, u1, u2 = model(inx, inxw, args.cluster_num)
                hidden_emb = (h1 + h2) / 2

                acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1

        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result/baseline.txt", "a+")
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result/baseline.txt", "a+")
    print(args.gnnlayers, args.lr, args.dims, file=file)
    print("ACC:", round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print("NMI:", round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print("ARI:", round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print("F1:", round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()
