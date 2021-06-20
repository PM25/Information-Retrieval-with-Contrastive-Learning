from tqdm import tqdm

import faiss
import torch
import fastcluster

import numpy as np
import scipy.cluster.hierarchy as sch


def extract_all_emb(loader, model, device):
    emb_lst = []
    print('[Runner] - Extracting sentence embedding vectors')
    with torch.no_grad():
        for batch in tqdm(loader):
            indexes, anchor_sample, positive_sample = batch
            anchor_sample, positive_sample = model.bert_extract(anchor_sample, positive_sample, device)
            emb = model.seq2vec(anchor_sample.to(device)).cpu().numpy()
            emb_lst.append(emb)
            emb = model.seq2vec(positive_sample.to(device)).cpu().numpy()
            emb_lst.append(emb)

    torch.cuda.empty_cache()
    emb_lst = np.vstack(emb_lst)
    return emb_lst


def get_cluster(d, k, clus_config, seed):
    clus = faiss.Clustering(d, k)
    clus.verbose = clus_config['verbose']
    clus.niter = clus_config['niter']
    clus.nredo = clus_config['nredo']
    clus.seed = seed
    clus.max_points_per_centroid = clus_config['max_points_per_centroid']
    clus.min_points_per_centroid = clus_config['min_points_per_centroid']
    return clus


def get_clus_idx(d, gpu_id):
    res = faiss.StandardGpuResources()

    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id

    index = faiss.GpuIndexFlatL2(res, d, cfg)
    return index


def run_kmeans(proto_nce_config, loader, model, device):
    x = extract_all_emb(loader, model, device)
    print('[Runner] - Performing kmeans clustering')
    results = {'emb2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(proto_nce_config['cluster']['num_cluster']):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        gpu_id = torch.cuda.current_device()
        clus = get_cluster(d, k, proto_nce_config['cluster'], seed)
        index = get_clus_idx(d, gpu_id)

        # train cluster
        clus.train(x, index)

        # for each sample, find cluster distance and assignments
        D, I = index.search(x, 1)
        emb2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for nis, i in enumerate(emb2cluster):
            Dcluster[i].append(D[nis][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(
            density, 90))  # clamp extreme values for stability
        density = proto_nce_config['temperature'] * \
            density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        emb2cluster = torch.LongTensor(emb2cluster).cuda()
        density = torch.Tensor(density).cuda()
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['emb2cluster'].append(emb2cluster)
    return results


def run_hierarchical_clustering(proto_nce_config, loader, model, device):
    x = extract_all_emb(loader, model, device)
    print('[Runner] - Performing hierarchical clustering')
    results = {'emb2cluster': [], 'centroids': [], 'density': []}
    dis = fastcluster.linkage(x, metric='euclidean', method='ward')

    for seed, num_cluster in enumerate(proto_nce_config['cluster']['num_cluster']):
        # intialize faiss clustering parameters

        emb2cluster = sch.fcluster(dis, num_cluster, criterion = 'maxclust') - 1
        cls_lst = [[] for c in range(num_cluster)]
        for i, cls in enumerate(emb2cluster):
            cls_lst[cls].append(x[i])
        
        # centroids and sample-to-centroid distances for each cluster
        centroids = []
        Dcluster = []
        for vecs in cls_lst:
            vecs = np.array(vecs)
            centroid = vecs.mean(axis=0)
            centroids.append(centroid)
            Dcluster.append(list(np.sum((vecs - centroid)**2, axis=1)))
        centroids = np.array(centroids)
        
        # concentration estimation (phi)
        density = np.zeros(num_cluster)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(
            density, 90))  # clamp extreme values for stability
        density = proto_nce_config['temperature'] * \
            density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.from_numpy(centroids).cuda()
        centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        emb2cluster = torch.LongTensor(emb2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['emb2cluster'].append(emb2cluster)
    return results