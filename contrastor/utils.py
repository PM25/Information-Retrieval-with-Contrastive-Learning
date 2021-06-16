from tqdm import tqdm

import faiss
import torch
import fastcluster

import numpy as np
import scipy.cluster.hierarchy as sch


def extract_all_emb(loader, model, device):
    emb_lst = []
    print('[Runner] - Extracting noise embedding vectors')

    with torch.no_grad():
        for batch in tqdm(loader):
            indexes, lengths, noisy_wavs, noise_q, noise_k = batch
            emb = model.noise2vec(noise_q.to(device)).cpu().numpy()
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
    results = {'noise2cluster': [], 'centroids': [], 'density': []}

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
        noise2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for nis, i in enumerate(noise2cluster):
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

        noise2cluster = torch.LongTensor(noise2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['noise2cluster'].append(noise2cluster)

    return results


def run_hierarchical_clustering(proto_nce_config, loader, model, device):
    x = extract_all_emb(loader, model, device)
    print('[Runner] - Performing hierarchical clustering')
    results = {'noise2cluster': [], 'centroids': [], 'density': []}
    dis = fastcluster.linkage(x, metric='euclidean', method='ward')

    for seed, num_cluster in enumerate(proto_nce_config['cluster']['num_cluster']):
        # intialize faiss clustering parameters

        noise2cluster = sch.fcluster(dis, num_cluster, criterion = 'maxclust') - 1
        cls_lst = [[] for c in range(num_cluster)]
        for i, cls in enumerate(noise2cluster):
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

        noise2cluster = torch.LongTensor(noise2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['noise2cluster'].append(noise2cluster)
    return results


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(
            non_blocking=True) / (args.world_size * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor
