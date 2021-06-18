import torch
from random import sample


class CrossEntropy(torch.nn.Module):
    def __init__(self, use_hard_label=True):
        super(CrossEntropy, self).__init__()
        self.use_hard_label = use_hard_label
        if self.use_hard_label:
            self.obj = torch.nn.CrossEntropyLoss(reduction='sum')
        else:
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, predict, target):
        return self.obj(predict, target.argmax(dim=1)) if self.use_hard_label \
            else -torch.sum(torch.sum(target*self.log_softmax(predict), dim=1))


class InfoNCE(torch.nn.Module):
    def __init__(self, loss_config):
        super(InfoNCE, self).__init__()
        self.T = loss_config['temperature']
        self.obj = torch.nn.CrossEntropyLoss()

    def forward(self, q, k, queue):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        loss = self.obj(logits, labels)

        return loss


class NCELoss(torch.nn.Module):
    def __init__(self, loss_config):
        super(NCELoss, self).__init__()
        self.T = loss_config['temperature']
        self.obj = torch.nn.CrossEntropyLoss(reduction='sum')
        if 'cluster' in loss_config:
            self.num_cluster = loss_config['cluster']['num_cluster']
            self.num_neg_proto = loss_config['cluster']['num_neg_proto']

    def _compute_info_loss(self, q, k, queue=None):
        labels = torch.cat([torch.arange(len(q)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(q.device)

        features = torch.cat([q, k], dim=0)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(q.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        l_pos = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        l_neg_batch = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        if queue is not None:
            # negative logits: N x queue_size
            l_neg_queue = torch.einsum(
                'nc,ck->nk', [q, queue.clone().detach()]).repeat(2, 1)
            # logits: N x (1+K+queue_size)
            logits = torch.cat([l_pos, l_neg_batch, l_neg_queue], dim=1)
        else:
            # logits: N x (1+K)
            logits = torch.cat([l_pos, l_neg_batch], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        info_loss = self.obj(logits, labels) / 2
        return info_loss

    def _compute_proto_loss(self, q, cluster_result, index):
        proto_labels = []
        proto_logits = []

        for n, (emb2cluster, prototypes, density) in enumerate(zip(cluster_result['emb2cluster'], cluster_result['centroids'], cluster_result['density'])):
            # get positive prototypes
            pos_proto_id = emb2cluster[index.tolist()]
            pos_prototypes = prototypes[pos_proto_id]

            # sample negative prototypes
            all_proto_id = [i for i in range(emb2cluster.max())]
            neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
            # sample r negative prototypes
            neg_proto_id = sample(neg_proto_id, self.num_neg_proto)
            neg_prototypes = prototypes[neg_proto_id]

            proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

            # compute prototypical logits
            logits_proto = torch.mm(q, proto_selected.t())

            # targets for prototype assignment
            labels_proto = torch.linspace(
                0, q.size(0)-1, steps=q.size(0)).long().to(q.device)

            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat(
                [pos_proto_id, torch.LongTensor(neg_proto_id).to(q.device)], dim=0)]
            logits_proto /= temp_proto

            proto_logits.append(logits_proto)
            proto_labels.append(labels_proto)

        proto_loss = 0
        for proto_out, proto_target in zip(proto_logits, proto_labels):
            proto_loss += self.obj(proto_out, proto_target)

        # average loss across all sets of prototypes
        proto_loss /= len(self.num_cluster)
        return proto_loss

    def forward(self, q, k, queue, cluster_result=None, index=None):
        loss = self._compute_info_loss(q, k, queue)
        if cluster_result is not None:
            loss += self._compute_proto_loss(q, cluster_result, index)
        return loss
