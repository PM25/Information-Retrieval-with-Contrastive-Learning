import copy
import torch
import torch.nn as nn


class RetrievalModelWrapper(nn.Module):
    def __init__(self, base_encoder, criterion, loss_config,
                 use_momentum=True, use_queue=True, use_LSTM=True):
        super(RetrievalModelWrapper, self).__init__()
        self.criterion = criterion
        self.loss_config = loss_config
        self.use_momentum = self.loss_config['use_momentum']
        self.use_queue = self.loss_config['use_queue']
        self.use_LSTM = use_LSTM
    
        self.encoder_q = copy.deepcopy(base_encoder)
        if self.use_momentum:
            self.encoder_k = copy.deepcopy(base_encoder)
            # set encoder initialization and requires_grad
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        if self.use_queue:
            # create the queue
            self.register_buffer("queue", torch.randn(
                self.loss_config['dim'], self.loss_config['queue_size']))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.add_queue_to_loss = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * \
                self.loss_config['momentum'] + param_q.data * \
                (1. - self.loss_config['momentum'])
        if self.use_LSTM:
            self.encoder_k.lstm.flatten_parameters()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        if self.loss_config['queue_size'] % batch_size == 0:
            ptr = int(self.queue_ptr)

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            
            # move pointer
            ptr = (ptr + batch_size) % self.loss_config['queue_size']

            self.queue_ptr[0] = ptr

    def forward(self, anchor_sample, positive_sample, cluster_result=None, indexes=None):
        # compute query features
        emb_q = self.seq2vec(anchor_sample)
        if self.use_LSTM:
            self.encoder_q.lstm.flatten_parameters()
        # compute key features
        emb_k = self.seq2vec(
            positive_sample, query=False) if self.use_momentum else self.seq2vec(positive_sample)  # keys: NxC

        queue = None if not self.use_queue or not self.add_queue_to_loss else self.queue
        loss = self.criterion(emb_q, emb_k, queue,
                                  cluster_result, indexes)

        # update key queue
        if self.use_queue and self.training:
            with torch.no_grad():
                self._dequeue_and_enqueue(emb_k.detach())

        return loss

    def seq2vec(self, seq, query=True):
        assert seq.ndim == 3

        # compute embedding feature
        if query:
            emb = self.encoder_q(features=seq).mean(dim=1)
        else:
            with torch.no_grad():
                emb = self.encoder_k(features=seq).mean(dim=1)
        emb = nn.functional.normalize(emb)
        return emb