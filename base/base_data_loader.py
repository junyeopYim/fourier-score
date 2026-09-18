"""Resumable step-based batch stream (source mnist_compare.run.BatchStream)."""
import torch


class BatchStream:
    def __init__(self, indices, batch_size, seed):
        self.indices = indices.cpu().long()
        self.batch_size = batch_size
        if not 1 <= batch_size <= len(indices):
            raise ValueError('batch_size must not exceed training split size')
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(indices), generator=self.generator)
        self.cursor = 0

    def next(self):
        # Deliberately drop the incomplete tail, matching the source runner.
        if self.cursor + self.batch_size > len(self.indices):
            self.order = torch.randperm(len(self.indices), generator=self.generator)
            self.cursor = 0
        ids = self.indices[self.order[self.cursor:self.cursor + self.batch_size]]
        self.cursor += self.batch_size
        return ids

    def state_dict(self):
        return dict(order=self.order.clone(), cursor=self.cursor,
                    rng=self.generator.get_state(), batch_size=self.batch_size)

    def load_state_dict(self, state):
        order = state['order'].cpu().long()
        if state['batch_size'] != self.batch_size or not torch.equal(order.sort().values, torch.arange(len(self.indices))):
            raise ValueError('Invalid saved stream permutation or batch size')
        if not 0 <= state['cursor'] <= len(self.indices):
            raise ValueError('Invalid stream cursor')
        self.order, self.cursor = order, state['cursor']
        self.generator.set_state(state['rng'].cpu())
