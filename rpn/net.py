import torch.nn as nn
from rpn.utils.eval_utils import binary_accuracy
from rpn.utils.config import dict_to_namedtuple


def compute_bce_loss(logits, labels, weight=None):
    loss_fn = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
    loss = loss_fn(logits, labels)
    return loss


def log_binary_accuracy_fpfn(preds, labels, summarizer, global_step, prefix):
    acc, fp, fn = binary_accuracy(preds, labels)
    for i, (fp, fn) in enumerate(zip(fp, fn)):
        summarizer.add_scalar(prefix + 'acc/%i/fp' % i, fp, global_step)
        summarizer.add_scalar(prefix + 'acc/%i/fn' % i, fn, global_step)
    summarizer.add_scalar(prefix + 'acc', acc, global_step)


def masked_symbolic_state_index(symbolic_state, mask):
    masked_state = (symbolic_state > 0.5).long().detach()
    masked_state.masked_fill_(mask < 0.5, 2)
    return masked_state


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        c = dict_to_namedtuple(kwargs, name='config')
        self._c = kwargs
        self.c = c
        self.policy_mode = False
        self.env = None
        self.verbose = kwargs.get('verbose', False)

    def forward_batch(self, batch):
        raise NotImplementedError

    @property
    def config(self):
        return self._c

    @staticmethod
    def log_losses(losses, summarizer, global_step, prefix):
        for name, loss in losses.items():
            summarizer.add_scalar(prefix + name, loss, global_step=global_step)

    @staticmethod
    def log_outputs(outputs, batch, summarizer, global_step, prefix):
        raise NotImplementedError

    def inspect(self, batch, env=None):
        return None

    def policy(self):
        self.policy_mode = True
        self.eval()

    def train(self, mode=True):
        super(Net, self).train(mode)
        if mode:
            self.policy_mode = False


def main():
    print()


if __name__ == '__main__':
    main()
