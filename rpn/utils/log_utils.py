import sys
import torch
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from rpn.utils.timer import Timer
from rpn.utils.torch_utils import to_numpy
from collections import OrderedDict, namedtuple


class StatsSummarizer(SummaryWriter):
    def __init__(self, log_dir=None, exp_name='exp', **kwargs):
        self._log_to_board = False
        if log_dir is not None:
            super(StatsSummarizer, self).__init__(log_dir, comment=exp_name, **kwargs)
            self._log_to_board = True
        self._stats = OrderedDict()
        self._stats_iter = dict()
        self._timers = OrderedDict()
        self._curr_step = 0
        self._name = exp_name

    def timer_tic(self, timer_name):
        if timer_name not in self._timers:
            self._timers[timer_name] = Timer()
        self._timers[timer_name].tic()

    def timer_toc(self, timer_name):
        if timer_name not in self._timers:
            raise KeyError
        self._timers[timer_name].toc()

    def _add_stats(self, key, val, n_iter):
        if key not in self._stats:
            self._stats[key] = []
            self._stats_iter[key] = []
        if isinstance(val, torch.Tensor):
            val = to_numpy(val)
        self._stats[key].append(val)
        self._stats_iter[key].append(n_iter)

    def add_scalar(self, tag, tag_scalar, global_step=None, walltime=None, is_stats=True):
        """ Add a summarization item given a key and a value.
        Create a new item if the key does not exist
        :param tag: key of the stat item
        :param tag_scalar: val of the stat item
        :param global_step: iteration of the stats
        :param walltime: event time
        :param is_stats: is a stats item
        :return: None
        """
        if self._log_to_board:
            super(StatsSummarizer, self).add_scalar(tag, tag_scalar, global_step, walltime)
        if is_stats:
            self._add_stats(tag, tag_scalar, global_step)
        if global_step is not None:
            self._curr_step = global_step

    @property
    def header(self):
        return '[%s] step=%i' % (self._name, self._curr_step)

    def summarize_stats(self, key, stats=['mean'], prec='.6f', last_n=0, include_header=True):
        """ Summarize a logged item with a set of stats
        :param key: key of the stat item
        :param stats: type of stats
        :param last_n: summarize the latest n values of the item
        :param prec: precision
        :param include_header: if include global header
        :return: a string summarizing the stats
        """
        if key not in self._stats:
            raise ValueError('%s has not been logged' % key)

        header = self.header + ' ' if include_header else ''
        header += key + ' '
        msgs = []

        if 'mean' in stats:
            msgs.append(('mean=%' + prec) %
                np.mean(self._stats[key][-last_n:])
            )

        if 'std' in stats:
            msgs.append(('std=%' + prec) %
                        np.std(self._stats[key][-last_n:])
                        )

        return header + ', \t'.join(msgs)

    def summarize_all_stats(self, prefix='', stats=['mean'], prec='.6f', last_n=0):
        """ Summarize all items in the logger
        :param prefix: prefix of the stats
        :param stats: type of stats
        :param last_n: summarize the latest n values of the item
        :param prec: precision
        :return: a string summarizing the stats
        """
        msg = ''
        for k, v in self._stats.items():
            if k.startswith(prefix):
                msg += self.summarize_stats(k, stats, prec, last_n) + '\n'
        return msg

    def summarize_timers(self, last_n=0):
        msg = []
        for k, v in self._timers.items():
            msg.append('%s=%.4f' % (k, v.recent_average_time(last_n)))
        return self.header + ' | Timer: ' + ', '.join(msg)


class PrintLogger(object):
    """
    This class redirects print statement to both console and a file
    """
    def __init__(self, log_file, fn_timestamp=True):
        self.terminal = sys.stdout
        if fn_timestamp:
            d = datetime.now()
            log_file += '.' + str(d.date()) + '-' + str(d.time()).replace(':', '-')
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
