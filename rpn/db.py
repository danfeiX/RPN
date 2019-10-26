"""A general-purpose DB class for storing nested sequential data."""

import numpy as np

SAMPLE = 0
SEQUENCE = 1


class BasicDB(object):
    """
    Base class DB
    """

    def __init__(self, keys=(), db=None):
        """
        Constructor
        :param keys: db keys
        :param db: external data dictionary
        """
        self._data = {}
        for k in keys:
            self._data[k] = []

        if db is not None:
            self._data = db

    @property
    def data(self):
        return self._data
    #
    # def subset(self, begin_idx, end_idx):
    #     raise NotImplementedError

    def get_db_item(self, key, index):
        raise NotImplementedError


class ReplayBuffer(BasicDB):
    """
    Python list-based DB. Good for experience replay. Can be consolidated to contiguous DB
    """
    def append(self, **kwargs):
        """
        Add entries to the db.
        :param kwargs: dictionary of new data entries. Each value is a list or an numpy n-D array
        """
        for k, v in kwargs.items():
            self._data[k].append(v)

    def get_empty_buffer(self):
        """
        Get a new empty buffer based on the current buffer's keys
        :return: a new empty buffer
        """
        return self.__class__(keys=self.data.keys())

    def get_db_item(self, key, index):
        """
        Fetch a db item
        :param key: db item key
        :param index: db item index
        :return: the target db item.
        """
        return self._data[key][index]

    def contiguous(self, n_level):
        """
        convert to contiguous dataset by recursively flatten the db
        :param n_level: Number of levels to recursively flatten.
        1: sample with variable size: 2: variable-length sequence of variable-size samples
        :return: a Contiguous DB
        """
        db = {}
        # length for each sample level
        all_sample_len = [dict() for _ in range(n_level)]
        for k, v in self._data.items():
            if len(v) > 0:
                sample_len = [[] for _ in range((n_level + 1))]
                db[k] = np.concatenate(self._contiguous_helper(v, n_level, sample_len), axis=0)
                for asl, sl in zip(all_sample_len, sample_len[:-1]):
                    asl[k] = sl

        # check length consistency
        # length of each item should be identical at all levels except the first (sample)
        for sl in all_sample_len[1:]:
            for all_len in zip(*sl.values()):
                if len(np.unique(all_len)) != 1:
                    raise ValueError('DB items do not have the same length')

        return ContiguousDB(db=db, sample_len=all_sample_len)

    def _contiguous_helper(self, db_item, n, sample_len):
        """
        Recursively flatten the db item
        :param db_item: current recurrent db item
        :param n: current recursion level
        :param sample_len: a list of sample length ordered by recursion level
        :return:
        """
        sample_len[n].append(len(db_item))
        if n == 0:
            return [db_item]
        flat_db = []
        for v in db_item:
            fdb = self._contiguous_helper(v, n - 1, sample_len)
            flat_db.extend(fdb)
        return flat_db


class ContiguousDB(BasicDB):
    """
    Stored data in contiguous NP arrays. Supports arbitrary level variable-sized samples
    """
    def __init__(self, db, sample_len):
        super(ContiguousDB, self).__init__((), db)
        self._sample_len = sample_len
        self.data_keys = sample_len[0].keys()
        # begin indices of each sample
        self._sample_begin_index = [dict() for _ in range(len(self._sample_len))]
        for sl, si in zip(self._sample_len, self._sample_begin_index):
            for k, v in sl.items():
                si[k] = np.append([0], np.cumsum(v[:-1])).astype(np.int64)

    @property
    def sample_len(self):
        return self._sample_len

    @property
    def sample_begin_index(self):
        return self._sample_begin_index

    def get_begin_index(self, key, index, level, end_level=-1):
        """
        Get the beginning index of a sample at some level wrt to a lower level

        :param key: key of the sample
        :param index: index of the query sample
        :param level: level of the query sample
        :param end_level: the lowest level of traversal. Default to be sub-sample
        :return: the beginning index (Integer)
        """
        for l in range(level, end_level, -1):
            index = self._sample_begin_index[l][key][index]
        return index

    def num_samples(self, level, key=None):
        # if key is None:
        #     assert(reduce((lambda x, y: x == y), [len(v) for v in self._sample_len[level].values()]))
        return len(list(self._sample_len[level].values())[0])

    # def num_samples_recursive(self, key, index, level, end_level=SAMPLE):
    #     sample_len = self._sample_len[level][key][index]
    #     if level > end_level:
    #         sample_len = self._count_helper(sample_len, level - 1, end_level)
    #     return sample_len
    #
    # def _count_helper(self, sample_len, level, end_level):
    #     if level + 1 == end_level:
    #         return sample_len if isinstance(sample_len, int) else len(sample_len)
    #     total_len = reduce((lambda x, y: x + y), [self._count_helper(sl, level - 1, end_level) for sl in sample_len])
    #     return total_len

    def get_db_item(self, key, index, end_index=None, level=SAMPLE):
        # TODO: FIX THIS CRAZY HACK
        if end_index is None:
            end_index = index + 1
        assert(end_index > index)

        sample_begin_idx = self.get_begin_index(key, index, level)
        if end_index < len(self.sample_len[level][key]):
            sample_end_idx = self.get_begin_index(key, end_index, level)
            return self._data[key][sample_begin_idx:sample_end_idx]
        else:
            return self._data[key][sample_begin_idx:]

    def get_db_item_list(self, key, begin_index, end_index, level=SAMPLE):
        """
        Get a contiguous list of samples
        :param key: db item to fetch
        :param begin_index: range begin index
        :param end_index: range end index
        :param level: level to fetch sample from
        :return: a list of samples
        """
        assert(end_index >= begin_index)
        if end_index == begin_index:
            return []

        range_item = self.get_db_item(key, begin_index, end_index, level)
        sample_lens = self.sample_len[level][key][begin_index:end_index]
        split_index = np.cumsum(sample_lens)[:-1]
        item_list = np.split(range_item, split_index)
        assert(len(item_list) == (end_index - begin_index))
        return item_list

    def get_db_sample(self, index, level=SAMPLE):
        """
        Get a db sample (all item at an index)
        :param index: index to fetch
        :param level: level to fetch sample from
        :return: a dictionary of db items
        """
        sample = {}
        for k in self._data.keys():
            sample[k] = self.get_db_item(k, index, level)
        return sample

    # def subset(self, begin_idx, end_idx, level=SAMPLE, **kwargs):
    #     """
    #     Get a subset of the DB
    #     :param begin_idx: beginning sample index
    #     :param end_idx: ending sample index (excluded)
    #     :return: A ContiguousDB object that contains the subset of the DB
    #     """
    #     assert(begin_idx >= 0 and end_idx <= self.num_samples(level))
    #     new_db = {}
    #     new_db_len = [d.copy() for d in self._sample_len]
    #     for k, v in self._data.items():
    #         s_begin_idx = self.get_begin_index(k, begin_idx, level)
    #         s_end_idx = self.get_begin_index(k, end_idx, level)
    #         new_db[k] = self._data[k][s_begin_idx:s_end_idx]
    #         new_db_len[level][k] = self._sample_len[level][k][begin_idx:end_idx]
    #     return self.__class__(db=new_db, sample_len=new_db_len)

    def serialize(self):
        rc = {
            'db': self._data,
            'sample_len': self._sample_len
        }
        return rc

    @classmethod
    def deserialize(cls, rc):
        return cls(db=rc['db'], sample_len=rc['sample_len'])


class SequenceDB(ContiguousDB):
    """
    A recursive ContiguousDB that has specialized sequence operators
    """
    def __init__(self, *args, **kwargs):
        super(SequenceDB, self).__init__(*args, **kwargs)
        self._sample_to_seq = np.arange(self.num_samples(SAMPLE))
        tmp_key = list(self.data_keys)[0]
        for seq_i in range(self.num_sequences):
            si = self.get_begin_index(tmp_key, seq_i, SEQUENCE, end_level=SAMPLE)
            self._sample_to_seq[si:si+self.sequence_length(seq_i)] = seq_i

    def sequence_begin_index(self, sequence_index):
        """
        Index of the first sample of the sequence
        :param sequence_index: index of a sequence
        :return: index of the first sample of the sequence
        """
        tmp_key = list(self.data_keys)[0]
        return self.get_begin_index(tmp_key, sequence_index, SEQUENCE, end_level=SAMPLE)

    def sequence_length(self, sequence_index):
        """
        Index of the first sample of the sequence
        :param sequence_index: index of a sequence
        :return: index of the first sample of the sequence
        """
        tmp_key = list(self.data_keys)[0]
        seq_len = self.sample_len[SEQUENCE][tmp_key][sequence_index]
        return seq_len

    @property
    def max_sequence_length(self):
        tmp_key = list(self.data_keys)[0]
        return max(self.sample_len[SEQUENCE][tmp_key])

    @property
    def num_sequences(self):
        return self.num_samples(SEQUENCE)

    def sequence_index_from_sample_index(self, sample_index):
        assert(sample_index < self.num_samples(SAMPLE))
        return self._sample_to_seq[sample_index]

    def index_without_eos(self):
        """
        Compute index of database such that End of Sequence (EoS) is not included
        :param db: database from db.py
        :return: indices in 1-D array
        """
        all_index = np.arange(self.num_samples(SAMPLE))
        k = list(self.data_keys)[0]
        sample_end_index = self.sample_begin_index[SEQUENCE][k] + self.sample_len[SEQUENCE][k] - 1
        return np.setdiff1d(all_index, sample_end_index)

    def average_sequence_len(self):
        return np.mean(self.sample_len[SEQUENCE]['actions'])
