import copy
import numpy as np
from collections import defaultdict
import time
from array import array
from tabulate import tabulate
import collections


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(self.name, time.time() - self.start))
        return exc_type is None


def print_seq_len_percentile(seqs, message):
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = np.percentile([len(v) for k, v in seqs.items()], percentiles)
    print(message)
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))


def issequence(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.Sequence)


class MultiArray(object):
    def __init__(self, list_of_typecode, initializer=None):
        self.arrays = []
        self.list_of_typecode = tuple(list_of_typecode)
        if initializer is None:
            initializer = [[] for i in range(len(self.list_of_typecode))]
        else:
            assert len(self.list_of_typecode) == len(initializer)
        for typecode, i in zip(self.list_of_typecode, initializer):
            self.arrays.append(array(typecode, i))

    def append(self, item):
        if len(self.arrays) > 1:
            assert len(item) == len(self.arrays)
            for array, i in zip(self.arrays, item):
                array.append(i)
        else:
            self.arrays[0].append(item)

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, slice):
            return MultiArray(self.list_of_typecode, [array[index_or_slice] for array in self.arrays])
        else:
            if len(self.arrays) > 1:
                return tuple(array[index_or_slice] for array in self.arrays)
            else:
                return self.arrays[0][index_or_slice]

    def __add__(self, other):
        assert len(self.list_of_typecode) == len(other.list_of_typecode)
        assert len(self.arrays) == len(other.arrays)
        arrays = [a + b for (a, b) in zip(self.arrays, other.arrays)]
        return MultiArray(self.list_of_typecode, arrays)

    def __len__(self):
        return len(self.arrays[0])

    def __reversed__(self):
        return MultiArray(self.list_of_typecode, [reversed(array) for array in self.arrays])

    def __copy__(self):
        return MultiArray(self.list_of_typecode, [copy.copy(array) for array in self.arrays])

    def __iter__(self):
        if len(self.arrays) > 1:
            return zip(*self.arrays)
        else:
            return (a for a in self.arrays)

    def __repr__(self):
        return repr(self.arrays)


def create_array(typecode='i'):
    return array(typecode, [])


def create_multiarray(typecode=('i', 'i')):
    return MultiArray(typecode)


def array_like(ref):
    if isinstance(ref, MultiArray):
        return MultiArray(list_of_typecode=ref.list_of_typecode)
    elif isinstance(ref, array):
        return array(ref.typecode)
    elif isinstance(ref, list):
        return list()


def train_val_test_partition(User):
    user_train = {}
    user_valid = {}
    user_test = {}
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # user_train[user] = np.array(User[user], dtype=np.int32)
            user_train[user] = User[user]
            user_valid[user] = array_like(User[user])
            user_test[user] = array_like(User[user])
        else:
            # user_train[user] = np.array(User[user][:-2], dtype=np.int32)
            user_train[user] = User[user][:-2]
            user_valid[user] = array_like(User[user])
            user_valid[user].append(User[user][-2])
            user_test[user] = array_like(User[user])
            user_test[user].append(User[user][-1])
    return user_train, user_valid, user_test


def load_hstu_ml_1m(fname, args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    import pandas as pd
    ratings = pd.read_csv('data/%s.csv' % fname, sep=',')
    item_to_id = {}
    for row in ratings.iterrows():
        row = row[1]
        user_id = int(row.user_id)
        usernum = max(user_id, usernum)
        sequence_item_ids = eval(row.sequence_item_ids)
        if isinstance(sequence_item_ids, int):
            sequence_item_ids = [sequence_item_ids, ]
        else:
            sequence_item_ids = list(sequence_item_ids)
        sequence = []
        if args.remap_hstu_ml_1m:
            for item_id in sequence_item_ids:
                if item_id in item_to_id:
                    sequence.append(item_to_id[item_id])
                else:
                    itemnum += 1
                    item_to_id[item_id] = itemnum
                    sequence.append(item_to_id[item_id])
        else:
            sequence = sequence_item_ids
            itemnum = max(itemnum, max(sequence))

        if args.load_timestamp:
            timestamp = eval(row.sequence_timestamps)
            if isinstance(timestamp, int):
                timestamp = [timestamp, ]
            else:
                timestamp = list(timestamp)
            sequence = MultiArray(('i', 'i'), initializer=(sequence, timestamp))

        User[user_id] = sequence

    print('user_num: {}'.format(usernum))
    print('item_num: {}'.format(itemnum))
    print('Total number of interactions in train + val: {}'.format(sum([len(v) for v in User.values()])))
    print('Average sequence length of train + val: {:.2f}'.format(sum([len(v) for v in User.values()]) / len(User.values())))
    print_seq_len_percentile(User, message='Sequence length percentile in train + val: ')

    user_train = {}
    user_valid = {}
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            # user_train[user] = np.array(User[user], dtype=np.int32)
            user_train[user] = User[user]
            user_valid[user] = array_like(User[user])
        else:
            # user_train[user] = np.array(User[user][:-2], dtype=np.int32)
            user_train[user] = User[user][:-1]
            user_valid[user] = array_like(User[user])
            user_valid[user].append(User[user][-1])

    return [user_train, user_valid, None, usernum, itemnum]


def print_item_frequency_percentile(frequencies):
    percentiles = [10, 25, 50, 75, 90]
    values = np.percentile(frequencies, percentiles)
    print('Item frequency percentile: ')
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))


def print_percentile(values):
    percentiles = [10, 25, 50, 75, 90]
    values = np.percentile(values, percentiles)
    print(tabulate([['Percentile'] + percentiles, ['Value'] + values.tolist()]))