import numpy as np
import multiprocessing


def random_neq(l, r, i_set, s):
    if len(i_set) > 0:
        t = np.random.choice(i_set)
    else:
        t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def compute_interval(timestamp, max_value=256):
    interval = np.expand_dims(timestamp, axis=1) - np.expand_dims(timestamp, axis=0)
    mask = (timestamp) != 0
    mask = mask.astype(np.int32)
    mask = np.expand_dims(mask, axis=1).dot(np.expand_dims(mask, axis=0))
    r = np.arange(len(timestamp))
    mask = mask * (r[:, None] > r)
    interval = interval * mask
    interval = np.minimum(interval, max_value)

    return interval


def compute_hstu_time_interval(timestamp, last_timestamp, time_interval_divisor, max_value):
    next_timestamp = np.concatenate([timestamp[1:], [last_timestamp, ]])
    interval = np.expand_dims(next_timestamp, axis=1) - np.expand_dims(timestamp, axis=0)
    interval = np.clip(np.log(np.maximum(1, interval)) / time_interval_divisor, 0, max_value).astype(np.int32)
    return interval


def sample_function(user_train, user_valid, user_test, usernum, itemnum, batch_size, maxlen, in_queue, result_queue,
                    stop_event, args, SEED):

    def sample():
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        timestamp = np.zeros([maxlen], dtype=np.int64)

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        user_hist = user_train[user]

        if args.load_timestamp:
            nxt = user_hist[-1][0]
        else:
            nxt = user_hist[-1]
        idx = maxlen - 1

        if args.load_timestamp:
            ts = set(user_hist.arrays[0])
        else:
            ts = set(user_hist)
        for i in reversed(user_hist[:-1]):
            if args.load_timestamp:
                seq[idx] = i[0]
                timestamp[idx] = i[1]
            else:
                seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, set(), ts)
            if args.load_timestamp:
                nxt = i[0]
            else:
                nxt = i
            idx -= 1
            if idx == -1: break
        if args.load_timestamp:
            if args.compute_hstu_time_interval:
                interval = compute_hstu_time_interval(timestamp, user_hist[-1][1],
                                                      time_interval_divisor=args.hstu_time_interval_divisor,
                                                      max_value=args.time_interval_attention_max_interval)
            else:
                interval = compute_interval(timestamp, max_value=args.time_interval_attention_max_interval)
        else:
            interval = None
        if len(user_hist) == 1:
            print('seq: {}'.format(seq))
            print('pos: {}'.format(pos))
        return (user, seq, interval, pos, neg)

    np.random.seed(SEED)
    while not stop_event.is_set():
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        one_batch = zip(*one_batch)
        one_batch = list(one_batch)
        one_batch.append(None)
        # logger.info('Put one batch')
        result_queue.put(one_batch)


class WarpSampler(object):
    def __init__(self, User, user_valid, user_test, usernum, itemnum, batch_size=64, maxlen=10, n_workers=3, args=None):
        ctx = multiprocessing.get_context('fork')
        create_worker = ctx.Process
        self.in_queue = None
        self.result_queue = ctx.Queue(maxsize=n_workers * 10)
        self.stop_event = ctx.Event()
        # self.manager = ctx.Manager()
        # User = self.manager.dict(User)
        self.processors = []
        self.length = len(User)

        for i in range(n_workers):
            self.processors.append(
                create_worker(target=sample_function, args=(User,
                                                            user_valid,
                                                            user_test,
                                                            usernum,
                                                            itemnum,
                                                            batch_size,
                                                            maxlen,
                                                            self.in_queue,
                                                            self.result_queue,
                                                            self.stop_event,
                                                            args,
                                                            np.random.randint(2e9)
                                                            )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self, debug=False):
        if debug:
            print('result_queue.size: {}'.format(self.result_queue.qsize()))
        if self.is_children_dead():
            print('Warning: Sampler worker killed')
        return self.result_queue.get()

    def is_children_dead(self):
        for p in self.processors:
            if not p.is_alive():
                return True

    def close(self):
        self.stop_event.set()
        while not self.result_queue.empty():
            self.result_queue.get()
        while self.in_queue:
            self.in_queue.get()
        for p in self.processors:
            # print('Terminate 1 processor')
            p.terminate()
            # print('Join 1 processors')
            p.join()
            # print('Join successfully')