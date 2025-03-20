import shutil
import time

import scipy

from modules import *
from tqdm import tqdm
import math
from lr_schedule import lr_warmup


class BatchInputAndConcatResult(object):
    def __init__(self, data, batch_size):
        self.data = data
        self.max_i = len(data[0])
        if batch_size == 0:
            self.batch_size = self.max_i
        else:
            self.batch_size = batch_size
        self.i = 0
        self.result = []
        self.result_tuple_size = 0

    def __iter__(self):
        self.i = 0
        self.result = []
        return self

    def __next__(self):
        if self.i < self.max_i:
            i = self.i
            self.i = min(i + self.batch_size, self.max_i)
            return [d[i:self.i] for d in self.data]
        else:
            raise StopIteration

    def __len__(self):
        return math.ceil(self.max_i / self.batch_size)

    def update_result(self, batch_result):
        if isinstance(batch_result, tuple):
            self.result_tuple_size = len(batch_result)
            if len(self.result) == 0:
                self.result = [[] for i in range(self.result_tuple_size)]
            for i in range(self.result_tuple_size):
                self.result[i].append(batch_result[i])
        else:
            self.result.append(batch_result)

    def concat_result(self):
        if self.result_tuple_size > 0:
            tuple_size = len(self.result)
            self.result = tuple([np.concatenate(self.result[i], axis=0) for i in range(tuple_size)])
        else:
            self.result = np.concatenate(self.result, axis=0)
        return self.result


def block_one_diagonal_matrix(num_block, block_size):
    blocks = [np.ones((block_size, block_size)) for i in range(num_block)]
    return scipy.linalg.block_diag(*blocks)


def to_sparse_tensor(array):
    array = array.astype(np.float32)
    nonzero_indices = np.nonzero(array)
    indices = np.vstack(nonzero_indices).T
    values = array[nonzero_indices]
    shape = array.shape
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return sparse_tensor


def boolean_mask_sparse_1d(sparse_tensor, mask, axis=0):  # mask is assumed to be 1D
    assert axis >= 0
    mask = tf.convert_to_tensor(mask)
    ind = sparse_tensor.indices[:, axis]
    mask_sp = tf.gather(mask, ind)
    new_size = tf.math.count_nonzero(mask)
    new_size = tf.cast(new_size, tf.int32)
    new_shape = tf.concat([sparse_tensor.dense_shape[:axis], [new_size],
                           sparse_tensor.dense_shape[axis + 1:]], axis=0)
    new_shape = tf.dtypes.cast(new_shape, tf.int64)
    mask_count = tf.cumsum(tf.dtypes.cast(mask, tf.int64), exclusive=True)
    masked_idx = tf.boolean_mask(sparse_tensor.indices, mask_sp)
    new_idx_axis = tf.gather(mask_count, masked_idx[:, axis])
    new_idx = tf.concat([masked_idx[:, :axis],
                         tf.expand_dims(new_idx_axis, 1),
                         masked_idx[:, axis + 1:]], axis=1)
    new_values = tf.boolean_mask(sparse_tensor.values, mask_sp)
    return tf.SparseTensor(new_idx, new_values, new_shape)


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=(), name='input_is_training')
        self.u = tf.placeholder(tf.int32, shape=(None, ), name='input_u')
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_seq')
        self.input_interval = tf.placeholder(tf.int32, shape=(None, args.maxlen, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen), name='input_neg')
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 initializer=args.embedding_initializer,
                                                 scale=args.embedding_scale,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            if args.positional_embedding:
                t, pos_emb_table = embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0),
                            [tf.shape(self.input_seq)[0], 1]),
                    vocab_size=args.maxlen,
                    num_units=args.hidden_units,
                    zero_pad=False,
                    initializer=args.embedding_initializer,
                    scale=False,
                    l2_reg=args.l2_emb,
                    scope="dec_pos",
                    reuse=reuse,
                    with_t=True
                )
                self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq) if args.pre_norm else self.seq,
                                                   keys=self.seq,
                                                   input_interval=self.input_interval,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   attention_type=args.attention_type,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   linear_projection_and_dropout=args.linear_projection_and_dropout,
                                                   args=args,
                                                   scope="self_attention")
                    if args.post_norm:
                        self.seq = normalize(self.seq)
                    if args.pre_norm:
                        self.seq = normalize(self.seq)
                    # Feed forward
                    if args.ffn:
                        self.seq = feedforward(self.seq,
                                               num_units=[args.inner_size, args.hidden_units],
                                               inner_act=args.inner_act,
                                               dropout_rate=args.dropout_rate,
                                               inner_dropout=args.inner_dropout,
                                               is_training=self.is_training)
                    self.seq *= mask
                    if args.post_norm:
                        self.seq = normalize(self.seq)

            if args.pre_norm:
                self.seq = normalize(self.seq)  # (N, T, C)

            pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])  # (N, )
            neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])  # (N, )
            seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])  # (N*T, C)
            # ignore padding items (0)
            istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])

            self.test_item = tf.placeholder(tf.int32, shape=(None, ), name='input_test_item')
            self.test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)  # (M, C)

        ######## Test Graph Prediction Layer ########
        if args.normalize_test_embedding:
            self.test_item_emb = tf.math.l2_normalize(self.test_item_emb, axis=-1)

        test_item_emb = self.test_item_emb

        if args.normalize_test_embedding:
            normalized_seq_emb = tf.math.l2_normalize(self.seq[:, -1, :], axis=-1)

            self.test_logits = tf.matmul(normalized_seq_emb, tf.transpose(test_item_emb))
        else:
            self.test_logits = tf.matmul(self.seq[:, -1, :], tf.transpose(test_item_emb))  # (N, M)

        if args.item_bias:
            item_bias_table = tf.get_variable('item_bias', shape=(itemnum + 1,), dtype=tf.float32)
            item_bias = tf.nn.embedding_lookup(item_bias_table, self.test_item)
            self.test_logits += tf.expand_dims(item_bias, axis=0)
        else:
            item_bias_table = None

        if args.eval_item_not_in_history:
            # Set corresponding logits to float min
            self.input_hist = tf.sparse.placeholder(dtype=tf.float32, shape=(None, itemnum + 1), name='input_hist')
            full_hist = tf.sparse.to_dense(self.input_hist, validate_indices=False)
            self.test_logits = tf.where(tf.greater(full_hist, 0), tf.float32.min * tf.ones_like(self.test_logits), self.test_logits)
        self.test_top_k = tf.nn.top_k(self.test_logits, k=100)

        self.test_top_k2_num = tf.placeholder(dtype=tf.int32, name='test_top_k_num')
        self.test_top_k2 = tf.nn.top_k(self.test_logits, k=self.test_top_k2_num)

        ######## Training Graph Prediction Layer #########
        if args.normalize_prediction_embedding:
            seq_emb = tf.math.l2_normalize(seq_emb, axis=-1)

        ######## Loss and Optimizer ########
        if args.loss_type == 'sparse_ce':
            self.loss = self.sparse_ce_loss(pos, item_emb_table, item_bias_table, seq_emb, istarget, args)
        else:
            raise ValueError('loss type [{}] is not supported.'.format(args.loss_type))

        tf.summary.scalar('loss', self.loss)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_losses = sum(reg_losses)
        tf.summary.scalar('reg_loss', reg_losses)
        self.loss += reg_losses
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.histogram('item_emb_norm', tf.norm(item_emb_table, axis=-1))

        if reuse is None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if args.warmup_steps > 0:
                lr = lr_warmup(self.global_step, args.warmup_steps, start_lr=0.0, target_lr=args.lr)
            else:
                lr = args.lr
            if args.optimizer == 'adamw':
                self.optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=args.weight_decay, learning_rate=lr,
                                                               beta2=args.adam_beta2)
            elif args.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=args.adam_beta2)
            elif args.optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            else:
                raise ValueError
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, batch_size=0):
        if batch_size == 0:
            return sess.run(self.test_logits,
                            {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
        else:
            batch_inputs = BatchInputAndConcatResult([u, seq], batch_size=batch_size)
            for batch_u, batch_seq in batch_inputs:
                batch_inputs.update_result(
                    sess.run(self.test_logits,
                             {self.u: batch_u,
                              self.input_seq: batch_seq,
                              self.test_item: item_idx,
                              self.is_training: False})
                )
            return batch_inputs.concat_result()

    def batch_u(self, batch_u):
        return np.asarray(batch_u, np.int32)

    def batch_seq(self, batch_seq):
        return np.asarray(batch_seq, np.int32)

    def batch_item_idx(self, batch_item_idx):
        return np.asarray(batch_item_idx, np.int32)

    def batch_interval(self, batch_interval):
        return np.asarray(batch_interval, np.int32)

    def predict_top_k(self, sess: tf.Session, u, seq, item_idx, batch_size=0,
                      user_hist_train: dict = None, user_hist_val: dict = None, user_hist_test: dict = None,
                      convert_hist_item_id=False, interval=(), top_k=100):
        if interval:
            data = [u, seq, interval]
        else:
            data = [u, seq]
        batch_inputs = BatchInputAndConcatResult(data, batch_size=batch_size)
        if convert_hist_item_id:
            item_idx = list(item_idx)
            item_id_to_input_id = {item_id: i for i, item_id in enumerate(item_idx)}
        else:
            item_id_to_input_id = {}

        def get_batch_sparse_input_hist(batch_u):
            indices = []
            shape = (len(batch_u), len(item_idx))
            for i, uu in enumerate(batch_u):
                for ii in user_hist_train[uu]:
                    if isinstance(ii, tuple):
                        ii = ii[0]
                    if convert_hist_item_id:
                        try:
                            ii = item_id_to_input_id[ii]
                        except KeyError:
                            continue
                    indices.append((i, ii))
                if user_hist_val:
                    for ii in user_hist_val[uu]:
                        if isinstance(ii, tuple):
                            ii = ii[0]
                        if convert_hist_item_id:
                            try:
                                ii = item_id_to_input_id[ii]
                            except KeyError:
                                continue
                        indices.append((i, ii))
                if user_hist_test:
                    for ii in user_hist_test[uu]:
                        if isinstance(ii, tuple):
                            ii = ii[0]
                        if convert_hist_item_id:
                            try:
                                ii = item_id_to_input_id[ii]
                            except KeyError:
                                continue
                        indices.append((i, ii))
                # Always add item 0
                indices.append((i, 0))
            return indices, np.ones((len(indices))), shape

        def topk_by_partition(input, k, axis=None, ascending=True):
            if not ascending:
                input *= -1
            ind = np.argpartition(input, k, axis=axis)
            ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
            input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values

            # sort within k elements
            ind_part = np.argsort(input, axis=axis)
            ind = np.take_along_axis(ind, ind_part, axis=axis)
            if not ascending:
                input *= -1
            val = np.take_along_axis(input, ind_part, axis=axis)
            return ind, val

        def predict_randomly(feed_dict):
            batch_u = feed_dict[self.u]
            batch_item_idx = feed_dict[self.test_item]
            shape = (len(batch_u), len(batch_item_idx))
            scores = np.random.random(size=shape)
            hist_indices, hist_values, hist_shape = feed_dict[self.input_hist]
            import scipy
            hist_sparse = scipy.sparse.csr_matrix((hist_values, zip(*hist_indices)), shape=hist_shape)
            # scores = scores - 10 * hist_sparse
            scores = np.array(scores)
            top_k_ind, top_k_value = topk_by_partition(scores, 100, axis=-1, ascending=False)
            return top_k_value, top_k_ind

        random_prediction = False
        # import global_variables
        # global_variables.total_time = 0.0
        # global_variables.total_occurrence = 0
        # print('total_time: {}'.format(global_variables.total_time))
        # print('total_occurrence: {}'.format(global_variables.total_occurrence))
        item_idx = np.array(item_idx, dtype=np.int32)
        cache_test_item_embedding = hasattr(self, 'input_test_item_emb')
        if cache_test_item_embedding:
            test_item_embedding = sess.run(
                self.test_item_emb, feed_dict={self.test_item: self.batch_item_idx(item_idx), self.is_training: False})
        for batch_data in tqdm(batch_inputs, ncols=70, unit='b'):
            if interval:
                batch_u, batch_seq, batch_interval = batch_data
            else:
                batch_u, batch_seq = batch_data
            batch_u = self.batch_u(batch_u)
            batch_seq = self.batch_seq(batch_seq)
            batch_item_idx = self.batch_item_idx(item_idx)
            feed_dict = {self.u: batch_u,
                         self.input_seq: batch_seq,
                         self.test_item: batch_item_idx,
                         self.is_training: False,
                         self.test_top_k2_num: top_k}
            if interval:
                feed_dict[self.input_interval] = self.batch_interval(batch_interval)
            if user_hist_train:
                feed_dict[self.input_hist] = get_batch_sparse_input_hist(batch_u)
            if random_prediction:
                batch_result = predict_randomly(feed_dict)
            else:
                if cache_test_item_embedding:
                    feed_dict.pop(self.test_item)
                    feed_dict[self.input_test_item_emb] = test_item_embedding
                    batch_result = sess.run(self.test_top_k2, feed_dict)
                else:
                    batch_result = sess.run(self.test_top_k2, feed_dict,
                                            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

            batch_inputs.update_result(batch_result)
        # print('total_time: {}'.format(global_variables.total_time))
        # print('total_occurrence: {}'.format(global_variables.total_occurrence))
        return batch_inputs.concat_result()

    def sparse_ce_loss(self, pos, item_emb_table, item_bias_table, seq_emb, istarget, args):
        if args.normalize_prediction_embedding:
            item_emb_table = tf.math.l2_normalize(item_emb_table, axis=-1)
        logits = tf.matmul(seq_emb, tf.transpose(item_emb_table))  # (N, M)
        if args.scale_logits_trainable:
            scale_logits = tf.get_variable('scale_logits', None, dtype=tf.float32, initializer=lambda: 16.0)
            tf.summary.scalar('scale_logits', scale_logits)
            scale_logits = clip_by_value_preserve_gradient(scale_logits, 2.0, 128.0)
            logits = scale_logits * logits
        elif args.scale_logits:
            logits = args.scale_logits * logits
        if item_bias_table is not None:
            logits += tf.expand_dims(item_bias_table, axis=0)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos, logits=logits)
        loss = tf.reduce_sum(losses * istarget) / tf.reduce_sum(istarget)
        return loss