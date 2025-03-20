import sys
import copy
import random
import time

import numpy as np
from tqdm import tqdm
import sampler


def predict_top_k_with_item_set(item_set, model, sess, us, seqs, batch_size, user_train=None, user_valid=None,
                                user_test=None, convert_hist_item_id=False, async_submitter=None):
    items = np.array(sorted(list(item_set)))
    if user_train:
        top_k_values, top_k_indices = model.predict_top_k(sess, us, seqs, items, batch_size=batch_size,
                                                          user_hist_train=user_train, user_hist_val=user_valid,
                                                          user_hist_test=user_test,
                                                          convert_hist_item_id=convert_hist_item_id,
                                                          async_submitter=async_submitter)
    else:
        top_k_values, top_k_indices = model.predict_top_k(sess, us, seqs, items, batch_size=batch_size)

    if async_submitter:
        return None, None

    # Convert back
    top_k_indices_shape = np.shape(top_k_indices)
    top_k_indices = top_k_indices.flatten()
    top_k_indices = items[top_k_indices]
    top_k_indices = np.reshape(top_k_indices, top_k_indices_shape)
    return top_k_values, top_k_indices


def build_seq(train, valid, test, u, target_timestamp, args):
    seq = np.zeros([args.maxlen], dtype=np.int32)
    timestamp = np.zeros([args.maxlen], dtype=np.int64)
    idx = args.maxlen - 1
    if test is not None:
        if args.load_timestamp:
            seq[idx] = test[u][0][0]
            timestamp[idx] = test[u][0][1]
        else:
            seq[idx] = test[u][0]
        idx -= 1

    if valid is not None:
        if args.load_timestamp:
            seq[idx] = valid[u][0][0]
            timestamp[idx] = valid[u][0][1]
        else:
            seq[idx] = valid[u][0]
        idx -= 1

    for i in reversed(train[u]):
        if args.load_timestamp:
            seq[idx] = i[0]
            timestamp[idx] = i[1]
        else:
            seq[idx] = i
        idx -= 1
        if idx == -1: break

    if args.load_timestamp:
        if args.compute_hstu_time_interval:
            interval = sampler.compute_hstu_time_interval(timestamp, target_timestamp, args.hstu_time_interval_divisor,
                                                          max_value=args.time_interval_attention_max_interval)
        else:
            interval = sampler.compute_interval(timestamp, max_value=args.time_interval_attention_max_interval)
    else:
        interval = None
    return seq, interval


def exact_evaluate(model, dataset, args, sess, mode='test', batch_size=0, sample_user_num=10000,
                   evaluate_user=(), evaluate_item=(), eval_item_not_in_history=False):
    train, valid, test, user_num, item_num = dataset

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    users = list(evaluate_user) if len(evaluate_user) > 0 else range(1, user_num + 1)
    if sample_user_num > 0 and len(users) > sample_user_num:
        users = random.sample(users, sample_user_num)
        eval_user_num = sample_user_num
    else:
        eval_user_num = len(users)

    us = []
    seqs = []
    intervals = []
    target_item_ids = []
    print('Preparing evaluate data...')
    for u in tqdm(users, total=eval_user_num, leave=False, ncols=70):
        if len(train[u]) < 1:
            continue
        if mode == 'valid' and len(valid[u]) < 1:
            continue
        if mode == 'test' and len(test[u]) < 1:
            continue

        target_timestamp = None
        if args.load_timestamp:
            if mode == 'valid':
                target_timestamp = valid[u][0][1]
            elif mode == 'test':
                target_timestamp = test[u][0][1]
        seq, interval = build_seq(train, valid if mode is 'test' else None, None, u, target_timestamp, args)

        # Put target_item_idx in 0-position
        target_item_id = valid[u][0] if mode == 'valid' else test[u][0]
        if args.load_timestamp:
            target_item_id = target_item_id[0]

        us.append(u)
        seqs.append(seq)
        if args.load_timestamp:
            intervals.append(interval)
        target_item_ids.append(target_item_id)
        valid_user += 1
    # predictions = model.predict(sess, us, seqs, list(range(1, item_num + 1)), batch_size=batch_size)  # (N, M)
    # target_item_scores = predictions[np.arange(int(valid_user)), np.array(target_item_ids, dtype=np.int32) - 1] #  (N, )
    # ranks = np.sum(np.expand_dims(target_item_scores, -1) < predictions, axis=-1)
    # NDCG = np.sum((1 / np.log2(ranks + 2)) * (ranks < 10))
    # HT = np.sum(ranks < 10)
    # print('NDCG: {} | HT: {}'.format(NDCG, HT))
    print('Predicting on evaluate data...')
    if eval_item_not_in_history:
        if len(evaluate_item) > 0:
            top_k_values, top_k_indices = predict_top_k_with_item_set(evaluate_item, model, sess, us, seqs,
                                                                      batch_size,
                                                                      user_train=train,
                                                                      user_valid=valid if mode == 'test' else None,
                                                                      convert_hist_item_id=True)
        else:
            top_k_values, top_k_indices = model.predict_top_k(sess, us, seqs, list(range(item_num + 1)), batch_size,
                                                              train, valid if mode == 'test' else None,
                                                              interval=intervals, top_k=200)
    elif len(evaluate_item) > 0:
        top_k_values, top_k_indices = predict_top_k_with_item_set(evaluate_item, model, sess, us, seqs, batch_size)
    else:
        top_k_values, top_k_indices = model.predict_top_k(sess, us, seqs, list(range(1, item_num + 1)),
                                                          batch_size=batch_size, interval=intervals)
        top_k_indices += 1

    top_10_indices = top_k_indices[:, :10]
    HT = np.sum(np.expand_dims(target_item_ids, axis=-1) == top_10_indices)
    NDCG = np.sum(
        (np.expand_dims(target_item_ids, axis=-1) == top_10_indices) *
        (1 / np.expand_dims(np.log2(np.arange(0, 10) + 2), axis=0))
    )
    # print('NDCG: {} | HT: {}'.format(NDCG, HT))
    HT /= valid_user
    NDCG /= valid_user

    top_100_indices = top_k_indices[:, :100]
    HT100 = np.sum(np.expand_dims(target_item_ids, axis=-1) == top_100_indices)
    NDCG100 = np.sum(
        (np.expand_dims(target_item_ids, axis=-1) == top_100_indices) *
        (1 / np.expand_dims(np.log2(np.arange(0, 100) + 2), axis=0))
    )
    HT100 /= valid_user
    NDCG100 /= valid_user

    top_200_indices = top_k_indices[:, :200]
    HT200 = np.sum(np.expand_dims(target_item_ids, axis=-1) == top_200_indices)
    NDCG200 = np.sum(
        (np.expand_dims(target_item_ids, axis=-1) == top_200_indices) *
        (1 / np.expand_dims(np.log2(np.arange(0, 200) + 2), axis=0))
    )
    HT200 /= valid_user
    NDCG200 /= valid_user

    return NDCG, HT, NDCG100, HT100, NDCG200, HT200


