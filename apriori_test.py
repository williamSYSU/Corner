# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : Corner-william
# @FileName     : apriori_test.py
# @Time         : Created at 2018/10/29
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

from nltk.corpus import stopwords
from tqdm import tqdm


def load_data_set(tag, select=None):
    """
    Load a sample data set
    :param tag: tag of file: 'neg' or 'pos'
    :param select: if select part of sentences, None: select all
    :return all_sent: A list of sentences
    :return all_pos: A list of POS dictionary of sentences
    :return data_set: A list of transactions. Each transaction contains several items.
    """
    tag = tag
    all_sent = []
    all_pos = []
    data_set = []

    file_name = 'data/clean_data/clean_{}.csv'.format(tag)
    with open(file_name, 'r') as file:
        all_lines = file.read().split('\n')

    for idx, line in enumerate(all_lines):
        if idx % 3 == 0:  # sentence
            all_sent.append(line.strip())
        elif idx % 3 == 1:  # pos_dict
            all_pos.append(eval(line.strip()))
        else:  # extracted noun or noun phrase
            data_set.append(line.strip().split('\t'))

    print('Load dataset finished!')
    return all_sent[:select], all_pos[:select], data_set[:select]


def create_C1(data_set):
    """
    Create frequent candidate 1-itemset C1 by scaning data set.
    :param data_set: A list of transactions. Each transaction contains several items.
    :return C1: A set which contains all frequent candidate 1-itemsets
    """
    C1 = set()
    for trans in data_set:
        for item in trans:
            item_set = frozenset([item])
            C1.add(item_set)

    return C1


def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    :param Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
    :param Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    :return
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    """
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    :param Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    :param k: the item number of a frequent itemset.
    :return Ck: a set which contains all all frequent candidate k-itemsets.
    """
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k - 2] == l2[0:k - 2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Generate Lk by executing a delete policy from Ck.
    :param data_set: A list of transactions. Each transaction contains several items.
    :param Ck: A set which contains all all frequent candidate k-itemsets.
    :param min_support: The minimum support.
    :param support_data: A dictionary. The key is frequent itemset and the value is support.
    :return Lk: A set which contains all all frequent k-itemsets.
    """
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


def generate_L(data_set, k, min_support):
    """
    Generate all frequent itemsets.
    :param data_set: A list of transactions. Each transaction contains several items.
    :param k: Maximum number of items for all frequent itemsets.
    :param min_support: The minimum support.
    :return L: The list of Lk.
    :return support_data: A dictionary. The key is frequent itemset and the value is support.
    """
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k + 1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    """
    Generate big rules from frequent itemsets.
    :param L: The list of Lk.
    :param support_data: A dictionary. The key is frequent itemset and the value is support.
    :param min_conf: Minimal confidence.
    :return big_rule_list: A list which contains all big rules. Each big rule is represented
                       as a 3-tuple.
    """
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list


def pruning(Lk, data_set, all_sent, support_data, dist_word, min_sent_count):
    all_phrase = []
    for freq_set in Lk:
        for phrase in freq_set:
            if phrase not in all_phrase:
                all_phrase.append(phrase)

    for item in all_phrase:
        compact_count = 0
        p_support = 0
        words = item.split()
        for sent, trans in zip(all_sent, data_set):
            if item in trans:
                # p-support for single word
                if len(words) == 1:
                    p_support += 1
                # compact-count for multiple word
                else:
                    for i in range(len(words) - 1):
                        if words.index(words[i + 1]) - words.index(words[i]) <= dist_word:
                            compact_count += 1
        # compact-count < 2 or p-support <3: delete word
        if compact_count < min_sent_count or p_support < 3:
            all_phrase.remove(item)

    # update Lk and support_data
    new_Lk = set()
    new_support_data = {}
    for freq_set in Lk:
        cp_set = freq_set.copy()
        freq_set = set(freq_set)
        for phrase in cp_set:
            if len(phrase.split()) > 1 and phrase not in all_phrase:
                freq_set.remove(phrase)
        tmp_set = frozenset(freq_set)
        new_Lk.add(tmp_set)
        new_support_data[tmp_set] = support_data[cp_set]

    return new_Lk, new_support_data


def get_near_adj(words, phrase, pos):
    """
    get adjective (opinion word) near by phrase (freq aspect)
    :param words: A list of words (a sentence)
    :param phrase: freq aspect
    :param pos: POS dictionary of sentence
    :return near_adj: opinion word
    """
    min_dist = len(words)
    near_adj = phrase
    for idx, w in enumerate(words):
        if pos[w] == 'JJ' or pos[w] == 'JJR' or pos[w] == 'JJS':
            w_dist = abs(idx - words.index(phrase))
            if w_dist < min_dist:
                min_dist = w_dist
                near_adj = w
    # if not found, return phrase
    return near_adj


def get_opinion_word(Lk, all_sent, all_pos):
    """
    get opinion word by freq_set
    :param Lk: freq_set
    :param all_sent: a list of all sentences
    :param all_pos: a list of POS dict of all sentences
    :return opinion_set: a set of opinion word
    """
    all_phrase = set()
    for freq_set in Lk:
        for phrase in freq_set:
            all_phrase.add(phrase)

    opinion_set = set()
    for phrase in all_phrase:
        for sent, pos in zip(all_sent, all_pos):
            words = sent.split()
            # get adj near by phrase in freq_set
            if phrase in words:
                near_adj = get_near_adj(words, phrase, pos)
                # not found adj when near_adj == phrase
                if near_adj != phrase:
                    opinion_set.add(near_adj)

    return opinion_set


# TODO: get near noun or noun phrase according to data_set
def get_near_noun(words, phrase, pos):
    """
    get adjective (opinion word) near by phrase (freq aspect)
    :param words: A list of words (a sentence)
    :param phrase: freq aspect
    :param pos: POS dictionary of sentence
    :param data: extracted noun or noun phrase of sentence
    :return near_noun: infreq aspect
    """
    stop_words = stopwords.words('english')
    min_dist = len(words)
    near_noun = phrase
    for idx, w in enumerate(words):
        if pos[w] == 'NN' or pos[w] == 'NNS':
            if w not in stop_words:
                w_dist = abs(idx - words.index(phrase))
                if w_dist < min_dist:
                    min_dist = w_dist
                    near_noun = w
    # if not found, return phrase
    return near_noun


def get_infreq_set(Lk, opinion_set, all_sent, all_pos):
    """
    get infreq_set by opinion word
    :param Lk: freq_set
    :param all_sent: a list of all sentences
    :param all_pos: a list of POS dict of all sentences
    :return:
    """
    all_phrase = set()
    for freq_set in Lk:
        for phrase in freq_set:
            all_phrase.add(phrase)

    infreq_set = set()
    for sent, pos in zip(all_sent, all_pos):
        words = sent.split()
        # if has no frequent features(noun or noun phrase) and has opinion words
        if len(set(words) & all_phrase) == 0 and len(set(words) & opinion_set) > 0:
            phrases = list(set(words) & opinion_set)
            for phrase in phrases:
                # find the nearest noun or noun phrase
                near_noun = get_near_noun(words, phrase, pos)
                if near_noun != phrase:
                    infreq_set.add(near_noun)

    return infreq_set


def get_sent_aspect(freq_set, infreq_set, support_data, sent_data):
    """
    get aspect of sentence
    :param freq_set: a set of frequent candidate aspect
    :param infreq_set: a set of infrequent candidate aspect
    :param support_data: a dict of freq_set
    :param sent_data: extracted noun or noun phrase of sentence
    :return max_aspect: most possible aspect
    """
    max_support = 0
    max_aspect = []
    sent_set = set(sent_data)

    # get aspect from freq_set with max support
    for item in freq_set:
        if len(item & sent_set) > 0:
            if support_data[item] > max_support:
                max_support = support_data[item]
                max_aspect = list(item & sent_set)

    if max_aspect:  # return if found aspect
        return max_aspect

    max_aspect = list(infreq_set & sent_set)  # get from infreq_set

    # if max_aspect:  # return if found aspect
    #     return max_aspect

    # return max_aspect ([]) if not found aspect
    return max_aspect


if __name__ == '__main__':
    # tag = ['pos', 'neg', 'clas']
    tag = ['clas']
    k = 2
    select = None  # None: select all
    min_support = 0.002

    '''get all data'''
    pos_sent, pos_pos, pos_data_set = load_data_set('pos', select)
    neg_sent, neg_pos, neg_data_set = load_data_set('neg', select)
    clas_sent, clas_pos, clas_data_set = load_data_set('clas', select)

    # all_sent = pos_sent + neg_sent + clas_sent
    # all_pos = pos_pos + neg_pos + clas_pos
    # data_set = pos_data_set + neg_data_set + clas_data_set
    all_sent = pos_sent
    all_pos = pos_pos
    data_set = pos_data_set

    print('=' * 100)
    print('Begin generate Lk')
    L, support_data = generate_L(data_set, k=k, min_support=min_support)

    Lk = set()
    for l_k in L:
        if len(l_k) > 0:
            Lk = l_k

    print('=' * 100)
    # print('frequent ' + str(len(list(Lk)[0])) + '-itemsets\t\t\tsupport')
    # print('=' * 100)
    # for freq_set in Lk:
    #     print(list(freq_set), support_data[freq_set])
    # print('=' * 100)
    print('total number:', len(Lk))

    print('=' * 100)
    print('Begin compactness pruning...')
    Lk, support_data = pruning(Lk, data_set, all_sent, support_data, dist_word=3, min_sent_count=2)
    # print('=' * 100)
    # print('After pruning, Lk:')
    # for freq_set in Lk:
    #     print(list(freq_set), support_data[freq_set])
    # print('=' * 100)
    print('total freq_set number:', len(Lk))

    print('=' * 100)
    print('Begin finding opinion words...')
    opinion_set = get_opinion_word(Lk, all_sent, all_pos)
    # print(list(opinion_set))
    # print('=' * 100)
    print('total opinion word:', len(list(opinion_set)))

    print('=' * 100)
    print('Begin finding infreq aspect...')
    infreq_set = get_infreq_set(Lk, opinion_set, all_sent, all_pos)
    # print(list(infreq_set))
    # print('=' * 100)
    print('total infreq aspect:', len(list(infreq_set)))

    print('=' * 100)
    print('Begin extract aspect...')
    for t in tag:
        nor_file = 'data/nor_data/nor_{}.csv'.format(t)
        aspect_file = 'data/aspect_data/aspect_{}.csv'.format(t)
        if t == 'pos':
            # save_sent = pos_sent
            save_data_set = pos_data_set
        elif t == 'neg':
            # save_sent = neg_sent
            save_data_set = neg_data_set
        else:
            # save_sent = clas_sent
            save_data_set = clas_data_set
        with open(aspect_file, 'w') as file:
            nor_sent = open(nor_file, 'r').read().strip().split('\n')
            for sent, sent_data in tqdm(zip(nor_sent, save_data_set)):
                aspect = get_sent_aspect(Lk, infreq_set, support_data, sent_data)
                file.write(sent + '\n')
                file.write('\t'.join(aspect) + '\n')
        print('extract aspect finished! Saved in {}'.format(aspect_file))
    print('=' * 100)
