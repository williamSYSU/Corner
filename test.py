from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from tqdm import tqdm

import config
import getpass
import os

user_name = getpass.getuser()
os.environ[
    'CLASSPATH'] = '/home/{}/stanford/postagger/stanford-postagger.jar:' \
                   '/home/{}/stanford/parser/stanford-parser.jar:' \
                   '/home/{}/stanford/parser/stanford-parser-3.9.2-models.jar'.format(
    user_name, user_name, user_name)


def apriori_test():
    tag = 'pos'
    all_lines = []
    trans_lines = []
    for idx in range(20):
        if idx == 0:
            continue
        file_name = 'data/trans_{}_piece/trans_{}_{}.csv'.format(
            tag, tag, idx)
        with open(file_name, 'r') as file:
            all_lines.extend(file.read().strip().split('\n'))
    for idx, line in enumerate(all_lines):
        if idx % 2 == 1:
            trans_lines.append(line)


def pos_test():
    stop_words = stopwords.words('english')
    eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
    eng_parser.java_options = '-mx3000m'
    sentence = "so now i run out take a few shot then run back in the house and xfer them to the pc"

    res = eng_parser.parse([w for w in sentence.lower().split()])
    lst_res = list(res)[0]
    with open('data/tree_test.txt', 'w') as file:
        file.write(sentence + '\t')
        file.write(str(lst_res) + '\t')
    # print(lst_res)
    lst_res.pretty_print()

    # lst_res.remove(Tree('NN', ['camera']))
    cleaned_sent = []
    for sent in lst_res:
        wnl = WordNetLemmatizer()
        tmp_sent = []
        for s in sent.subtrees(lambda t: t.height() <= 4 and t.label() == 'NP'):
            '''clean stop words & stemming'''
            tmp = [wnl.lemmatize(w, pos='n') for w in s.leaves() if w not in stop_words]
            '''lenght <= 3 & filter repeated list'''
            if 0 < len(tmp) <= 3 and tmp not in tmp_sent:
                tmp_sent.append(tmp)

        cleaned_sent.append(tmp_sent)

    # get opinion word
    # for w in cleaned_sent[0]:
    #     print(w)
    #     words = sentence.split()
    #     min_dist = len(words)
    #     min_asp = w
    #     for s in lst_res.subtrees(lambda t: t.label() == 'JJ'):
    #         if abs(words.index(s.leaves()[0]) - words.index(w[0])) < min_dist:
    #             min_dist = pos_test
    #             min_asp = s.leaves()[0]
    #
    #     if min_asp == w:
    #         print('not found')
    #     else:
    #         print(min_asp)

    print(cleaned_sent)


# filter noun phrase and stemming
def filter_test(tag):
    tag = tag
    # times: number of file pieces
    if tag == 'neg':
        times = 20
    elif tag == 'pos':
        times = 50
    else:
        times = 1
    for i_file in tqdm(range(times)):
        # file_name = 'data/{}_sent/{}_sent_{}.csv'.format(tag, tag, i_file)
        # save_file = 'data/{}_sent_clean/{}_sent_clean_{}.csv'.format(tag, tag, i_file)
        file_name = 'data/clas_sent.csv'
        save_file = 'data/clean_clas.csv'
        stop_words = stopwords.words('english')
        with open(file_name, 'r') as file:
            all_lines = file.read().split('\t')

        all_sent = []
        all_tree = []
        for idx, line in enumerate(all_lines):
            if idx % 2 == 0:
                all_sent.append(line.strip())
            else:
                all_tree.append(Tree.fromstring(line.strip()))

        all_sent.remove('')
        wnl = WordNetLemmatizer()

        all_pos = []
        cleaned_sent = []
        for idx, (sent, tree) in tqdm(enumerate(zip(all_sent, all_tree))):
            dict_pos = dict(tree.pos())
            words = sent.split()
            for i, w in enumerate(words):
                if dict_pos[w] == 'NNS':
                    words[i] = wnl.lemmatize(w, pos='n')
                    dict_pos[words[i]] = 'NN'

            all_sent[idx] = ' '.join(words)
            all_pos.append(dict_pos)

            tmp_sent = []
            for s in tree.subtrees(lambda t: t.height() <= 4 and t.label() == 'NP'):
                '''clean stop words & stemming'''
                tmp = []
                for w in s.leaves():
                    if w not in stop_words:
                        # if have been stemming above
                        if w not in words:
                            tmp.append(wnl.lemmatize(w, pos='n'))
                        else:
                            tmp.append(w)
                '''length <= 3 & filter repeated list'''
                if 0 < len(tmp) <= 2 and tmp not in tmp_sent:
                    tmp_sent.append(tmp)
            cleaned_sent.append(tmp_sent)

        with open(save_file, 'w') as file:
            for sent, pos, cl_sent in zip(all_sent, all_pos, cleaned_sent):
                file.write(sent + '\n')
                file.write(str(pos) + '\n')
                file.write('\t'.join([' '.join(NP) for NP in cl_sent]) + '\n')


def update_clas_data():
    """add origin info into clas_aspect.csv (such as label and subjective info)"""
    raw_filename = 'data/nor_data/nor_clas.csv'
    tar_filename = 'data/aspect_data/unused_aspect_clas_retain.csv'
    save_filename = 'data/aspect_data/update_aspect_clas_retain.csv'

    raw_lines = open(raw_filename, 'r').read().strip().split('\n')
    tar_lines = open(tar_filename, 'r').read().strip().split('\n')
    tar_sents = [tar_lines[i] for i in range(len(tar_lines)) if i % 2 == 0]
    for idx, (raw_sent, tar_sent) in enumerate(zip(raw_lines, tar_sents)):
        items = raw_sent.strip().split()
        tmp_sent = [items[0], items[1], tar_sent]
        tar_lines[idx * 2] = ' '.join(tmp_sent)
        # tar_lines[idx * 2] = raw_sent

    with open(save_filename, 'w') as file:
        file.write('\n'.join(tar_lines))


def repeat_asp_sent(if_retain=False):
    """one aspect for one input sentence, remove sentence without aspect"""
    tag = ['pos', 'neg', 'clas']
    # tag = ['clas']
    retain = ''
    if if_retain:
        retain = '_retain'

    for t in tag:
        file_name = 'data/aspect_data/aspect_{}{}.csv'.format(t, retain)
        save_file = 'data/final_aspect_data/final_aspect_{}{}.csv'.format(t, retain)
        tmp = open(file_name, 'r').read().strip().split('\n')
        sentences = [tmp[i] for i in range(len(tmp)) if i % 2 == 0]
        aspects = [tmp[i] for i in range(len(tmp)) if i % 2 == 1]

        with open(save_file, 'w') as file:
            for sent, asp in zip(sentences, aspects):
                items = asp.split('\t')
                for i in items:  # ignore sentence withou aspect
                    if i != '':
                        file.write(sent + '\n')
                        file.write(i + '\n')


def retain_asp_sent():
    tag = ['pos', 'neg', 'clas']
    for t in tag:
        raw_file = 'data/clean_data/clean_{}.csv'.format(t)
        tar_file = 'data/aspect_data/aspect_{}.csv'.format(t)
        save_file = 'data/aspect_data/aspect_{}_retain.csv'.format(t)

        tmp = open(raw_file, 'r').read().strip().split('\n')
        raw_data = [tmp[i] for i in range(len(tmp)) if i % 3 == 2]

        tmp = open(tar_file, 'r').read().strip().split('\n')
        tar_sent = [tmp[i] for i in range(len(tmp)) if i % 2 == 0]
        tar_data = [tmp[i] for i in range(len(tmp)) if i % 2 == 1]

        with open(save_file, 'w') as file:
            for idx, (sent, data) in enumerate(zip(tar_sent, tar_data)):
                file.write(sent + '\n')
                if data != '':
                    file.write(data + '\n')
                else:
                    file.write(raw_data[idx] + '\n')


if __name__ == '__main__':
    # update_clas_data()
    repeat_asp_sent(if_retain=True)
