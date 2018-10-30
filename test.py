# from nltk.tag import StanfordPOSTagger
#
# eng_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# print(eng_tagger.tag('What is the airspeed of an unladen swallow ?'.split()))

from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from tqdm import tqdm


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
    sentence = "the sd memory card is missing"

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
def filter_test():
    tag = 'neg'
    for i_file in tqdm(range(20)):
        file_name = 'data/{}_sent/{}_sent_{}.csv'.format(tag, tag, i_file)
        save_file = 'data/{}_sent_clean/{}_sent_clean_{}.csv'.format(tag, tag, i_file)
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
            # print(words)
            # print(dict_pos)
            for i, w in enumerate(words):
                if dict_pos[w] == 'NNS':
                    # print('\nsent {}'.format(idx))
                    # print('before:', w)
                    words[i] = wnl.lemmatize(w, pos='n')
                    # print('after:', words[i])
                    # dict_pos.pop(w)
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


if __name__ == '__main__':
    pos_test()
