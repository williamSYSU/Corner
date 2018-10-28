from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import os
import getpass

user_name = getpass.getuser()
os.environ[
    'CLASSPATH'] = '/home/{}/stanford/postagger/stanford-postagger.jar:' \
                   '/home/{}/stanford/parser/stanford-parser.jar:' \
                   '/home/{}/stanford/parser/stanford-parser-3.9.2-models.jar'.format(
    user_name, user_name, user_name)

tag = 'pos'
idx = 0
# file_name = 'data/normalize_{}_piece/nor_{}_{}.csv'.format(tag, tag, idx)
file_name = 'data/tmp_file.csv'

with open(file_name, 'r') as file:
    sentences = file.read().strip().split('\n')

stop_words = stopwords.words('english')
eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

print('=' * 100)
print('current tag: {}, file idx: {}'.format(tag, idx))

'''POS'''
print('=' * 100)
print('Starting POS...')
pos_sent = []
for sent in tqdm(sentences):
    pos_sent.append(list(eng_parser.parse(
        [w for w in sent.split()]))[0])

'''filter noun phrase & NLTK stemming'''
print('=' * 100)
print('Starting filter and stemming...')
cleaned_sent = []
for sent in tqdm(pos_sent):
    wnl = WordNetLemmatizer()
    tmp_sent = []
    for s in sent.subtrees(lambda t: t.height() <= 4 and t.label() == 'NP'):
        '''clean stop words & stemming'''
        tmp = [wnl.lemmatize(w, pos='n') for w in s.leaves() if w not in stop_words]
        '''lenght <= 3 & filter repeated list'''
        if 0 < len(tmp) <= 3 and tmp not in tmp_sent:
            tmp_sent.append(tmp)
    cleaned_sent.append(tmp_sent)

'''save file'''
save_file = 'data/trans_{}_piece/trans_{}_{}.csv'.format(tag, tag, idx)
with open(save_file, mode='w') as file:
    for idx, sent in enumerate(sentences):
        file.write(sent + '\n')
        file.write('\t'.join([' '.join(NP) for NP in cleaned_sent[idx]]) + '\n')
print('Finish!')
