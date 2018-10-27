# from nltk.tag import StanfordPOSTagger
#
# eng_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# print(eng_tagger.tag('What is the airspeed of an unladen swallow ?'.split()))

from nltk.parse.stanford import StanfordParser
from nltk.corpus import stopwords
from nltk.draw.tree import draw_trees
from nltk.tree import Tree
import config
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')

eng_parser = StanfordParser(model_path=u'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
sent = 'I am absolutely in awe of this auto-focus camera this autofocus camera is awesome'

res = eng_parser.parse([w for w in sent.lower().split()])
lst_res = list(res)[0]
# print(lst_res.pos())
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
print(cleaned_sent)
