import gensim
import logging
from gensim import corpora,models,similarities
model=gensim.models.keyedvectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True)
keyword = []
f_w1=open('most_similar.txt','w',encoding='utf-8')
with open('/home/ubuntu/PycharmProjects/prompt_studby/get_entity.csv','r',encoding='utf-8') as f:
    for ii,line in enumerate(f):
        line=line.split('\t')
        # print(line)
        entity=line[1].split(' ')
        # print(entity)
        # text=line[0]
        # print(text)
        label=line[-1].replace('\n','')
        # print(label)
        for i in entity:
            score = []
            y = model.similarity(i,label)
            score.append(y)
        keyword.append(entity[score.index(max(score))])

for i in keyword:
    f_w1.write(i + '\n')


