from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format('./data/crawl-300d-2M-subword/crawl-300d-2M-subword.vec', binary=False)

# print(model.similar_by_word('news', 50))


l_word = []
res1 = []
res2 = []
res3 = []
with open('verbalizer/newstitle_verbalizer.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        l_word.append(line.split(',')[0].strip())
        res1.append(line.split(',')[1:])
#print(l_word)
with open('result/newstitle_bert_result.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res2.append(line.split(',')[1:])

with open('result/news_fasttext_result.txt', 'w') as f:

    for i, word in enumerate(l_word):
        l_temp = []
        for word1 in res1[i]:
            if word1 in model:
                l_temp.append((word1, model.similarity(word, word1)))
        #print(l_temp)
        res_temp = [j[0] for j in sorted(l_temp, key=lambda x: x[1], reverse=True)[:15]]
        f.write(word + ',' + ','.join(res_temp) + '\n')
        res3.append(set(res_temp) | set(res2[i]))
#print(res3)
with open('result/newstitle_all_result.txt', 'w') as f:
    for i, word in enumerate(l_word):
        #print(i + '---' + word)
        f.write(word + ',' + ','.join(list(res3[i])) + '\n')

