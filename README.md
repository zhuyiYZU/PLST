This is the repository for the "Prompt-Learing for Short Text Classification".

see paper one arxiv : https://arxiv.org/abs/2202.11345

First you need to install OpenPrompt.
see https://github.com/thunlp/OpenPrompt

Then you need to use the methods to filter tag words,In this process, you need to download the corresponding vocabulary.
These words need to be put scripts/TextClassification，and modify the corresponding position.

Also, you can put your own dataset in datasets/TextClassification.

example shell scripts:

python main.py  --verbalizer cpt


code cpt_verbalizer inspired from https://github.com/abhishek0318/Probase/tree/af6cb903c39b0fe2bf1326fdd9673e77386e087f
