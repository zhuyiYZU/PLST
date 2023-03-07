from openprompt_.data_utils.text_classification_dataset import CustomProcessor_Temp as CustomProcessor
from tqdm import tqdm
import argparse
import pickle
import string
import spacy
import re
import sys

class Chunk:
    """Class containing chunk and its root."""

    def __init__(self, chunk, chunk_root):
        self.chunk = chunk
        self.chunk_root = chunk_root

    def __eq__(self, other):
        return self.chunk == other.chunk and self.chunk_root == other.chunk_root

    def __hash__(self):
        return hash((self.chunk, self.chunk_root))

    def __str__(self):
        return "text: " + self.chunk + ' root: ' + self.chunk_root


class Probase:
    """Class to generate probase

    You must run syntactic extraction before execution build_knowledge_base
    """

    def __init__(self):
        # dictionary which stores the count of a superconcept
        # in the knowledge base.
        self.n_super_concept = {}

        # dictionary which stores the count of pairs in knowledge base
        # that has 'x' and 'y' as superconcept and subconcept respectively.
        self.n_super_concept_sub_concept = {}

        self.knowledge_base_size = 1

        self.super_concepts_corpus = []
        self.sub_concepts_corpus = []

    def syntactic_extraction(self):
        """Extracts subconcepts and superconcepts from the corpus.

        Args:
            filename: path of corpus.

        Returns:
            nothing.
        """

        nlp = spacy.load('en_core_web_sm')

        dataset = {}
        
        with open(args.intput_folder+'/classes.txt','r') as f:
                labels = f.read().split('\n')[:-1]
        dataset['train'] = CustomProcessor(labels).get_examples(args.intput_folder,"train")
        class_labels = CustomProcessor(labels).get_labels()

        for inputs in tqdm(dataset['train']):
                sentence,label = inputs.text_a,inputs.label
                sentence_parsed = False
                super_concepts = []
                parsed_sentence = nlp(sentence)
                
                element = Chunk(str(label),str(label))
                self.super_concepts_corpus.append([element])
                for chunk in parsed_sentence.noun_chunks:
                    if (chunk.root.tag_ == 'NNS' or chunk.root.tag_ == 'NNPS'):
                        super_concepts.append(
                                    Chunk(chunk.text.lower(), chunk.root.text.lower()))
                if super_concepts:
                    self.super_concepts_corpus.append(super_concepts)
                    sub_concepts = []
                    sub_concepts.append(str(label))
                    self.sub_concepts_corpus.append(sub_concepts)

    def load_concepts_corpus(self, super_concepts_corpus, sub_concepts_corpus):
        """Load concepts.

        Args:
            super_concepts_corpus: list of super_concept chunks.
            sub_concepts_corpus: list of sub_concepts strings.

        Returns:
            nothing.
        """

        self.super_concepts_corpus = super_concepts_corpus
        self.sub_concepts_corpus = sub_concepts_corpus
            
    def p_x(self, super_concept):
        """Return probability of superconcept in knowledgebase.

        Args:
            super_concept: superconcept object.

        Returns: 
            The percentage of pairs that have 'super_concept' as the 
            superconcept in knowledge base.
        """

        probability = self.n_super_concept.get(
            super_concept.chunk, 0) / self.knowledge_base_size
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept.get(
                super_concept.chunk_root, 0) / self.knowledge_base_size
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def p_y_x(self, sub_concept, super_concept):
        """Return probability of sub_concept given a superconcept in knowledgebase.

        Args:
            super_concept: superconcept object.
            sub_concept: subconcept string

        Returns: 
            The percentage of pairs in knowledgebase that have `sub_concept` as
            the sub-concept given `super_concept` is the super-concept.
        """

        probability = self.n_super_concept_sub_concept.get(
            (super_concept.chunk, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk, 1)
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept_sub_concept.get(
                (super_concept.chunk_root, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk_root, 1)
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def super_concept_detection(self, super_concepts, sub_concepts):
        """Return most likely super concept.
        Args:
            superconcepts: list of superconcepts of a sentence.
            subconcepts: list of subconcepts of a sentence.
        Returns:
            Most likely super concept object.
        """

        likelihoods = {}
        for super_concept in super_concepts:
            probability_super_concept = self.p_x(super_concept)
            likelihood = probability_super_concept
            for sub_concept in sub_concepts:
                probability_y_x = self.p_y_x(sub_concept, super_concept)
                likelihood *= probability_y_x
            likelihoods[super_concept] = likelihood

        sorted_likelihoods = sorted(
            likelihoods.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_likelihoods) == 1:
            return super_concepts[0]
        if sorted_likelihoods[1][1] == 0:
            return sorted_likelihoods[0][0]
        ratio = sorted_likelihoods[0][1] / sorted_likelihoods[1][1]
        if ratio > self.threshold_super_concept:
            return sorted_likelihoods[0][0]
        else:
            return None

    def sub_concept_detection(self, most_likely_super_concept, sub_concepts):
        """Return a list of filtered sub concepts.

        Args:
            most_likely_super_concept: super-concept object.
            sub_concepts: list of sub-concept strings

        Returns:
            filtered_sub_concepts: list of filtered sub_concepts.
        """

        # find k till which we can consider subconcepts
        for k, sub_concept in enumerate(sub_concepts):
            if self.p_y_x(sub_concept, most_likely_super_concept) < self.threshold_k:
                break
        k = max(k, 1)
        return sub_concepts[:k]

    @staticmethod
    def increase_count(dictionary, key):
        """Increases count of key in dictionary"""
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    def build_knowledge_base(self, epsilon=0.01, threshold_super_concept=1.1, threshold_k=0.01):
        """Takes in list of subconcepts and superconcepts and generates probase.

        Args:
            super_concepts_corpus: list of list of super_concepts in a sentence.
            sub_concepts_corpus: list of list of sub_concepts in a sentence.

        Returns:
            nothing.
        """

        self.threshold_super_concept = threshold_super_concept
        self.epsilon = epsilon
        self.threshold_k = threshold_k

        iteration = 0
        while True:
            iteration += 1
            n_super_concept_sub_concept_new = {}
            n_super_concept_new = {}
            knowledge_base_size_new = 1
            
            for super_concepts, sub_concepts in tqdm(zip(self.super_concepts_corpus, self.sub_concepts_corpus)):
                most_likely_super_concept = self.super_concept_detection(
                    super_concepts, sub_concepts)
                if most_likely_super_concept is None:
                    continue
                sub_concepts = self.sub_concept_detection(
                    most_likely_super_concept, sub_concepts)
                for sub_concept in sub_concepts:
                    self.increase_count(n_super_concept_sub_concept_new, (most_likely_super_concept.chunk, sub_concept))
                    self.increase_count(n_super_concept_new, most_likely_super_concept.chunk)
                    knowledge_base_size_new += 1
            size_old = len(self.n_super_concept_sub_concept)
            size_new = len(n_super_concept_sub_concept_new)
            if size_new == size_old:
            	
                break
            else:
                self.n_super_concept_sub_concept = n_super_concept_sub_concept_new
                self.n_super_concept = n_super_concept_new
                self.knowledge_base_size = knowledge_base_size_new

    def save_file(self):
        """Saves probase as filename in text format"""

        liste = []
        elements = []
        for key, value in self.n_super_concept_sub_concept.items():
                liste.append((key[1],str(value),key[0]))
                if key[1] not in elements:
                    elements.append(key[1])
        liste = sorted(liste,key=lambda x:x[1],reverse=True)
        liste = sorted(liste,key=lambda x:x[0])
        
        string = ''
        old_i = '0'
        for i,j,k in liste:
        	if k not in elements:
        		if string == '':
	        		string = string + k
        		elif string[-1] == '\n':
	        		string = string + k
        		else:
	        		string = string + ',' + k 
        			
        	if i != old_i:
        		old_i =i
        		string = string + '\n'
        with open(args.output_folder+"cpt_vebalizer.txt","w") as f:
                f.write(string) 
if True:
    parser = argparse.ArgumentParser(description='Generate probase')
    parser.add_argument('--intput_folder', type=str, default='datasets/TextClassification/custom/')
    parser.add_argument('--output_folder', type=str, default='scripts/TextClassification/custom/')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Eplison')
    parser.add_argument('--threshold_super_concept', type=float, default=1.1, help='Threshold (super-concept)')
    parser.add_argument('--threshold_k', type=float, default=0.01, help='Threshold (k)')
    args = parser.parse_args()
    probase = Probase()

    print('Extracting super-concepts and sub-concepts..')
    probase.syntactic_extraction()
    #concepts = pickle.dump((probase.super_concepts_corpus, probase.sub_concepts_corpus), open('./data/concepts.pkl', 'wb'))
    #concepts = pickle.load(open('./data/concepts.pkl', 'rb'))
    #probase.load_concepts_corpus(*concepts)
    print('\nBuilding knowledge base..')
    probase.build_knowledge_base(epsilon=args.epsilon, threshold_super_concept=args.threshold_super_concept, threshold_k=args.threshold_k)
    print('\nSaving probase as text file..')
    probase.save_file()
