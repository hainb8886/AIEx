#https://www.nltk.org/
#Tokenize and tag some text:
#Ensure the necessary NLTK data files are downloaded:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

#1. Tokenization
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print("Tokens:", tokens)
# Output: ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning', 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']

#2. Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("POS Tags:", tagged)
print("POS Tags [a:b]:", tagged[0:6])
# Output: [('At', 'IN'), ('eight', 'CD'), ("o'clock", 'JJ'), ('on', 'IN'), ('Thursday', 'NNP'), ('morning', 'NN')]

#3. Named Entity Recognition (NER)
#This step identifies named entities (like people, organizations, locations). In this case, "Arthur" is recognized as a PERSON.
entities = nltk.chunk.ne_chunk(tagged)
print("Named Entities:", entities)

#4. Displaying a Parse Tree
from nltk.corpus import treebank
nltk.download('treebank')  # Run this only once
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()