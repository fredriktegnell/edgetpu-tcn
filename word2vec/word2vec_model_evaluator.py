from gensim.models import Word2Vec

# Load the trained Word2Vec MODEL
MODEL = Word2Vec.load('word2vec/model/word2vec_model.bin')

# Output file path for evaluation results
OUTPUT = 'word2vec/output.txt'

def evaluate_similarity(word1, word2):
    """
    Evaluate and log the semantic similarity between two words.
    
    Parameters:
    - word1, word2 (str): Words to be compared.
    """
    similarity_score = MODEL.wv.similarity(word1, word2)
    with open(OUTPUT, 'a') as file:
        file.write(f"Similarity between '{word1}' and '{word2}': {similarity_score}\n")

def evaluate_analogies(word1, word2, word3, word4):
    """
    Evaluate and log word analogies.
    
    Parameters:
    - word1, word2, word3, word4 (str): Words for analogy comparison.
    """
    analogy_result = MODEL.wv.most_similar(positive=[word2, word3], negative=[word1])
    with open(OUTPUT, 'a') as file:
        file.write(f"Analogical reasoning: '{word1}' is to '{word2}' as '{word3}' is to '{word4}'\n")
        for word, similarity in analogy_result:
            file.write(f"'{word}': {similarity}\n")

def evaluate_vector_arithmetic(positive_words, negative_words):
    """
    Evaluate and log word vector arithmetic.
    
    Parameters:
    - positive_words (list): List of words to be added.
    - negative_words (list): List of words to be subtracted.
    """
    result = MODEL.wv.most_similar(positive=positive_words, negative=negative_words)
    with open(OUTPUT, 'a') as file:
        file.write("Vector arithmetic:\n")
        for word, similarity in result:
            file.write(f"'{word}': {similarity}\n")

def evaluate_oov_words(words):
    """
    Evaluate and log whether words are in the MODEL's vocabulary or out-of-vocabulary.
    
    Parameters:
    - words (list): List of words to be checked.
    """
    with open(OUTPUT, 'a') as file:
        in_vocab = [word for word in words if word in MODEL.wv]
        out_vocab = [word for word in words if word not in MODEL.wv]

        if in_vocab:
            file.write("Words in vocabulary:\n")
            for word in in_vocab:
                file.write(f"'{word}'\n")
        
        if out_vocab:
            file.write("Out-of-vocabulary words:\n")
            for word in out_vocab:
                file.write(f"'{word}'\n")

# Should have high similarity score
evaluate_similarity('film', 'movie')

# The word 'man' is to the word 'king' as the word 'woman' is to an unknown word.
# The unknown word is expected to be similar to 'queen'.
evaluate_analogies('man', 'king', 'woman', 'queen')

# faster + daughter - son = mother
evaluate_vector_arithmetic(['father', 'daughter'], ['son'])

evaluate_oov_words(['movie', 'actor', 'story', 'cinematography', 'narrative', 'soundtrack',
                    'protagonist', 'filmography', 'blockbuster', 'parasitology', 'neuroscience',
                    'nanotechnology', 'Hermione', 'Gandalf', 'Tyrion', 'quantum', 'algorithm',
                    'biotechnology'])
