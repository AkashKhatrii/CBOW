import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


def get_eval_stats(emb_file):   
    wv = KeyedVectors.load_word2vec_format(datapath(emb_file), binary=False)
    sim = wv.evaluate_word_pairs("/scratch/general/vast/u1469481/cs6957/assignment1/models/test_files/wordsim_similarity_goldstandard.txt")
    analogy = wv.evaluate_word_analogies("/scratch/general/vast/u1469481/cs6957/assignment1/models/test_files/wordsim_similarity_goldstandard.txt")
    print(f"Word Similarity Test Pearson Correlation: {sim[0][0]}")
    print(f"Accuracy on Analogy Test: {analogy[0]}")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", "/scratch/general/vast/u1469481/cs6957/assignment1/models/embeddings.txt", type=str)
    args = vars(parser.parse_args())
    get_eval_stats(args["emb_file"])

