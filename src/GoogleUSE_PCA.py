'''
Text encoding via Google USE and PCA
'''

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.decomposition import PCA
import csv
import pandas as pd
import glob
module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def get_vectors(messages):

    # Reduce logging output.
    embeddings = []
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(embed(messages))

      for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        #message_embedding_snippet = ", ".join(
        #    (str(x) for x in message_embedding[:3]))
        #print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
        embeddings.append(message_embedding)
    return embeddings

def pca(embeddings):
    retVal = []
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(embeddings)
    for component in principalComponents:
        toAdd = ", ".join((str(x) for x in component))
        retVal.append(toAdd)
    return retVal

def main(folder):
    for filename in glob.glob(folder):
        print(filename)
        headline_vectors = []
        snippet_vectors = []
        with open(filename) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                headline_vectors.append(row["headline"])
                snippet_vectors.append(row["snippet"])
        headline_v = get_vectors(headline_vectors)
        snippet_v = get_vectors(snippet_vectors)
        csv_input = pd.read_csv(filename)
        pcah = pca(headline_v)
        csv_input['headline_vector'] = headline_v
        csv_input['snippet_vector'] = snippet_v
        csv_input['headline_vector_pca'] = pca(headline_v)
        csv_input['snippet_vector_pca'] = pca(snippet_v)
        csv_input.to_csv('EndPCA/' + filename.split("/")[1], index=False)
        print("finished 1 file")

if __name__ == '__main__':
    main("PrePCA/*.csv") # Input Google or NYtimes raw news folder
