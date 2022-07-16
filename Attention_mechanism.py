# The attention mechanism was introduced to improve the performance of the encoder-decoder model for machine translation.
# The idea behind the attention mechanism was to permit the decoder to utilize the most relevant parts of the input sequence in a flexible manner,
# by a weighted combination of all of the encoded input vectors, with the most relevant vectors being attributed the highest weights. 

# We wanna inspect 1- How the attention mechanism uses a weighted sum of all of the encoder hidden states to flexibly focus the attention of the decoder
# to the most relevant parts of the input sequence.

# 2- How the attention mechanism can be generalized for tasks where the information may not necessarily be related in a sequential fashion.

# 3- How to implement the general attention mechanism in Python with NumPy and SciPy.

from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax
 
# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])
 
# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])
 
# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = words @ W_Q
K = words @ W_K
V = words @ W_V

# scoring the query vectors against all key vectors
scores = Q @ K.transpose()

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = weights @ V

print(attention)
