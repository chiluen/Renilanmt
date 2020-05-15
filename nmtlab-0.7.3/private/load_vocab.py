import sys
sys.path.append(".")
from explab import *

from nmtlab import Vocab
import tensorflow as tf

vocab_path = "{}/iwslt14.tokenized.de-en/iwslt14.en.bpe40k.vocab".format(MAINLINE_ROOT)
vocab = Vocab(vocab_path)
index_table = vocab.get_index_table()

inp = tf.placeholder_with_default(["UNK", "system", "sadfwef"], shape=(3,))
ss = tf.Session()
ss.run(index_table.init)
print(ss.run(index_table.lookup(inp)))

