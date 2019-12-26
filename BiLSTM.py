from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell


class ABLSTM(object):
    def _int_(self, config):
          self.max_len = config["max_len"]
          self.hidden_size = config["hidden_size"]
          self.vocab_size = config["vocab_size"]
          self.embedding_size = config["embedding_size"]

