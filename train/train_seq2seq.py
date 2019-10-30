import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from common.optimizer import Adam
from common.trainer import Trainer
from data import util
from seq2seq import Seq2seq

# data loading
(x_train, t_train), (x_test, t_test) = util.load_data('date.txt')
char_to_id, id_to_char = util.get_vocab()

# inverse data
# x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# hyper parameters
vocab_size   = len(char_to_id)
wordvec_size = 16
hidden_size  = 256
batch_size   = 128
max_epoch    = 10
max_grad     = 5.0

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()

trainer = Trainer(model, optimizer)