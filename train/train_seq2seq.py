import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from common.optimizer import Adam
from common.trainer import Trainer
from data import sequence
from seq2seq import Seq2seq

# data loading
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# inverse data
is_reverse = False
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# hyper parameters
vocab_size   = len(char_to_id)
wordvec_size = 16
hidden_size  = 256
batch_size   = 128
max_epoch    = 10
max_grad     = 5.0
eval_interval = 20

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()

trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):

    trainer.fit(x_train, t_train, max_epoch=1, batch_size, max_grad, eval_interval)

    # validation test
    correct_count = 0
    for i in range(len(x_test)):
        inputs, correct = x_test[[i]], t_test[[i]] # neither x_test[i] nor x_test[:][i]
        
        verbose = i < 10 # print out for the first some steps
        correct_count += model.evaluate(inputs, correct, id_to_char, verbose, is_reverse)

    acc = float(correct_count) / len(x_test)
    acc_list.append(acc)
    print('Accuracy: %.3f%%' % (acc * 100))

model.save_params()

# visualization
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()
