import sys
sys.path.append('..')
import os
import numpy

id_to_char = []
char_to_id = []

def get_vocab():
	return char_to_id, id_to_char 


def _update_vocab(txt):
    for i, char in enumerate(list(txt)):
        if char not in char_to_id:
            idx = len(char_to_id)
            char_to_id[char] = idx
            id_to_char[idx]  = char


def load_data(file_name, seed=1234):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name

    if not os.path.exists(file_path):
        raise IOError(file_name + ': No such file')

    inputs, outputs = [], []

    for line in open(file_path, 'r'):
        idx = line.find('_')
        inputs.append(line[:idx])
        outputs.append(line[idx:-1])

    for i in range(len(inputs)):
        j, k = inputs[i], outputs[i]
        _update_vocab(j)
        _update_vocab(k)
    
    x = np.array((len(inputs), len(inputs[0])), dtype=np.int)
    t = np.array((len(outputs), len(outputs[0])), dtype=np.int)

    for i in range(len(inputs)):
        x[i] = [char_to_id[char] for char in list(inputs[i])]
        y[i] = [char_to_id[char] for char in list(outputs[i])]
    
    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # split for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)  