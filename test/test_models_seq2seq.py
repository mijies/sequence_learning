import unittest
import sys
sys.path.append('..')
from models.seq2seq import np, Seq2seq, Encoder, Decoder

class TestModelsEncoder(unittest.TestCase):
    """test class for models/seq2seq.py
    """
    def test_forward_backward(self):
        V, D, H = 12, 6, 3
        idx = np.array([ 
            [1, 3, 1, 2] , [2, 0, 1, 0] # shape is (2, 4)
        ])
        N, T = idx.shape

        layer = Encoder(V, D, H)
        hs = layer.forward(idx)
        self.assertEqual(hs.shape, (N, H)) # return hs[:, -1, :]
        self.assertEqual(layer.hs.shape, (N, T, H))

        dh = np.ones(N*H).reshape(N, H) # (N, H)
        dout = layer.backward(dh)
        self.assertEqual(dout, None)


class TestModelsDecoder(unittest.TestCase):
    """test class for models/seq2seq.py
    """
    def test_forward_backward(self):
        V, D, H, N, tT = 12, 6, 3, 2, 4 # tT is sample_size

        ts = np.arange(N*tT).reshape(N, tT)
        docoder_xs= ts[:, :-1]
        hs = np.ones(N*H).reshape(N, H)

        layer = Decoder(V, D, H)
        score = layer.forward(docoder_xs, hs)
        self.assertEqual(score.shape, (N, tT-1, V))

        dx = np.ones(N*(tT-1)*V).reshape(N, tT-1, V) # (N, tT-1, V)
        dh = layer.backward(dx)
        self.assertEqual(dh.shape, (N, H))


    def test_generate(self):
        V, D, H, N = 12, 6, 3, 1

        correct = np.arange(N*4)
        start_id = correct[0] # separator
        correct  = correct[1:]
        hs = np.ones(N*H).reshape(N, H)

        layer = Decoder(V, D, H)
        samples = layer.generate(hs, start_id, len(correct))
        self.assertEqual(len(samples), len(correct))



class TestModelsSeq2seq(unittest.TestCase):
    """test class for models/seq2seq.py
    """
    def test_forward_backward(self):
        V, D, H, tT = 12, 6, 3, 5

        idx = np.array([ 
            [1, 3, 1, 2] , [2, 0, 1, 0] # shape is (2, 4)
        ])
        N, T = idx.shape
        ts = np.arange(N*tT).reshape(N, tT)

        model = Seq2seq(V, D, H)
        loss = model.forward(idx, ts)
        self.assertEqual(type(loss), np.float64)

        model.backward()


    def test_generate(self):
        V, D, H, N, T = 12, 6, 3, 1, 2

        correct = np.arange(N*4)
        start_id = correct[0] # separator
        correct  = correct[1:]
        xs = np.ones(D).reshape(1, D).astype('int')

        model = Seq2seq(V, D, H)
        samples = model.generate(xs, start_id, len(correct))
        print(samples)
        self.assertEqual(len(samples), len(correct))


    def test_evaluate(self):
        V, D, H, N, T = 12, 6, 3, 1, 2

        correct = np.arange(N*4)
        inputs = np.ones(D).reshape(1, D).astype('int')
        id_to_char = {}
        for i in range(V):
            id_to_char[i] = str(i)

        model = Seq2seq(V, D, H)
        correct_count = model.evaluate(inputs, correct, id_to_char, verbose=False, is_reverse=False)
        self.assertEqual(correct_count, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)