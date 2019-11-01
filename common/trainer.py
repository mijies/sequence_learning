import sys
sys.path.append('..')
# import numpy
import time
import matplotlib.pyplot as plt

from common.util import np

# DONE
class Trainer:
    def __init__(self, model, optimizer):
        self.model     = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None


    def fit(self, x, t, max_epoch, batch_size, max_grad=None, eval_interval):
        data_size = len(x)
        max_iter = data_size // batch_size
        self.eval_interval = eval_interval

        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # shuffle
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iter in range(max_iter):
                batch_x = x[iter * batch_size:(iter+1) * batch_size]
                batch_t = t[iter * batch_size:(iter+1) * batch_size]

                loss = model.forward(batch_x, batch_t)
                total_loss += loss
                loss_count += 1

                model.backward()

                # concatenate shared params and grads into one
                params, grads = concat_duplicate(model.params, model.grads)
                # gradient clipping
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                optimizer.update(params, grads)
            
                # evaluation
                if (eval_interval is not None) and (iter // eval_interval) == 0:
                    elapsed_time = time.time() - start_time
                    avg_loss = total_loss / loss_count

                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (epoch + 1, iter + 1, max_iter, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
    

    def plot(self, ylim=None): # ylim : (int, int)
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='training')
        plt.xlabel('iterations : interval ' + str(self.eval_interval))
        plt.ylabel('loss')
        plt.show()



# params can share the objects. So sums up the gradients for computational efficiency
def concat_duplicate(params, grads):
    params, grads = params[:], grads[:] # deep copy

    while True:
        found = False
        param_size = len(params)

        for i in range(0, param_size -1):
            for j in range(1, param_size):

                if params[i] is params[j]: # if identical
                    found = True
                    grads[i] += grads[j] # add up gradients
                    params.pop(j)
                    grads.pop(j)

                # if all tranposed values are same
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and \
                     np.all(params[i].T == params[j]):
                    found = True
                    grads[i] += grads[j] # add up gradients
                    params.pop(j)
                    grads.pop(j)
                
                if found:break
            if found:break

        if not found: break
    
    return params, grads


# gradient clipping for exploding gradient problem
#   - checked at every min-batch
#   - normalize gradients when L2 norm exceeds the threshold
def clip_grads(grads, max_norm):
    norm = 0
    for grad in grads:
        norm += np.sum(grad ** 2)
    norm = np.sqrt(norm)

    rate = max_norm / (norm + 1e-7)
    if rate < 1:
        for grad in grads:
            grad *= rate