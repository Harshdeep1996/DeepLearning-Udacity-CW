import numpy as np
from pre_process import pre_process_data


class RNN:

    def __init__(self, word_dimension, hidden_dimension=100, bptt_truncate=4):
        """
        Constructor for the RNN

        :params word_dimension: number of vocab words length, total number of
            vocab words.
        :params hidden_dim: number of nodes in the hidden dimension.
        :param bptt_truncate: NOT YET KNOWN.
        """
        self.word_dimension = word_dimension
        self.hidden_dimension = hidden_dimension
        self.bptt_truncate = bptt_truncate

        # Initialize the weights between -1/sqrt(n) and 1/sqrt(n),
        # n is the number of incoming connections, which is equal to the
        # number of nodes in the previous layer.
        self.U = np.random.uniform(
            -np.sqrt(1./word_dimension),
            np.sqrt(1./word_dimension),
            (hidden_dimension, word_dimension)
        )
        self.W = np.random.uniform(
            -np.sqrt(1./hidden_dimension),
            np.sqrt(1./hidden_dimension),
            (hidden_dimension, hidden_dimension)
        )
        self.V = np.random.uniform(
            -np.sqrt(1./hidden_dimension),
            np.sqrt(1./hidden_dimension),
            (word_dimension, hidden_dimension)
        )

    def forward_propagation(self, x):
        # Number of time steps an RNN unfolds into depending upon vocab size
        time_steps = len(x)
        # We need `s` to save all hidden states and an initial extra element
        # for the starting
        s = np.zeros((time_steps + 1, self.hidden_dimension))
        s[-1] = np.zeros(self.hidden_dimension)

        # This is needed to store the output for each time step, and the word
        # probability associated with each word for each time step.
        o = np.zeros((time_steps, self.word_dimension))
        for t in np.arange(time_steps):
            # Follow the previous and current time step
            s[t] = np.tanh(self.U[:, x[t]], self.W.dot(s[t-1]))
            o[t] = self.V.dot(s[t])

        # need to call it here, otherwise it will just be a vector, no point!
        return s, softmax(o)

    def predict(self, x):
        s, o = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_cross_entropy(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        loss = 0
        print('This is how long the loop will run {}'.format(len(y)))
        for i in np.arange(len(y)):
            s, o = self.forward_propagation(x[i])

            # This is a crucial and tricky bit of numpy code.
            # Instead of having tuples indexing the matrice, it is set of list
            # with the matrices index marked in list X and Y.
            # For example to index, (0,0) (1,2) (0,1) in a 3 X 3 matirx
            # it will be X[[0,1,0], [0,2,1]]
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            loss += -1 * np.sum(np.log(correct_word_predictions))
            print('This example {} is done'.format(i))
        return loss / N

    def bptt(self, x, y):
        pass


def softmax(w):
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e.T / np.sum(e, axis=1)
    return dist.T


if __name__ == '__main__':
    np.random.seed(10)
    model = RNN(8000)
    X_train, Y_train = pre_process_data()
    print "Actual loss: %f" % model.calculate_cross_entropy(
        X_train[:1000], Y_train[:1000])
    # predictions = model.predict(X_train[10])
    # print predictions.shape
    # print predictions
    # s, o = model.forward_propagation(X_train[10])
    # print o.shape
    # print o
