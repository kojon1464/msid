import byes
import torchvision
import utils.mnist_reader as reader

EPOCHS = 100
MINIBATCH_SIZE = 50

if __name__ == "__main__":

    torchvision.datasets.FashionMNIST('', train=True, download=True)
    torchvision.datasets.FashionMNIST('', train=False, download=True)

    X_train, y_train = reader.load_mnist('FashionMNIST/raw', kind='train')
    X_test, y_test = reader.load_mnist('FashionMNIST/raw', kind='t10k')

    a_values = [1, 3, 10, 30, 100, 300, 1000]
    b_values = [1, 3, 10, 30, 100, 300, 1000]

    X_train = (X_train > 137).astype(int)
    X_test = (X_test > 137).astype(int)

    best_error, best_a, best_b, errors, best_p_x_1_y = byes.model_selection_nb(X_train[:5000], X_train[-5000:], y_train[:5000],
                                                            y_train[-5000:], a_values, b_values)

    p_y = byes.estimate_a_priori_nb(y_train[:5000])
    p_x_y = byes.p_y_x_nb(p_y, best_p_x_1_y, X_test)
    error = byes.classification_error(p_x_y, y_test)
    print(1 - error)
