import numpy as np
import warnings

def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    N = np.shape(p_y_x)[0]
    M = np.shape(p_y_x)[1]
    # fliping array so that last biggest argument is taken
    p_y_x = np.flip(p_y_x, axis=1)
    # M-1 - argmax, because we fliped array line abowe and we wat to have proper argumnets
    y = (M-1)*np.ones(shape=(1, N)) - np.argmax(p_y_x, axis=1)
    return np.count_nonzero(y_true - y)/N


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    result = np.empty((10,))
    for m in range(10):
        result[m] = np.sum(y_train == m) / np.shape(y_train)[0]

    return result


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    result = np.empty((10, np.shape(X_train)[1]))
    for m in range(10):
        mask = y_train == m
        denominator = np.sum(mask) + a + b - 2

        filtered_class_entries = X_train[mask, :]
        numerator = np.sum(filtered_class_entries, axis=0) + a - 1
        result[m] = numerator / denominator

    return result


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """

    result = np.empty((np.shape(X)[0], 10))
    for m in range(10):
        p_x_y_each_word = p_x_1_y[m] ** X * (1 - p_x_1_y[m]) ** (1 - X)
        p_x_y = np.prod(p_x_y_each_word, axis=1)
        result[:, m] = p_x_y * p_y[m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = result / np.sum(result, axis=1)[:, None]
    return result


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors, best_p, best_p_x_1_y), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]), "best_p_x_1_y" najlepszy rozkład prawdopodobieństw p(x=1|y) MxD.
    """

    best_error = float('inf')
    best_a = 0
    best_b = 0
    errors = []
    best_p_x_1_y = None

    p_y = estimate_a_priori_nb(y_train)
    for a in a_values:
        errors_inner = []
        for b in b_values:
            p_x_1_y = estimate_p_x_y_nb(X_train, y_train, a, b)
            p_y_x = p_y_x_nb(p_y, p_x_1_y, X_val)
            error = classification_error(p_y_x, y_val)
            if error < best_error:
                best_error = error
                best_a = a
                best_b = b
                best_p_x_1_y = p_x_1_y
            errors_inner.append(error)
        errors.append(errors_inner)
    return (best_error, best_a, best_b, errors, best_p_x_1_y)
