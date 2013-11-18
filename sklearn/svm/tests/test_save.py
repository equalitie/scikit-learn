# toy sample
X = [[-2, -1, 0], [-1, -1, 0], [-1, -2, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0]]
Y = [1, 1, 1, 2, 2, 2]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [1, 2, 2]

from sklearn import svm, linear_model, datasets, metrics, base

def test_libsvm_save_libsvm_model():
    """
    Test if it is possible to save the model in libsvm format
    """
    clf = svm.SVC(kernel='linear').fit(X, Y)
    clf.save_libsvm_model('test_model.txt')

if __name__ == "__main__":
    test_libsvm_save_libsvm_model()
