import cv2
import numpy as np


def read_image(path):
    return np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def write_image(image, path):
    cv2.imwrite(path, image)


def extract_characters(image):
    characters = []
    gray = image
    if(len(image.shape) > 2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('thresh', thresh)
    ctrs, hier = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        roi = image[y:y + h, x:x + w]
        roi = cv2.copyMakeBorder(
            roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        area = w * h

        if 250 < area < 900:
            r = roi
            resized = cv2.resize(r, (28, 28), interpolation=cv2.INTER_AREA)
            ret, resized = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
            resized = cv2.bitwise_not(resized)
            characters.append(resized)
            rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('x', image)
    cv2.waitKey(0)
    return characters


DATA_DIR = 'data/mnist/'
TEST_DIR = 'test/'
TEST_DATA_FILE = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABELS_FILE = DATA_DIR + 't10k-labels-idx1-ubyte'
TRAIN_DATA_FILE = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABELS_FILE = DATA_DIR + 'train-labels-idx1-ubyte'


def bytes_to_int(bytedata):
    return int.from_bytes(bytedata, 'big')


def read_file(filename, max_image=None):
    images = []
    with open(filename, 'rb') as f:
        m = f.read(4)
        nimage = bytes_to_int(f.read(4))
        if max_image:
            nimage = max_image
        rows = bytes_to_int(f.read(4))
        columns = bytes_to_int(f.read(4))
        for image_idx in range(nimage):
            image = []
            for rows_idx in range(rows):
                row = []
                for col_idx in range(columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, max_label=None):
    labels = []
    with open(filename, 'rb') as f:
        m = f.read(4)
        nlabel = bytes_to_int(f.read(4))
        if max_label:
            nlabel = max_label
        for label_idx in range(nlabel):
            label = f.read(1)
            labels.append(label)
    return labels


def flatten(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten(sample) for sample in X]


def dist(x, y):
    # print(x,y)
    return sum([
        (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
        for x_i, y_i in zip(x, y)
    ])**(0.5)


def get_most_frequent_element(l):
    return (max(l, key=l.count))


def get_training_dis_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def knn(X_train, y_train, X_test, y_test, k=3):
    y_prediction = []
    for test_sample_idx, sample in enumerate(X_test):
        training_distances = get_training_dis_for_test_sample(X_train, sample)
        sorted_distance_indices = [
            pair[0]
            for pair in (
                sorted(enumerate(training_distances), key=lambda x: x[1])
            )
        ]
        candidates = [bytes_to_int(y_train[idx])
                      for idx in sorted_distance_indices[:k]]
        # print(
        # f'Point is {bytes_to_int(y_test[test_sample_idx])} and we guessed {candidates}')
        top_candidate = get_most_frequent_element(candidates)
        y_prediction.append(top_candidate)
    return y_prediction


def main():
    X_train = read_file(TRAIN_DATA_FILE, 10000)
    y_train = read_labels(TRAIN_LABELS_FILE, 10000)
    # X_test = read_file(TEST_DATA_FILE, 5)
    y_test = read_labels(TEST_LABELS_FILE, 5)
    X_test = [read_image(f'{TEST_DIR}test.png')]
    characters = extract_characters(X_test[0])
    for idy, char in enumerate(characters):
        X_test = [char]
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}-{idy}.png')

            X_trainx = extract_features(X_train)
            X_testx = extract_features(X_test)

            y_pred = knn(X_trainx, y_train, X_testx, y_test, 3)

            print(y_pred)
            correct_predictions = sum([
                int(y_pred_i == bytes_to_int(y_test_i)) for y_pred_i, y_test_i in zip(y_pred, y_test)]) / len(y_test)
            # print(f'Accuracy: {correct_predictions}')


if __name__ == '__main__':
    main()
