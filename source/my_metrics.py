import numpy as np


def check_performance(mark: list, prediction: list, threshold=0.5):
    prediction = prediction > threshold
    prediction = prediction.astype(int)
    intersect_of_pixel = (mark * prediction).sum()
    union_of_pixel = mark.sum() + prediction.sum()

    return intersect_of_pixel / (union_of_pixel - intersect_of_pixel)


def performance_over_thresholds(mark, prediction):
    mark = mark.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    res = list()
    for threshold in range(50, 100, 5):
        threshold = threshold / 100

        res.append(check_performance(mark, prediction, threshold))

    return sum(res) / len(res)


if __name__ == "__main__":
    true = np.array([1, 0, 1, 0])
    pred = np.array([0.6, 0.3, 0.7, 0.4])
    res = check_performance(true, pred, threshold=0.5)

    print(res)

    print(performance_over_thresholds(true, pred))
