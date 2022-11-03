import os
import mnist
import pickle
import numpy as np
import scipy
import random
import matplotlib
matplotlib.use('agg')
import datetime
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf) #
# softmax cho code bằng thư viện numpy
from scipy.special import softmax
#lấy địa chỉ file data chứa tập dữ liệu mnist
mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # đưa label về dạng one-hot
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # đưa label về dạng one-hot

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset
##code function softmax
# def softmax(Z):
#     Z = np.array(Z)
#     e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
#     A = e_Z / e_Z.sum(axis = 0) # lấy max theo column
#     return A

def multinomial_logreg_loss_i(x, y, gamma, W):
    # x(785,1) y(10,1), W(10,785)
    x = x.reshape(-1, 1)
    yHat = softmax(np.dot(W, x))
    yHat = np.log(yHat)
    ans = -1 * np.dot(y.T, yHat) # mất mát trên một điểm dữ liệu
    ans += (gamma / 2) * (np.linalg.norm(W, 'fro')) ** 2
    ans = ans.item()
    return ans

def multinomial_logreg_grad_i(x, y, gamma, W):
    # x(785,1) y(10,1), W(10,785)
    yHat = softmax(np.matmul(W, x)) - y
    ans = np.matmul(yHat, x.T) + gamma * W
    return ans


# kiểm tra hàm multinomial_logreg_grad_i có phải là đạo hàm của hàm  multinomial_logreg_loss_i hay không
def test_gradient(Xs, Ys, gamma, W, alpha):
    # Xs(785,60000) Ys(10,60000), W(10,785)
    num_examples = Xs.shape[1]
    count = 0
    for i in range(num_examples):
        X_i, Y_i = Xs[:, i], Ys[:, i]
        V = np.random.rand(W.shape[0], W.shape[1])
        # Tính theo định nghĩa
        f_1 = multinomial_logreg_loss_i(X_i, Y_i, gamma, W + alpha * V)
        f_2 = multinomial_logreg_loss_i(X_i, Y_i, gamma, W)
        RHS = (f_1 - f_2) / alpha
        V = V.reshape(-1, 1)
        grad = multinomial_logreg_grad_i(X_i, Y_i, gamma, W)
        grad = grad.reshape(-1, 1)
        LHS = np.dot(V.T, grad).item()
        count += abs(LHS - RHS)
    return count / num_examples

#Tính lỗi bộ phân lớp cho bởi tham số W
def multinomial_logreg_error(Xs, Ys, W):
    #Xs(785,60000), Ys(10, 60000), W(10, 785)
    Ys = Ys.T
    yHat = softmax(np.dot(W, Xs), axis=0).T
    count = 0
    for i in range(len(Ys)):
        pred = np.argmax(yHat[i])
        if (Ys[i, pred] != 1):
            count += 1
    return count / len(Ys)


def multinomial_logreg_total_grad(Xs, Ys, gamma, W, st = False):
    # Xs(785,60000), Ys(10, 60000), W(10, 785)
    (d, n) = Xs.shape #d = 785, n = 60000
    ret = 0
    if st == True:
        ret = W * 0.0
        for i in range(n):
            ret += multinomial_logreg_grad_i(Xs[:, i], Ys[:, i], gamma, W)
    else:
        y_hat = softmax(np.dot(W, Xs), axis=0)
        del_L = np.dot(y_hat - Ys, Xs.T)
        ret = del_L + gamma * W
    return ret / n


def multinomial_logreg_total_loss(Xs, Ys, gamma, W, st=False):
    # Xs(785,60000), Ys(10, 60000), W(10, 785)
    (d, n) = Xs.shape #d = 785, n = 60000
    ret = 0
    if st == True:
        for i in range(n):
            ret += multinomial_logreg_loss_i(Xs[:, i], Ys[:, i], gamma, W)
    else:
        y_hat = softmax(np.dot(W, Xs), axis=0)
        log_y_hat = -1 * np.log(y_hat)
        y_dot_y_hat = np.multiply(log_y_hat, Ys)
        L_y_y_hat = np.sum(y_dot_y_hat)  # môt số float
        ret = L_y_y_hat + (gamma / 2) * (np.linalg.norm(W, 'fro')) ** 2
    return ret / n


def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq, st=False):
    # Xs(785,60000), Ys(10, 60000), W(10, 785)
    # monitor_freq tần suất xuất ra vector W
    params = []
    loss = []
    error = [] # lỗi tính theo %
    for i in range(num_iters):
        if (i % monitor_freq == 0):
            params.append(W0)
        W0 = W0 - alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W0, st) # update theta(W)
    params.append(W0)
    error.append(multinomial_logreg_error(Xs, Ys, W0))
    loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0, st))
    return error

def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    X_sub, Y_sub = Xs.T, Ys.T
    n = Xs.shape[1]
    perm = [np.random.randint(0, n) for _ in range(nsamples)]
    X_sub = Xs.T[perm]
    Y_sub = Ys.T[perm]
    estimated_error = multinomial_logreg_error(X_sub.T, Y_sub.T, W)
    return estimated_error


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # Chuyển về dang ma trân
    Xs_tr, Xs_te, Ys_tr, Ys_te = np.matrix(Xs_tr), np.matrix(Xs_te), np.matrix(Ys_tr), np.matrix(Ys_te)
    Xs_tr = Xs_tr.transpose()
    Xs_te = Xs_te.transpose()
    print("Kích thước ban đầu X train:", Xs_tr.shape)
    print("Kích thước ban đầu Y train:", Ys_tr.shape)
    Xs_tr = np.vstack((np.ones(60000), Xs_tr))
    Xs_te = np.vstack((np.ones(10000), Xs_te))
    print("Kích thước sau khi thêm hàng x0 = 1 vào đầu  X train:", Xs_tr.shape)
    print("Shape of initial Y train:", Ys_tr.shape)
    np.seterr(divide='ignore')
    # Part 1
    print("Part1\n")
    gamma = 0.0001
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    alpha = 10**-5
    loss_1 = multinomial_logreg_loss_i(Xs_tr[:,1], Ys_tr[:,1], gamma, W)
    print("Mất mát tại (1,1) = ",loss_1)
    grad_1 = multinomial_logreg_grad_i(Xs_tr[:,1], Ys_tr[:,1], gamma, W)
    print("Đạo hàm hàm mất mát tại (1,1) =", grad_1)
    derivative_loss_1 = test_gradient(Xs_tr[:,1], Ys_tr[:,1], gamma, W, alpha)
    #Tính độ chênh lệch đạo hàm (tính theo loss_i so với đạo hàm theo định nghĩa)
    #Tại 1 điểm dữ liệu
    print("Chệch lệch đào hàm hàm mất mát",derivative_loss_1)
    #Trung bình trên toàn tập dữ liệu
    alpha = [10**-1, 10**-3, 10**-5]
    for a in alpha:
        ret = test_gradient(Xs_tr, Ys_tr, gamma, W, a)
        print("Với alpha = ", a, "đô chênh lệch trung bình = ", ret)
    # Part 2
    print("Part2")
    gamma = 0.0001
    alpha = 1.0
    numberIter = 10
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])

    print("Với alpha=", alpha, "gamma=", gamma, "iterations=", numberIter, "monitorFreq=", monitorFreq)
    start = datetime.datetime.now()
    Ws_starter = np.array(gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq, True))
    print(Ws_starter)
    end = datetime.datetime.now()
    print(f"Time taken for the above config is:  {end-start}")

    #
    Part 3
    print("Part3")
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    end = datetime.datetime.now()
    gamma = 0.001
    alpha = 0.1
    numberIter = 10
    monitorFreq = 10
    print("Với alpha=", alpha, "gamma=", gamma, "iterations=", numberIter, "monitorFreq=", monitorFreq)
    start = datetime.datetime.now()
    Ws_numpy = np.array(gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq))
    print(Ws_numpy)
    end = datetime.datetime.now()
    print("Thời gian update W mới:",  end - start)

    # Part 4
    print("Part4")
    numberIter = 1000
    alpha = 0.1
    gamma = 0.0001
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    print("Với alpha=", alpha, "gamma=", gamma, "iterations=", numberIter, "monitorFreq=", monitorFreq)
    start = datetime.datetime.now()
    Ws_numpy = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    end = datetime.datetime.now()
    print("Thời gian update W mới:",  end - start)
    # Sử dụng tập con (test)
    estimate_error_ = estimate_multinomial_logreg_error(Xs_te, Ys_te, Ws_numpy)
    print(estimate_error_)
    est_err_tr_100, est_err_tr_1000, error, loss, est_err_te_1000, est_err_te_100, loss_np_te, error_np_te = [], [], [], [], [], [], [], []
    # Đo thời gian chạy của estimate_multinomial_logreg_error đánh giá độ lỗi
    start = datetime.datetime.now()
    _ = multinomial_logreg_error(Xs_tr, Ys_tr, Ws_numpy[-1])
    end = datetime.datetime.now()
    print("Thời gian để nhận được lỗi training với toàn bộ dữ liệu trên mô hình =:", end - start)

    start = datetime.datetime.now()
    _ = estimate_multinomial_logreg_error(Xs_tr, Ys_tr, Ws_numpy[-1], 100)
    end = datetime.datetime.now()
    print("Thời gian để nhận được lỗi training với 100 mẫu  trên mô hình =:", end - start)

    start = datetime.datetime.now()
    _ = estimate_multinomial_logreg_error(Xs_tr, Ys_tr, Ws_numpy[-1], 1000)
    end = datetime.datetime.now()
    print("Thời gian để nhận được lỗi training với 100 mẫu  trên mô hình =:", end - start)

    for w in Ws_numpy:
        loss.append(multinomial_logreg_total_loss(Xs_tr, Ys_tr, gamma, w))
        loss_np_te += [multinomial_logreg_total_loss(Xs_te, Ys_te, gamma, w)]

        error.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_np_te += [multinomial_logreg_error(Xs_te, Ys_te, w)]

        num_ex = 100
        est_err_tr_100.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, w, num_ex))
        est_err_te_100.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, w, num_ex))

        num_ex = 1000
        est_err_tr_1000.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, w, num_ex))
        est_err_te_1000.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, w, num_ex))

    plt.plot(range(numberIter // monitorFreq + 1), loss_np_te)
    plt.savefig("data/img/entire_ds_loss_test_" + str(numberIter) + ".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), error_np_te)
    plt.savefig("data/img/entire_ds_error_test_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), loss)
    plt.savefig("data/img/entire_ds_loss_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), error)
    plt.savefig("data/img/entire_ds_error_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_tr_100)
    plt.savefig("data/img/subsample_100_estimated_err_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_tr_1000)
    plt.savefig("data/img/subsample_1000_estimated_err_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_te_100)
    plt.savefig("data/img/subsample_100_estimated_err_test_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_te_1000)
    plt.savefig("data/img/subsample_1000_estimated_err_test_"+str(numberIter)+".png")
    plt.close()

    print("End.\n")




