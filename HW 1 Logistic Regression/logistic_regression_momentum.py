import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



from preprocess import data_preprocess, feature_extraction_bow, normalization

def sigmoid(z):
    """
    TODO: Implement sigmoid function. s is a vector with the size as z.
    """
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_w_and_b(dimension):
    w = np.zeros((dimension, 1))
    b = 0
    return w, b

def forward_prop(X, w, b):
    """
    Do the forward computation and return f(x)= \sigma (Wx+b)
    """
    f_x =sigmoid(np.dot(w.T, X) + b)
    return f_x

def compute_loss(A, Y, m):
    """
    TODO:
        Compute the loss function based on the formula you derived.
        loss is a scalar
        Hint:
            1) The formula should be (-1.0 / m) * np.sum(...)
    """
    loss = (-1.0 / m) * (np.dot(Y, (np.log(A)).T) + np.dot((1-Y),(np.log(1-A)).T)) 
    loss =  loss[0][0]                                               
    return loss

def back_prop(X, A, Y, m):
    """
    TODO:
        Compute the gradient based on the formula you derived.
        dw and db are two scalars.
        Hint:
            1) The formula of dw should be (1.0 / m) * np.dot(...)
            2) The formula of db should be (1.0 / m) * np.sum(...)
    """
    dw = (1.0 / m) * np.dot(X,(A-Y).T)
    db = (1.0 / m) * np.sum(A-Y)
    Vdw = np.zeros(dw.shape)
    Vdb = np.zeros(db.shape)
    beta = 0.9
    Vdw = beta * Vdw + (1-beta) * dw
    Vdb = beta * Vdb + (1-beta) * db

    return {"dw": Vdw, "db": Vdb}

def optimize(w, b, X, Y, X_dev, Y_dev, num_iterations, learning_rate, output_name):
    m = X.shape[1] # m is the number of the samples
    max_acc = 0
    max_w, max_b = w, b
    start_time = time.time()
    log = open(output_name + '.log', 'w')
    log.write('iteration, train acc, dev acc\n')

    for i in range(num_iterations):
        f_x = forward_prop(X, w, b)
        cost = compute_loss(f_x, Y, m)
        grads = back_prop(X, f_x, Y, m)

        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        
        Y_prediction_train = predict(w, b, X)
        Y_prediction_dev = predict(w, b, X_dev)
        train_acc = compare(Y_prediction_train, Y)
        dev_acc = compare(Y_prediction_dev, Y_dev)
        log.write('{},{},{}\n'.format(str(i + 1), str(train_acc), str(dev_acc)))

        if dev_acc > max_acc: # keep the best parameters
            max_acc = dev_acc
            max_w, max_b = w, b

        print(f"iteration: {i + 1}, time {time.time() - start_time:.2f}")
        print("\tTraining accuracy: {0:.4f} %, cost: {1:.4f}".format(train_acc, cost))
        print("\tDev accuracy: {0:.4f} %".format(dev_acc))

    params = {"w": max_w, "b": max_b}
    return params

def predict(w, b, X):
    """
    TODO:
        Predict the sentiment class based on the f(x) value.
        if f(x) > 0.5, then pred value is 1, otherwise is 0.
        Y_prediction is a 2-D array with the size (1*nb_sentence)
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    f_x = forward_prop(X, w, b)
    Y_prediction[f_x>0.5] = 1
    Y_prediction[f_x<=0.5] = 0
    return Y_prediction


def compare(pred, gold):
    """
    TODO:
        Compute the accuracy based on two array, pred and gold, and return a scalar between [0, 100]
    """
    acc = (pred==gold).sum()*100/gold.shape[1]
    return acc

def write_testset_prediction(parameters, test_data, file_name="myPrediction.csv"):
    Y_prediction_test = predict(parameters['w'], parameters['b'], test_data)
    f_pred = open(file_name, 'w')
    f_pred.write('ID\tSentiment')
    ID = 1
    for pred in Y_prediction_test[0]:
        sentiment_pred = 'pos' if pred == 1 else 'neg'
        f_pred.write(str(ID) + ',' + sentiment_pred + '\n')
        ID += 1

def model(X_train, Y_train, X_dev, Y_dev, output_name, num_iterations=100, learning_rate=0.005):
    w, b = initialize_w_and_b(X_train.shape[0])

    parameters = optimize(w, b, X_train, Y_train, X_dev, Y_dev, num_iterations, learning_rate, output_name)

    Y_prediction_dev = predict(parameters["w"], parameters["b"], X_dev)
    print("Best dev accuracy: {} %".format(compare(Y_prediction_dev, Y_dev)))

    np.save(output_name + '.npy', parameters)

    return parameters



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    parser.add_argument('-lr', '--learning_rate', required=False, default=0.1)
    parser.add_argument('-i', '--num_iter', required=False, default=200)
    parser.add_argument('-fn', '--file_name', help='file name', required=False, default='myTest')
    args = vars(parser.parse_args())
    print(args)

    print('[Read the data from twitter-sentiment.csv...]')
    revs, word2idx = data_preprocess('./twitter-sentiment.csv', args['clean'], int(args['max_vocab']))

    print('[Extract features from the read data...]')
    data, label = feature_extraction_bow(revs, word2idx)
    data = normalization(data)

    # shuffle data
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    data = data[shuffle_idx]
    label = label[shuffle_idx]
    print('[Start training...]')
    
    X_train, X_dev, Y_train, Y_dev = train_test_split(data, label, test_size=0.2, random_state=0)
    
    parameters = model(X_train.T, Y_train.T, X_dev.T, Y_dev.T, args['file_name'], num_iterations=int(args['num_iter']), learning_rate=float(args['learning_rate']))

    print('\n[Start evaluating on the official test set and dump as {}...]'.format(args['file_name'] + '.csv'))
    revs, _ = data_preprocess("./twitter-sentiment-testset.csv", args['clean'], int(args['max_vocab']))
    test_data, _ = feature_extraction_bow(revs, word2idx)
    write_testset_prediction(parameters, test_data.T, args['file_name'] + '.csv')
'''
    logmodel = LogisticRegression().fit(X_train, Y_train)
    logpredictions = logmodel.predict(X_dev)
    print (classification_report(Y_dev, logpredictions))   
    '''