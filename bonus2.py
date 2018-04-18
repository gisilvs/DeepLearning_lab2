import numpy as np
import matplotlib.pyplot as plt
import pickle

def flip_img(z):
    '''the image is first transformed from vector to the image format, flipped, and then vectorized again'''
    z = z.reshape(3, 32, 32).transpose(1, 2, 0)
    z=np.fliplr(z)
    z=z.transpose(2,0,1)
    z=z.reshape(-1)
    return z

def check_grad(grad_a, grad_n, eps):
    '''function to compare the analitical (grad_a) and numerical (grad_n) gradients'''
    diff = np.abs(grad_a - grad_n) / max(eps, np.amax(np.abs(grad_a) + np.abs(grad_n)))
    if np.amax(diff) < 1e-3:
        return True
    else:
        return False

def compute_accuracy(X,y,W1,b1,W2,b2):
    '''percentage of correct answers'''
    P,h,s1=evaluate_classifier(X,W1,b1,W2,b2)
    k=np.argmax(P,axis=0)
    return (k == y).sum()*100/np.shape(X)[1]

def compute_accuracy_Leaky(X,y,W1,b1,W2,b2):
    '''percentage of correct answers'''
    P,h,s1=evaluate_classifier_Leaky(X,W1,b1,W2,b2)
    k=np.argmax(P,axis=0)
    return (k == y).sum()*100/np.shape(X)[1]

def compute_gradients(X,Y,W1,b1,W2,b2,lambd,dropout=False,drop_prob=0.5):
    '''compute the analytical gradients'''
    n=np.shape(X)[1]
    P,h,s1=evaluate_classifier(X,W1,b1,W2,b2,dropout,drop_prob)
    grad_b2=np.zeros(np.shape(b2))#(K,1)
    grad_W2=np.zeros((np.shape(W2)))#(K,m)
    grad_b1 = np.zeros(np.shape(b1))# (m,1)
    grad_W1 = np.zeros((np.shape(W1)) ) # (m,d)
    G=-(Y-P)
    grad_b2=np.sum(G,axis=1).reshape(-1,1)
    grad_W2=np.dot(G,h.T)
    G=np.dot(W2.T,G)
    G=G*np.where(s1>0,1,0)
    grad_b1=np.sum(G,axis=1).reshape(-1,1)
    grad_W1=np.dot(G,X.T)
    grad_b1/=(n)
    grad_W1/=(n)
    grad_b2 /= (n)
    grad_W2 /= (n)
    return grad_W1+2*lambd*W1,grad_b1,grad_W2+2*lambd*W2,grad_b2

def compute_gradients_Leaky(X,Y,W1,b1,W2,b2,lambd):
    '''compute the analytical gradients'''
    n=np.shape(X)[1]
    P,h,s1=evaluate_classifier_Leaky(X,W1,b1,W2,b2)
    grad_b2=np.zeros(np.shape(b2))#(K,1)
    grad_W2=np.zeros((np.shape(W2)))#(K,m)
    grad_b1 = np.zeros(np.shape(b1))# (m,1)
    grad_W1 = np.zeros((np.shape(W1)) ) # (m,d)
    G=-(Y-P)
    grad_b2=np.sum(G,axis=1).reshape(-1,1)
    grad_W2=np.dot(G,h.T)
    G=np.dot(W2.T,G)
    G=G*np.where(s1>0,1,0.1)
    grad_b1=np.sum(G,axis=1).reshape(-1,1)
    grad_W1=np.dot(G,X.T)
    grad_b1/=(n)
    grad_W1/=(n)
    grad_b2 /= (n)
    grad_W2 /= (n)
    return grad_W1+2*lambd*W1,grad_b1,grad_W2+2*lambd*W2,grad_b2

def compute_grad_num_slow(X, Y, W1, b1,W2,b2, lambd, h):
    '''centered difference gradient for both W and b'''
    grad_W1 = np.zeros((np.shape(W1)))
    grad_b1 = np.zeros((np.shape(b1)))
    grad_W2 = np.zeros((np.shape(W2)))
    grad_b2 = np.zeros((np.shape(b2)))

    # iterate over all indexes in b
    it = np.nditer(b1, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ib = it.multi_index
        old_value = b1[ib]
        b1[ib] = old_value - h  # use original value
        c1 = compute_cost(X, Y, W1, b1, W2,b2,lambd)
        b1[ib] = old_value + h  # use original value
        c2 = compute_cost(X, Y, W1, b1, W2,b2,lambd)
        grad_b1[ib] = (c2 - c1) / (2 * h)
        b1[ib] = old_value  # restore original value
        it.iternext()

    it = np.nditer(b2, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ib = it.multi_index
        old_value = b2[ib]
        b2[ib] = old_value - h  # use original value
        c1 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        b2[ib] = old_value + h  # use original value
        c2 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        grad_b2[ib] = (c2 - c1) / (2 * h)
        b2[ib] = old_value  # restore original value
        it.iternext()

    # iterate over all indexes in W
    it = np.nditer(W1, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = W1[iW]
        W1[iW] = old_value - h
        c1 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        W1[iW] = old_value + h
        c2 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        grad_W1[iW] = (c2 - c1) / (2 * h)
        W1[iW] = old_value
        it.iternext()

    it = np.nditer(W2, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = W2[iW]
        W2[iW] = old_value - h
        c1 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        W2[iW] = old_value + h
        c2 = compute_cost(X, Y, W1, b1, W2, b2, lambd)
        grad_W2[iW] = (c2 - c1) / (2 * h)
        W2[iW] = old_value
        it.iternext()

    return grad_W1, grad_b1, grad_W2, grad_b2

def compute_grad_num_slow_Leaky(X, Y, W1, b1,W2,b2, lambd, h):
    '''centered difference gradient for both W and b'''
    grad_W1 = np.zeros((np.shape(W1)))
    grad_b1 = np.zeros((np.shape(b1)))
    grad_W2 = np.zeros((np.shape(W2)))
    grad_b2 = np.zeros((np.shape(b2)))

    # iterate over all indexes in b
    it = np.nditer(b1, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ib = it.multi_index
        old_value = b1[ib]
        b1[ib] = old_value - h  # use original value
        c1 = compute_cost_Leaky(X, Y, W1, b1, W2,b2,lambd)
        b1[ib] = old_value + h  # use original value
        c2 = compute_cost_Leaky(X, Y, W1, b1, W2,b2,lambd)
        grad_b1[ib] = (c2 - c1) / (2 * h)
        b1[ib] = old_value  # restore original value
        it.iternext()

    it = np.nditer(b2, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ib = it.multi_index
        old_value = b2[ib]
        b2[ib] = old_value - h  # use original value
        c1 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        b2[ib] = old_value + h  # use original value
        c2 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        grad_b2[ib] = (c2 - c1) / (2 * h)
        b2[ib] = old_value  # restore original value
        it.iternext()

    # iterate over all indexes in W
    it = np.nditer(W1, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = W1[iW]
        W1[iW] = old_value - h
        c1 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        W1[iW] = old_value + h
        c2 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        grad_W1[iW] = (c2 - c1) / (2 * h)
        W1[iW] = old_value
        it.iternext()

    it = np.nditer(W2, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iW = it.multi_index
        old_value = W2[iW]
        W2[iW] = old_value - h
        c1 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        W2[iW] = old_value + h
        c2 = compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd)
        grad_W2[iW] = (c2 - c1) / (2 * h)
        W2[iW] = old_value
        it.iternext()

    return grad_W1, grad_b1, grad_W2, grad_b2


def compute_cost(X, Y, W1, b1,W2,b2, lambd):
    '''cross-entropy loss + regularization'''
    N = np.shape(X)[1]
    P,h,s1= evaluate_classifier(X, W1, b1,W2,b2)
    # The loss function is computed by element-wise product between Y and P, and summing column by column
    return np.sum(-np.log(np.sum(Y * P, axis=0))) / N + lambd * (np.sum(W1 ** 2)+np.sum(W2**2))

def compute_cost_Leaky(X, Y, W1, b1,W2,b2, lambd):
    '''cross-entropy loss + regularization'''
    N = np.shape(X)[1]
    P,h,s1= evaluate_classifier_Leaky(X, W1, b1,W2,b2)
    # The loss function is computed by element-wise product between Y and P, and summing column by column
    return np.sum(-np.log(np.sum(Y * P, axis=0))) / N + lambd * (np.sum(W1 ** 2)+np.sum(W2**2))

def evaluate_classifier(X,W1,b1,W2,b2,Dropout=False,drop_prob=0.5):
    '''classification function Softmax(WX+b)'''
    s1=np.dot(W1,X)+b1
    h=np.maximum(0,s1)
    if Dropout:
        u1 = np.random.binomial(1, drop_prob, size=h.shape) / drop_prob
        h*=u1
    s=np.dot(W2,h)+b2
    return softmax(s),h,s1

def evaluate_classifier_Leaky(X,W1,b1,W2,b2):
    '''classification function Softmax(WX+b)'''
    s1=np.dot(W1,X)+b1
    h=np.maximum(0.1*s1,s1)
    s=np.dot(W2,h)+b2
    return softmax(s),h,s1

def initialize_parameters(d,K,m,seed=None):
    '''randomly initialize the matrices W1,W2 and the vectors b1 and b2'''
    '''
        d--> dimension of one image
        K--> number of classes
        m--> number of nodes in the network
    '''
    if seed:
        np.random.seed(seed)
    W1=np.random.normal(0,0.001,(m,d))
    W2=np.random.normal(0,0.001,(K,m))
    b1 = np.zeros(m).reshape(-1,1)
    b2 = np.zeros(K).reshape(-1,1)

    return W1,W2,b1,b2

def initialize_parameters_HE(d,K,m,seed=None):
    '''randomly initialize the matrices W1,W2 and the vectors b1 and b2'''
    '''
        d--> dimension of one image
        K--> number of classes
        m--> number of nodes in the network
    '''
    if seed:
        np.random.seed(seed)
    W1=np.random.normal(0,np.sqrt(2/d),(m,d))
    W2=np.random.normal(0,np.sqrt(2/m),(K,m))
    b1 = np.zeros(m).reshape(-1,1)
    b2 = np.zeros(K).reshape(-1,1)

    return W1,W2,b1,b2



def labels_to_onehot(labels):
    '''Take as input the vector of labels and return the one_hot matrix of dimensions (K,N)'''
    '''
        K=number of classes
        N=number of labeled inputs
    '''
    len=np.max(labels)+1
    Y=np.zeros((len,np.size(labels)))
    for i in range(np.size(labels)):
        Y[labels[i],i]=1
    return Y
def load_all():
    '''hardcoded function that loads the five cifair set and add the last 10000 values to the validation set'''
    X_train=np.asarray([])
    Y_train=np.asarray([])
    y_train=np.asarray([])

    X_1,Y_1,y_1=load_batch('cifar-10-batches-py/data_batch_1')
    X_2, Y_2, y_2 = load_batch('cifar-10-batches-py/data_batch_2')
    X_3, Y_3, y_3 = load_batch('cifar-10-batches-py/data_batch_3')
    X_4, Y_4, y_4 = load_batch('cifar-10-batches-py/data_batch_4')
    X_5, Y_5, y_5 = load_batch('cifar-10-batches-py/data_batch_5')


    X=np.hstack((X_1,X_2,X_3,X_4,X_5))
    Y = np.hstack((Y_1, Y_2, Y_3, Y_4, Y_5))
    y = np.hstack((y_1, y_2, y_3, y_4, y_5))

    X_train=X[:,:-1000]
    Y_train=Y[:,:-1000]
    y_train=y[:-1000]

    X_valid = X[:,-1000:]
    Y_valid=Y[:,-1000:]
    y_valid=y[-1000:]

    return X_train,Y_train,y_train, X_valid, Y_valid, y_valid


def load_batch(file_name):
    '''load one batch from cfair-10'''
    '''
        return:
        data --> the images converted to float and divided by 255
        onehot --> the onehot matrix of the labels
        labels --> labels of the images
    '''
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data=np.asarray(dict[b'data'],dtype=float)/255
    labels=np.asarray(dict[b'labels'])
    onehot=labels_to_onehot(labels)
    return data.T,onehot,labels

def mini_batch_gd(X,Y,y,X_valid,Y_valid,y_valid,n_batch,eta,n_epochs,W1,b1,W2,b2,lambd,rho,E,rand_jitt,
                  momentum=True,weight_decay=True, Leaky=True, early_stopping=True, random_jitter=True,
                  dropout=False,drop_prob=0.5):
    '''compute the mini batch gd
        return the final weights W and b
    '''
    if momentum:
        mom_W1=np.zeros(W1.shape)
        mom_b1=np.zeros(b1.shape)
        mom_W2 = np.zeros(W2.shape)
        mom_b2 = np.zeros(b2.shape)
    accuracy=[]
    cost=[]
    valid_accuracy=[]
    valid_cost=[]
    count = 0  # For early stopping
    best_accuracy = 0  # For early stopping
    for epoch in range(n_epochs):
        print(epoch)
        ##use for shuffle of the batch order#
        #X,Y,y=shuffle_X_Y(X,Y,y)
        for j in range(np.shape(X)[1]//n_batch): #loop trough all the mini batches of the dataset
            j_start=j*n_batch
            j_end=(j+1)*n_batch
            X_batch=X[:,j_start:j_end]
            Y_batch=Y[:,j_start:j_end]
            if random_jitter:
                for i in range(X_batch.shape[1]):
                    if np.random.random()>rand_jitt:
                        X_batch[:,i]=flip_img(X_batch[:,i])
            if Leaky:
                grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients_Leaky(X_batch, Y_batch, W1, b1, W2, b2, lambd)
            else:
                grad_W1,grad_b1, grad_W2,grad_b2=compute_gradients(X_batch,Y_batch,W1,b1,W2,b2,lambd)
            if momentum:
                mom_W1=rho*mom_W1+eta*grad_W1
                mom_b1=rho*mom_b1+eta*grad_b1
                mom_W2 = rho * mom_W2 + eta * grad_W2
                mom_b2 = rho * mom_b2 + eta * grad_b2
                W1-=mom_W1
                b1-=mom_b1
                W2-=mom_W2
                b2 -= mom_b2
            else:
                W1-=eta*grad_W1
                b1-=eta*grad_b1
                W2 -= eta * grad_W2
                b2 -= eta * grad_b2

        if Leaky:
            accuracy.append(compute_accuracy_Leaky(X, y, W1, b1, W2, b2))  # accuracy for training set
            cost.append(compute_cost_Leaky(X, Y, W1, b1, W2, b2, lambd))  # cost for training set

            # The function on validation use global variables!!!
            a = compute_accuracy_Leaky(X_valid, y_valid, W1, b1, W2, b2)
            valid_accuracy.append(a) # accuracy for validation set
            valid_cost.append(compute_cost_Leaky(X_valid, Y_valid, W1, b1, W2, b2, lambd))  # cost for validation set
        else:
            accuracy.append(compute_accuracy(X,y,W1,b1,W2,b2))#accuracy for training set
            cost.append(compute_cost(X,Y,W1,b1,W2,b2,lambd))#cost for training set

            #The function on validation use global variables!!!
            a = compute_accuracy(X_valid, y_valid, W1, b1, W2, b2)
            valid_accuracy.append(a)
            valid_cost.append(compute_cost(X_valid,Y_valid,W1,b1,W2,b2,lambd))#cost for validation set

        if early_stopping:
            '''Early stopping'''
            if a>best_accuracy:
                best_index=epoch
                count=0
                best_accuracy=a
                best_W1=W1
                best_b1=b1
                best_b2=b2
                best_W2=W2

            else:
                count+=1
            if count==E:
                break


        if weight_decay:
            eta=eta*0.92
    ##various plotting and printing##
    plt.plot(accuracy,c='g',label='train')
    plt.plot(valid_accuracy,c='r',label='validation')
    plt.xlabel('epochs')
    plt.ylabel('accuracy %')
    plt.legend()
    plt.show()
    plt.plot(cost,c='g',label='train')
    plt.plot(valid_cost, c='r', label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    ###print last obtained accuracy fortraining and validation###
    print(accuracy[-1])
    print(valid_accuracy[-1])
    ###print last obtained loss fortraining and validation###
    print(cost[-1])
    print(valid_cost[-1])
    return W1,b1,W2,b2

def softmax(S):
    return np.exp(S)/np.sum(np.exp(S),axis=0)

load=True #1: load one batch, 0: load whole data set
if not load:
    X,Y,y=load_batch('cifar-10-batches-py/data_batch_1')
    X_valid,Y_valid,y_valid=load_batch('cifar-10-batches-py/data_batch_2')
else:
    X,Y,y,X_valid,Y_valid,y_valid=load_all()
X_test,Y_test,y_test=load_batch('cifar-10-batches-py/test_batch')

HE=False
lambd=0.002620036321987215169
h=1e-5
n_batch=100
eta=0.0841490501602296056
n_epochs=300
rho=0.9
momentum=True
w_d=True
Leaky=False
early_stopping=True
n_early=30
rand_jitt=0.5
random_jitt=False
m=50 #number of nodes
dropout=True
drop_prob=0.5


X_mean=np.mean(X,axis=1).reshape(-1,1)
X-=X_mean
X_valid-=X_mean
X_test-=X_mean
d=np.shape(X)[0]#dimension of input image

K=np.shape(Y)[0]#number of classes
if not HE:
    W1,W2,b1,b2=initialize_parameters(d,K,m,100)
else:
    W1,W2,b1,b2=initialize_parameters_HE(d,K,m,100)


if not Leaky:
    W1, b1, W2, b2 = mini_batch_gd(X, Y, y, X_valid, Y_valid, y_valid, n_batch, eta, n_epochs, W1, b1, W2, b2, lambd,
                                   rho,n_early,rand_jitt, momentum=momentum, weight_decay=w_d,Leaky=Leaky,
                                   early_stopping=early_stopping,random_jitter=random_jitt,dropout=dropout,
                                   drop_prob=drop_prob)
    test_accuracy=compute_accuracy(X_test,y_test,W1,b1,W2,b2)
    print(test_accuracy)
else:
    W1, b1, W2, b2 = mini_batch_gd(X, Y, y, X_valid, Y_valid, y_valid, n_batch, eta, n_epochs, W1, b1, W2, b2, lambd,
                                   rho, n_early,rand_jitt, momentum=momentum, weight_decay=w_d, Leaky=Leaky,
                                   early_stopping=early_stopping,random_jitter=random_jitt,dropout=dropout,
                                   drop_prob=drop_prob)
    test_accuracy = compute_accuracy_Leaky(X_test, y_test, W1, b1, W2, b2)
    print(test_accuracy)