import numpy as np
import matplotlib.pyplot as plt
import pickle

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

def compute_gradients(X,Y,W1,b1,W2,b2,lambd):
    '''compute the analytical gradients'''
    n=np.shape(X)[1]
    P,h,s1=evaluate_classifier(X,W1,b1,W2,b2)
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

def compute_cost(X, Y, W1, b1,W2,b2, lambd):
    '''cross-entropy loss + regularization'''
    N = np.shape(X)[1]
    P,h,s1= evaluate_classifier(X, W1, b1,W2,b2)
    # The loss function is computed by element-wise product between Y and P, and summing column by column
    return np.sum(-np.log(np.sum(Y * P, axis=0))) / N + lambd * (np.sum(W1 ** 2)+np.sum(W2**2))

def evaluate_classifier(X,W1,b1,W2,b2):
    '''classification function Softmax(WX+b)'''
    s1=np.dot(W1,X)+b1
    h=np.maximum(0,s1)
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

def mini_batch_gd(X,Y,y,X_valid,Y_valid,y_valid,n_batch,eta,n_epochs,W1,b1,W2,b2,lambd,rho,momentum=True,weight_decay=True):
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
    for epoch in range(n_epochs):
        print(epoch)
        ##use for shuffle of the batch order#
        #X,Y,y=shuffle_X_Y(X,Y,y)
        for j in range(np.shape(X)[1]//n_batch): #loop trough all the mini batches of the dataset
            j_start=j*n_batch
            j_end=(j+1)*n_batch
            X_batch=X[:,j_start:j_end]
            Y_batch=Y[:,j_start:j_end]
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
        accuracy.append(compute_accuracy(X,y,W1,b1,W2,b2))#accuracy for training set
        cost.append(compute_cost(X,Y,W1,b1,W2,b2,lambd))#cost for training set

        #The function on validation use global variables!!!
        valid_accuracy.append(compute_accuracy(X_valid,y_valid,W1,b1,W2,b2))#accuracy for validation set
        valid_cost.append(compute_cost(X_valid,Y_valid,W1,b1,W2,b2,lambd))#cost for validation set

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

load=0 #1: load one batch, 0: load whole data set
if load:
    X,Y,y=load_batch('cifar-10-batches-py/data_batch_1')
    X_valid,Y_valid,y_valid=load_batch('cifar-10-batches-py/data_batch_2')
else:
    X,Y,y,X_valid,Y_valid,y_valid=load_all()
X_test,Y_test,y_test=load_batch('cifar-10-batches-py/test_batch')
lambd=0.002620036321987215169
h=1e-5
n_batch=100
eta=0.0841490501602296056
n_epochs=30
rho=0.9
momentum=True
w_d=True

X_mean=np.mean(X,axis=1).reshape(-1,1)
X-=X_mean
X_valid-=X_mean
X_test-=X_mean
d=np.shape(X)[0]#dimension of input image
m=50 #number of nodes
K=np.shape(Y)[0]#number of classes
W1,W2,b1,b2=initialize_parameters(d,K,m,100)

'''UNCOMMENT TO TEST GRADIENTS'''
#grad_num_W1, grad_num_b1, grad_num_W2, grad_num_b2=compute_grad_num_slow(X[:,:2],Y[:,:2],W1,b1,W2,b2,lambd,h)#numerical gradients
#b1_good=check_grad(grad_b1,grad_num_b1,1e-7)
#W1_good=check_grad(grad_W1,grad_num_W1,1e-7)
#b2_good=check_grad(grad_b2,grad_num_b2,1e-7)
#W2_good=check_grad(grad_W2,grad_num_W2,1e-7)
#grad_W1,grad_b1,grad_W2,grad_b2=compute_gradients(X,Y,W1,b1,W2,b2,lambd)
#W1,b1,W2,b2=mini_batch_gd(X[:,:100],Y[:,:100],y[:100],X_valid[:,:100],Y_valid[:,:100],y_valid[:100],10,0.1,200,W1,b1,W2,b2,lambd,rho,False,False)


#W1,b1,W2,b2=mini_batch_gd(X,Y,y,X_valid,Y_valid,y_valid,n_batch,eta,n_epochs,W1,b1,W2,b2,lambd,rho,momentum,weight_decay=w_d)


'''THIS LAST PART WAS USED FOR THE SEARCH OF GOOD PARAMETHERS
emin=-2
emax=-1
lmin=-3
lmax=-2
n_samples=10

results=np.loadtxt('results33.txt')
a=0
#results=sorted(results,key=lambda x: x[2])
#np.savetxt('results33.txt',results)
results=[]
for n in range(n_samples):
    np.random.seed(None)
    e = emin + (emax - emin) * np.random.uniform(0, 1)
    l = lmin + (lmax - lmin) * np.random.uniform(0, 1)
    eta=10**e
    lambd=10**l
    W1, W2, b1, b2 = initialize_parameters(d, K, m, 100)
    W1, b1, W2, b2 = mini_batch_gd(X, Y, y, X_valid, Y_valid, y_valid, n_batch, eta, n_epochs, W1, b1, W2, b2, lambd,
                                   rho, momentum, weight_decay=w_d)
    J = compute_cost(X_valid, Y_valid, W1, b1, W2, b2, lambd)
    acc=compute_accuracy(X_valid,y_valid,W1,b1,W2,b2)
    results.append((eta,lambd,acc))
np.savetxt('results3.txt',results)'''
W1, b1, W2, b2 = mini_batch_gd(X, Y, y, X_valid, Y_valid, y_valid, n_batch, eta, n_epochs, W1, b1, W2, b2, lambd,
                                   rho, momentum, weight_decay=w_d)
test_accuracy=compute_accuracy(X_test,y_test,W1,b1,W2,b2)
print(test_accuracy)