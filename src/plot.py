import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_transcoder

def remove_labels(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def remove_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def example_results(test_set,encoder,decoder,n=10,shuffle=False):
    transc = get_transcoder(encoder,decoder)
    if shuffle is True:
        np.random.shuffle(test_set)

    im_true = test_set[np.arange(n)]
    decoded_imgs = transc(im_true)

    plt.figure(figsize=(20, 4))
    for i in np.arange(n):
        x_true = im_true[i]
        x_pred = decoded_imgs[i]
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_true.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_pred.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def orig_vs_transformed(DL,index=None,part='test',cmap='gray',clean=True):
    if part == 'test':
        X = DL.x_test
        sX = DL.sx_test
    elif part == 'train':
        X = DL.x_train
        sX = DL.sx_train
    
    if index is None:
        index = np.random.randint(0,len(X))
    
    
    figure,axs = plt.subplots(1,2)
    axs[0].imshow(X[index].reshape(28,28),cmap=cmap,vmin=0,vmax=1)
    axs[1].imshow(sX[index].reshape(DL.scale*28,DL.scale*28),cmap=cmap,vmin=0,vmax=1)
    
    if clean:
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
    
    return (part,index)

def joint_plot(x_var,y_var,kind,xlabel,ylabel):
    g = sns.jointplot(x_var,y_var,kind=kind)
    g.set_axis_labels(xlabel=xlabel,ylabel=ylabel)
    
def Z_color_scatter(z,idx,c):
    plt.scatter(z[:,idx[0]],z[:,idx[1]],c=c)
    plt.colorbar()
    plt.title(r"dx in $\hat{Z}$")
    plt.xlabel(r"$\hat{Z_%d}$"%idx[0])
    plt.ylabel(r"$\hat{Z_%d}$"%idx[1])
    
def enc_dec_samples(x_inp,sx_inp,Z,Y,generator,num_examples=5):
    
    fig,axs = plt.subplots(num_examples,4,figsize=(6,8))
    choices = np.random.choice(np.arange(len(Z)),num_examples)
    
    lat_vec_ = np.concatenate([Y[choices],Z[choices]],axis=1)
    dec_test = generator.predict(lat_vec_)
    
    for i,idx in enumerate(choices):
        rec_true_im = x_inp[idx].reshape(28,28)
        in_im = sx_inp[idx].reshape(28*2,28*2)
        dec_im = dec_test[i].reshape(28*2,28*2)

        axs[i,0].imshow(rec_true_im,cmap='gray')
        remove_labels(axs[i,0])
        # axs[i,0].set_xticklabels([])
        # axs[i,0].set_yticklabels([])

        axs[i,1].imshow(in_im,cmap='gray')
        remove_labels(axs[i,1])
        # axs[i,1].set_xticklabels([])
        # axs[i,1].set_yticklabels([])

        axs[i,2].imshow(dec_im,cmap='gray')
        remove_labels(axs[i,2])
        # axs[i,2].set_xticklabels([])
        # axs[i,2].set_yticklabels([])
    #     axs[2,i].set_xlabel("class: {}".format(str(np.argmax(y_class_enc[idx]))))

        axs[i,3].imshow(Y[idx].reshape(-1,1).T,cmap='gray',vmin=0,vmax=1)
        axs[i,3].set_xlabel("class: {}".format(str(np.argmax(Y[idx]))))
        remove_labels(axs[i,3])
        # axs[i,3].set_xticklabels([])
        # axs[i,3].set_yticklabels([])
        
def plot_train_history(hist_df,key,ax=None):
    if ax is not None:
        ax.plot(hist_df[key])
    else:
        plt.plot(hist_df[key])
        
    return ax