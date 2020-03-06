import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from xt_training import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import random as rand
import os
import inspect

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def roc_auc(y, logits, mode, fig=None, do_softmax=True):
    """Calculate ROC curve and AUC score.

    Arguments:
        y {torch.tensor} -- 1D array of labels.
        logits {torch.tensor} -- 2D array of probabilities (2 x N).
        mode {str} -- Name of test set.

    Keyword Arguments:
        fig {matplotlib.pyplot.Figure} -- ROC figure. If None, a new one is created. (default: None)

    Returns:
        tuple -- Tuple of AUC score and matplotlib figure object.
    """
    thresholds = np.arange(0, 1+1e-8, 0.005, dtype=np.float32)
    cm_array = metrics._confusion_matrix_array(
        logits.float(), y,
        thresholds, do_softmax=do_softmax
    )

    row_sums = cm_array.sum(dim=2)
    negatives = row_sums[:, 0]
    positives = row_sums[:, 1]

    fpr = cm_array[:, 0, 1] / negatives
    tpr = cm_array[:, 1, 1] / positives

    auc_score = metrics._auc(fpr, tpr).cpu().item()

    fig = metrics._generate_plot(
        fpr.cpu(), tpr.cpu(), thresholds.round(3).astype(str),
        'False positive rate', 'True positive rate',
        f"{mode} (AUC: {auc_score:.3f})", fig
    )

    out_data = pd.DataFrame({'fpr': fpr.numpy(), 'tpr': tpr.numpy(), 'thresholds': thresholds})

    return auc_score, fig, out_data


def pr_auc(y, logits, mode, fig=None, do_softmax=True):
    """Calculate PR curve and AUC score.

    Arguments:
        y {torch.tensor} -- 1D array of labels.
        logits {torch.tensor} -- 2D array of probabilities (2 x N).
        mode {str} -- Name of test set.

    Keyword Arguments:
        fig {matplotlib.pyplot.Figure} -- PR figure. If None, a new one is created. (default: None)

    Returns:
        tuple -- Tuple of AUC score and matplotlib figure object.
    """
    if do_softmax:
        probs = F.softmax(logits, dim=1).cpu().detach().numpy()
    else:
        probs = logits.cpu().detach().numpy()
    prec, recall, thresholds = precision_recall_curve(y, probs[:, 1], pos_label=1)
    auc_score = auc(recall, prec)

    fig = generate_plot(
        recall, prec, thresholds,
        'Recall', 'Precision',
        f"{mode} (AUC: {auc_score:.3f})", fig
    )

    return auc_score, fig


def generate_plot(x, y, thresholds, xlabel, ylabel, label, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax = fig.axes[0]

    p = ax.plot(x, y, label=label)
    ax.legend()
    
    ann_thresholds = np.arange(0.1, 1.0, 0.1)
    ann_inds = np.searchsorted(thresholds, ann_thresholds) - 1
    for i, t in zip(ann_inds, ann_thresholds):
        if i != -1:
            ax.annotate(
                str(round(t, 1)),
                xy=(x[i], y[i]),
                xytext=(x[i] + 0.05, y[i] - 0.05),
                color=p[0].get_color(),
                arrowprops={"arrowstyle": "-", "color": p[0].get_color()},
            )
    
    return fig


def print_metrics(labels, logits, threshold=0.5, plot_roc=False, save_dir=""):
    """Print metrics.

    Print confusion metrics for a given model output, labels and threshold. It also has
    functionality to save the roc curve.

    Arguments:
        labels {pytorch tensor} -- Labels tensor containing the true labels (tensor of shape (N)
            where N - batch size)
        logits {pytorch tensor} -- Logits tensor obtained from the model (tensor of shape (N,C)
            where N - batch size and C - number of classes)

    Keyword Arguments:
        threshold {float} -- The value used as threshold to convert the softmax output to 
        predictions (default: 0.5)
        plot_roc {bool} -- Boolean value determining whether or not to plot ROC 
        ( Receiver Operating Characteristic) (default: {False})
        save_dir {str} -- File path for saving the roc figure (default: {''})
    """

    preds = metrics.logit_to_label(logits, threshold).detach().numpy()
    labels = labels.cpu().detach().numpy()
    cm = confusion_matrix(labels, preds)
    print(cm)

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits).detach().numpy()

    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    fn_r = (FN) / (FN + TN)
    fp_r = (FP) / (FP + TP)
    print(f"FN ratio: {fn_r:.2f} and FP ratio: {fp_r:.2f}")

    if plot_roc:
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1], pos_label=1)
        plt.plot(fpr, tpr)
        plt.savefig(save_dir + ".png")


def split_objects(filepaths, train_prop, use_in_train=['background']):
    """Split dataset by object combinations. E.g., all glock+phone's are in either the train or
    test set, but not both.
    
    Arguments:
        filepaths {str} -- List of data file paths.
        train_prop {float} -- Proportion of data to keep in train set.
        use_in_train {list} -- Object combinations to ensure are in the train set.
    """
    rand.seed(2)
    objs = np.unique([fp.split('/')[-2] for fp in filepaths])
    objs = np.unique(['_'.join(o.split('_')[2:]) for o in objs]).tolist()
    print('Objects in dataset:\n', '\n'.join(objs))
    objs = [o for o in objs if o not in use_in_train]

    train_objs = rand.sample(objs, int(len(objs) * train_prop))
    train_objs = train_objs + use_in_train
    data_objs = ['_'.join(fp.split('/')[-2].split('_')[2:]) for fp in filepaths]

    # print(np.unique([obj for i, obj in enumerate(data_objs) if obj in train_objs]))
    # print(np.unique([obj for i, obj in enumerate(data_objs) if not obj in train_objs]))
    
    train_inds = [i for i, obj in enumerate(data_objs) if obj in train_objs]
    val_inds = [i for i in range(len(filepaths)) if not i in train_inds]
    rand.shuffle(train_inds)
    rand.shuffle(val_inds)

    return train_inds, val_inds


def split_samples(filepaths, train_prop):
    """Split dataset randomly.
    
    Arguments:
        filepaths {str} -- List of data file paths.
        train_prop {float} -- Proportion of data to keep in train set.
    """
    rand.seed(2)
    train_inds = rand.sample(range(len(filepaths)), int(train_prop * len(filepaths)))
    val_inds = [i for i in range(len(filepaths)) if not i in train_inds]

    return train_inds, val_inds


def load_model(config, device=device):
    config_path = os.path.dirname(inspect.getfile(config))
    model = config.model['cls'](**config.model['params']).to(device)

    use_best = False if not hasattr(config, 'use_best') else config.use_best
    if use_best:
        chkpt_path = os.path.join(config_path, 'best.pt')
    else:
        chkpt_path = os.path.join(config_path, 'latest.pt')
    model.load_state_dict(torch.load(chkpt_path, map_location=device))

    return model.eval()


def average(logits):
    preds = F.softmax(torch.as_tensor(logits), dim=2)
    return preds.mean(dim=0)


def plot_samples_ae(y,y_pred, label='test'):
    c = list(range(0, y.size(0)))
    inds = rand.sample(c,4)
    plt.figure(1)
    for count,ind in enumerate(inds):
        plt.subplot(4,2,2*count+1)
        plt.plot(y.cpu().detach()[ind].transpose(0,1))
        plt.gca().set_title('Original Signal')
        plt.subplot(4,2,2*count+2)
        plt.plot(y_pred.cpu().detach()[ind].transpose(0,1))
        plt.gca().set_title('Reproduced Signal')
    plt.tight_layout()
    plt.suptitle('Plot for samples in '+label+' set')
    plt.show()


def vis_tensorboard_embeddings(model, train_dataset, writer):

        ys, zs, label_names = [], [], []
        train_dataset.dataset.get_label = True
        train_dataset.dataset.get_tags = True
        for x, y, f in train_dataset:
            ys.append(y)
            label_names.append(f)
            x = x.to(device)
            z = model.encoder(x.unsqueeze(0))
            zs.append(z.detach().cpu())
        # ys = torch.stack(ys)
        # print(len(ys))
        zs = torch.stack(zs).squeeze(1)
        # print(zs.shape)
        writer.add_embedding(zs,metadata=list(zip(ys,label_names)), metadata_header=["Label value" , "Label Names"])