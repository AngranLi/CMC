import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, num_classes=2):
        """ Combo Loss that combines dice loss and cross entropy loss
        
        Keyword Arguments:
            alpha {float} -- Weight passed to cross entropy loss  (default: {0.5})
            beta {float} --  Weight for cross entropy contribution to final loss (default: {0.5})
            smooth {int} -- Smoothness parameter (default: {1})
            num_classes {int} -- Number of classes (default: {2})
        """
        nn.Module.__init__(self)
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        c = y_true.size()[0]
        y_true = y_true.reshape(c, 1)
        y_true = (
            y_true
            == torch.arange(self.num_classes).reshape(1, self.num_classes).to(device)
        ).float()
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        intersection = torch.sum(y_pred * y_true)
        d = (2.0 * intersection + self.smooth) / (
            torch.sum(y_pred) + torch.sum(y_true) + self.smooth
        )
        y_pred = torch.clamp(y_pred, e, 1.0 - e)
        out = -(
            self.alpha
            * (
                (y_true * torch.log(y_pred))
                + ((1 - self.alpha) * (1.0 - y_true) * torch.log(1.0 - y_pred))
            )
        )
        weighted_ce = torch.mean(out, axis=-1)
        combo = (self.beta * weighted_ce) - ((1 - self.beta) * d)
        return combo


class CostLoss(nn.Module):
    def __init__(self, tp_w=0.25, fp_w=0.25, fn_w=0.25, tn_w=0.25, num_classes=2):
        """Calculate cost sensitive loss for the batch
        
        Keyword Arguments:
            tp_w {float} -- weight for true positives (default: {0.25})
            fp_w {float} -- weight for false positives (default: {0.25})
            fn_w {float} -- weight for false negatives (default: {0.25})
            tn_w {float} -- weight for true negatives (default: {0.25})
            num_classes {int} -- Number of classes (default: {2})
        """
        nn.Module.__init__(self)
        self.tp_w = tp_w  # TP
        self.fp_w = fp_w  # FP
        self.fn_w = fn_w  # FN
        self.tn_w = tn_w  # TN
        self.num_classes = num_classes
        self.nllLoss = nn.NLLLoss(reduction="none")
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, y_pred, y_true):

        log_prob = self.softmax(y_pred)
        prob = torch.exp(log_prob)
        preds = torch.argmax(prob, dim=1)
        weights = torch.zeros(preds.size()[0]).to(device)
        tp = (preds == 0) & (y_true == 0)
        fp = (preds == 1) & (y_true == 0)
        fn = (preds == 0) & (y_true == 1)
        tn = (preds == 1) & (y_true == 1)
        weights[tp] = self.tp_w
        weights[fp] = self.fp_w
        weights[fn] = self.fn_w
        weights[tn] = self.tn_w

        loss = self.nllLoss(log_prob, y_true)
        final_loss = torch.sum(loss * weights)
        return final_loss


class DiceLoss(nn.Module):

    """    """

    def __init__(self, smooth=1, num_classes=2):
        """Calculate dice loss for the batch
        
        Keyword Arguments:
            smooth {int} -- Smoothness parameter (default: {1})
            num_classes {int} -- Number of classes (default: {2})
        """
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        c = y_true.size()[0]
        y_true = y_true.reshape(c, 1)
        y_true = (
            y_true
            == torch.arange(self.num_classes).reshape(1, self.num_classes).to(device)
        ).float()
        y_true = y_true.view(c, -1)
        y_pred = y_pred.view(c, -1)
        intersection = torch.sum(y_pred * y_true)
        d = (2.0 * intersection + self.smooth) / (
            torch.sum(y_pred) + torch.sum(y_true) + self.smooth
        )
        return 1 - torch.sum(d) / c


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        """Calculate focal loss for the batch
        
        Keyword Arguments:
            alpha {float} -- Weight for handling class imbalance (default: 0.5)
            gamma {float} -- Weight for penalizing easy & hard examples (default: 2.0)
        """
        nn.Module.__init__(self)
        self.weight = torch.Tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nllLoss = nn.NLLLoss(weight=self.weight)

    def forward(self, input_tensor, target_tensor):
        log_prob = self.log_softmax(input_tensor)
        prob = torch.exp(log_prob)
        fix_weights = (1 - prob) ** self.gamma
        logits = fix_weights * log_prob
        return self.nllLoss(logits, target_tensor)


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        """Calculate hinge loss 
        
        Keyword Arguments:
            margin {float} -- Margin parameter that defines the boundary (default: {1.0})
        """
        nn.Module.__init__(self)
        self.margin = margin

    def forward(self, input_tensor, target_tensor):
        targets = target_tensor.clone().detach()
        targets[targets == 0] = -1
        hinge_loss = self.margin - targets * input_tensor[:, 1]
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)
