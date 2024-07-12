import torch.optim as optim
import torch.nn as nn

# オプティマイザーを取得する関数
def get_optimizer(optimizer_name, params, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(params, lr)
    if optimizer_name == 'momentum':
        return optim.SGD(params, lr, momentum=0.8)
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr)
    if optimizer_name == 'Adagrad':
        return optim.Adagrad(params, lr)
    if optimizer_name == 'RMSProp':
        return optim.RMSprop(params, lr)
    if optimizer_name == 'Adadelta':
        return optim.Adadelta(params, lr)

# 損失関数を取得する関数
def get_loss(loss_name):
    if loss_name == 'mse_loss':
        return nn.MSELoss()
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'binary_corss_entropy':
        return nn.BCELoss()
    if loss_name == 'nll_loss':
        return nn.NLLLoss()
    if loss_name == 'hinge_embedding_loss':
        return nn.HingeEmbeddingLoss()