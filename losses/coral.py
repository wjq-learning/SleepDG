import torch
import torch.nn as nn


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def coral_cal(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    def forward(self, features, domains):
        bz = features.shape[0]
        loss = 0
        num = 0
        for i in range(bz):
            for j in range(i + 1, bz):
                if domains[i] != domains[j]:
                    num += 1
                    loss += self.coral_cal(features[i], features[j])
        if num > 1:
            loss /= num
        print(loss)
        return loss

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    torch.cuda.set_device(7)
    features = torch.randn((16, 20, 512)).cuda()
    domains = torch.tensor([0, 1, 0, 1, 2, 1, 3, 2, 0, 1, 0, 1, 2, 1, 3, 2])

    loss_function = CORAL()
    loss = loss_function(features, domains)
