from nanaograd.nn.nn import Module


class MSELoss(Module):
    def forward(self, true_y, pred_y):
        return sum((yout-ygt)**2 for ygt, yout in zip(true_y,pred_y))/len(true_y)

    def __call__(self,true_y, pred_y):
        return self.forward(true_y,pred_y)
