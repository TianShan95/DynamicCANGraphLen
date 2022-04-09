import torch.nn as nn


# 参考 https://github.com/lanpa/tensorboardX/issues/319
class Wrapper(nn.Module):
    def __init__(
            self,
            net1,
            net2
    ):
        super().__init__()

        # build policy and value functions
        self.net1 = net1
        self.net2 = net2

    def forward(self, state, act):
        # Perform a forward pass through all the networks and return the result
        q1 = self.net1(state)
        q2 = self.net2(state, act)
        return q1, q2
