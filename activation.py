import torch
import torch.nn as nn


# class for Softmax Activation used in Attention Mechanism and Final Projection, which gives Probablic Distribution over the given logits/tensors
class Softmax(nn.Module):

    def __init__(self, axis, keepdim=True):

        """
        SoftMax:

        The softmax function computes a new vector where each element is transformed,
        such that the values are all positive and sum up to 1.

        More clearly, it projects the input vector into probablities on the scale from 0 to 1 based on their attention. Sum of the Output Vector will be 1.
        """

        super(Softmax, self).__init__()

        self.axis = axis # axis along the softmax is applied
        self.keepdims = keepdim # whether to keep the structure of the dimension but shape will be 1 on guven axis or it'll be squeezed along the gven axis

    def forward(self, x):

        """
        Input:
            x: Attention Vector
        Output:
            x: Probablic Distribution along the given axis
        """

        logsumexp = torch.logsumexp(x, dim=self.axis, keepdim=self.keepdims) # logsumexp is used here to avoid underflow by division by large numbers. you can also use normal sumexp
        prob = torch.exp(x - logsumexp) # Element Wise Subtraction
        return prob # Output Probablities
