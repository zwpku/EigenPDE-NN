import torch

class Polynomial13(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    def forward(self, x):
        return self.a + self.b * x + self.c * x**2 #+ self.d * x**3

class Polynomial22(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(Polynomial22, self).__init__()
        self.w = [torch.nn.Parameter(torch.randn(())) for i in range(4)]
        for i, w in enumerate(self.w):
            self.register_parameter('param %d' % i, w)
    def forward(self, x):
        tmp = self.w[0] + self.w[1] * x[:,0] + self.w[2] * x[:,0]**2 + self.w[3] * x[:,0]**3
        return torch.unsqueeze(tmp, 1)
        #return tmp

class MySequential(torch.nn.Module):
    def __init__(self, size_list, ReLU_flag):
        super(MySequential, self).__init__()
        self.nn_dims = size_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.nn_dims[i], self.nn_dims[i+1]) for i in range(len(self.nn_dims)-1)])
        if ReLU_flag :
            self.activations = torch.nn.ModuleList([torch.nn.ReLU() for i in range(len(self.nn_dims)-2)])
        else :
            self.activations = torch.nn.ModuleList([torch.nn.Tanh() for i in range(len(self.nn_dims)-2)])

        # initialize the tensors
        for tt in self.linears:
            torch.nn.init.normal_(tt.weight, 0, 0.5)
            torch.nn.init.normal_(tt.bias, 0, 0.5)

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1) :
            x = self.linears[i](x)
            if i < len(self.nn_dims) - 2 :
                x = self.activations[i](x)
        return x

# Network whose ith output gives the ith eigenfunction
class MyNet(torch.nn.Module):
    def __init__(self, size_list, ReLU_flag, d_out):
        super(MyNet, self).__init__()
        self.d_out = d_out
        self.nets = torch.nn.ModuleList([MySequential(size_list, ReLU_flag) for i in range(d_out)])
        #self.nets = torch.nn.ModuleList([Polynomial22(d_in=d_in, d_out=1) for i in range(d_out)])

    def forward(self, x):
        return torch.cat([self.nets[i](x) for i in range(self.d_out)], dim=1)

class DenseNet(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(DenseNet, self).__init__()
        self.nn_dims = [d_in, 2, 1, 2, d_out]
        self.W = [item for sublist in
                [[torch.nn.Parameter(torch.randn(sum(self.nn_dims[:i + 1]),
                    self.nn_dims[i + 1], requires_grad=True) * 0.5),
                    torch.nn.Parameter(torch.zeros(self.nn_dims[i + 1], requires_grad=True))] for i in range(len(self.nn_dims) - 1)]
                  for item in sublist]
        for i, w in enumerate(self.W):
            self.register_parameter('param %d' % i, w)

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = torch.matmul(x, self.W[2 * i]) + self.W[2 * i + 1]
            else:
#                x = torch.cat([x, pt.nn.functional.relu(pt.matmul(x, self.W[2 * i]) + self.W[2 * i + 1])], dim=1)
                x = torch.cat([x, torch.tanh(torch.matmul(x, self.W[2 * i]) + self.W[2 * i + 1])], dim=1)
        return x
