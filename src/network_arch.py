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
    def __init__(self, size_list, activation_name, features):
        super(MySequential, self).__init__()

        # Be careful, not just use: self.nn_dims = size_list 
        self.nn_dims = size_list[:]
        self.features = features
        self.num_features = len(self.features)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.nn_dims[i], self.nn_dims[i+1]) for i in range(len(self.nn_dims)-1)])

        try :
            activation = getattr(torch.nn, activation_name) 
            print ('[Info] Build NN with activation function %s.' % activation_name)
        except : 
            #self.activations = torch.nn.ModuleList([activation()  for i in range(len(self.nn_dims)-2)])
            print ('[Warning] No activation function with name %s. Use CELU instead' % activation_name)
            activation = torch.nn.CELU

        self.activations = torch.nn.ModuleList([activation()  for i in range(len(self.nn_dims)-2)])

        # initialize the tensors
        for tt in self.linears:
            torch.nn.init.normal_(tt.weight, 0, 0.5)
            torch.nn.init.normal_(tt.bias, 0, 0.5)

    def shift_and_normalize(self, mean, var) :
        with torch.no_grad() :
            self.linears[-1].bias -= mean 
            self.linears[-1].weight /= torch.sqrt(var) 
            self.linears[-1].bias /= torch.sqrt(var)

    def extract_features(self, x):
        xf = torch.zeros(x.shape[0], 2 * self.num_features).double()
        num_atoms = int(x.shape[1] / 3)
        x = x.reshape((-1, num_atoms, 3))
        # Compute diheral angles for each group of 4 atoms
        for i in range(self.num_features) :
            ag = [int(xx) for xx in self.features[i][1:]]
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r34 = x[:, ag[3], :] - x[:, ag[2], :]
            n1 = torch.cross(r12, r23)
            n2 = torch.cross(r23, r34)
            cos_phi = (n1*n2).sum(dim=1)
            sin_phi = (n1 * r34).sum(dim=1) * torch.norm(r23, dim=1)
            xf[:, 2*i] = cos_phi
            xf[:, 2*i+1] = sin_phi

        return xf

    def forward(self, x):
        if self.num_features > 0 :
            xf = self.extract_features(x)
        else :
            xf = x

        for i in range(len(self.nn_dims) - 1) :
            xf = self.linears[i](xf)
            if i < len(self.nn_dims) - 2 :
                xf = self.activations[i](xf)
        return xf

# Network whose ith output gives the ith eigenfunction
class MyNet(torch.nn.Module):
    def __init__(self, size_list, activation_name, d_out, features):
        super(MyNet, self).__init__()
        self.d_out = d_out
        self.nets = torch.nn.ModuleList([MySequential(size_list, activation_name, features) for i in range(d_out)])
        #self.nets = torch.nn.ModuleList([Polynomial22(d_in=d_in, d_out=1) for i in range(d_out)])

    def shift_and_normalize(self, mean_list, var_list) :
        for i in range(self.d_out) :
            self.nets[i].shift_and_normalize(mean_list[i], var_list[i]) 

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
