import torch
import data_set

class MySequential(torch.nn.Module):
    def __init__(self, size_list, activation_name):
        super(MySequential, self).__init__()

        self.nn_dims = size_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.nn_dims[i], self.nn_dims[i+1]) for i in range(len(self.nn_dims)-1)])

        try :
            activation = getattr(torch.nn, activation_name) 
            print ('[Info] Build NN with activation function %s.' % activation_name)
        except : 
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

    def feature_forward(self, xf):
        for i in range(len(self.nn_dims) - 1) :
            xf = self.linears[i](xf)
            if i < len(self.nn_dims) - 2 :
                xf = self.activations[i](xf)
        return xf

    # x should be an instance of the class data_set 
    def forward(self, x):
        xf = x.map_to_all_features()
        x.align()
        return self.feature_forward(xf) 

# Network whose ith output gives the ith eigenfunction
class MyNet(torch.nn.Module):
    def __init__(self, size_list, activation_name, d_out):
        super(MyNet, self).__init__()
        self.d_out = d_out
        self.nets = torch.nn.ModuleList([MySequential(size_list, activation_name) for i in range(d_out)])

    def shift_and_normalize(self, mean_list, var_list) :
        for i in range(self.d_out) :
            self.nets[i].shift_and_normalize(mean_list[i], var_list[i]) 

    def feature_forward(self, x):
        return torch.cat([self.nets[i].feature_forward(x) for i in range(self.d_out)], dim=1)

    def forward(self, x):
        return torch.cat([self.nets[i](x) for i in range(self.d_out)], dim=1)

