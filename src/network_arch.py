import torch
import data_set

class MySequential(torch.nn.Module):
    """
      This class implements a feedforward neural network.

      :param size_list: number of neurons for each layer.
      :type size_list: list

      :param activation_name: name of activation function.
          It should match one of the names of non-linear activations functions for |torch_activation_link|.
      :type activation_name: string

      .. |torch_activation_link| raw:: html

         <a href="http://pytorch.org/docs/stable/nn.html" target="_blank"> PyTorch</a>
      
    """

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
        """
        Substract the neural network function by `mean` and divide it by :math:`\sqrt{var}`.
        This is achieved by modifying parameters of the neural network.
        """

        with torch.no_grad() :
            self.linears[-1].bias -= mean 
            self.linears[-1].weight /= torch.sqrt(var) 
            self.linears[-1].bias /= torch.sqrt(var)

    def forward(self, x):
        """
        Map the input data set to output tensor by the neural network.

        :param x: input data set 
        :type x: :py:mod:`data_set.data_set`

        :return: output tensor
        :rtype: torch tensor
        """
        xf = x.pre_processing_layer() 

        for i in range(len(self.nn_dims) - 1) :
            xf = self.linears[i](xf)
            if i < len(self.nn_dims) - 2 :
                xf = self.activations[i](xf)

        return xf

class MyNet(torch.nn.Module):
    """
    List of :py:mod:`network_arch.MySequential` neural networks. The 
    ith entry is used to store the ith eigenfunction. 

    :param size_list: number of neurons for each layer. 
       Each neural network has the same `size_list`.
    :type size_list: list 

    :param activation_name: name of the non-linear activation function.
    :type activation_name: string 

    :param d_out: number of neural networks in the list.
    :type d_out: int
    """

    def __init__(self, size_list, activation_name, d_out):
        super(MyNet, self).__init__()
        self.d_out = d_out
        self.nets = torch.nn.ModuleList([MySequential(size_list, activation_name) for i in range(d_out)])

    def shift_and_normalize(self, mean_list, var_list) :
        """
        For each neural networks in the list, substract and divide the function 
        by the values specified in the mean_list and var_list. It simply
        calls the function :py:meth:`network_arch.MySequential.shift_and_normalize`.

        :param mean_list: list of mean values. 
        :type mean_list: list of double
        
        :param var_list: list of variances.
        :type var_list: list of positive double
        """

        for i in range(self.d_out) :
            self.nets[i].shift_and_normalize(mean_list[i], var_list[i]) 

    def forward(self, x):
        """
        Map the input data set `x` to list of output tensors. For each neural
        network in the list, it simply calls :py:meth:`network_arch.MySequential.forward` function.

        :param x: input data set
        :type x: :py:mod:`data_set.data_set`

        """
        return torch.cat([self.nets[i](x) for i in range(self.d_out)], dim=1)

