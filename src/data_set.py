import torch
import numpy as np
import random

class data_set():
    """
    The class for data set. 

    :param xvec: array containing trajectory data 
    :type xvec: 2d numpy array

    :param weights: weights of trajectory data
    :type weights: 1d numpy array

    :ivar batch_size: current batch-size 
    :ivar K: number of data 
    :ivar active_index: indices of active states. 
        A state is included in mini-batch, if the entry is 1. 
    :ivar batch_uniform_weight: when True (default), the data is included in mini-batch with equal probability.
    """
    def __init__(self, xvec, weights) :
        self.X_vec = torch.from_numpy(xvec).double()
        self.weights = torch.from_numpy(weights).double()
        # Number of states 
        self.K = self.X_vec.shape[0]
        # Dimension of state
        self.tot_dim = self.X_vec.shape[1]
        # Indices of states that will appear in mini-batch. Use all by default.
        self.active_index = range(self.K)
        self.batch_size = self.K

        self.batch_uniform_weight = True 
        self.cum_weights = np.arange(1, self.K+1)
        
        print ('[Info]  Range of weights: [%.3e, %.ee]' % (self.weights.min(), self.weights.max()) )

    @classmethod 
    def from_file(cls, states_filename) :
        """
        Initialize the data_set from a data file.

        """
        # Reads states from file 
        state_weight_vec = np.loadtxt(states_filename, skiprows=1)

        print("[Info] %d sampled data loaded from file: %s" % (state_weight_vec.shape[0], states_filename))

        return cls(state_weight_vec[:,:-1], state_weight_vec[:,-1])

    def set_nonuniform_batch_weight(self) :
        """
        By default, :py:data:`batch_uniform_weight` is True and indices for mini-batch are selected with equal
        probability, unless this function is called, which set
        :py:data:`batch_uniform_weight` to False, and indices for mini-batch will
        be selected randomly according to their weights. 
        """
        self.batch_uniform_weight = False
        self.cum_weights = np.cumsum(self.weights).numpy()

    def weights_minibatch(self) :
        """
        Return the array containing the weights of states in mini-batch. 
        The array will be constant 1 if :py:data:`batch_uniform_weight` is False.
        """
        if self.batch_uniform_weight == True : 
            return self.weights[self.active_indices]
        else :
            return torch.ones(self.batch_size, dtype=torch.float64)

    def generate_minibatch(self, batch_size, minibatch_flag=True) :
        """
        Generate a mini-batch.

        :param batch_size: size of mini-batch.
        :type batch_size: int 

        :param minibatch_flag: control whether generate mini-batch or entire data set.
        :type minibatch_flag: bool
        """

        if minibatch_flag == True :
            # Randomly generate indices of samples from data set 
            self.active_indices = random.choices(range(self.K), cum_weights=self.cum_weights, k=batch_size)
            self.batch_size = batch_size
        else :
            if batch_size != self.K:
                print ("\tWarning: all states (%d) are selected. Value of batch_size (%d) is ingored." % (self.K, batch_size))
            self.active_indices = range(self.K)
            self.batch_size = self.K

        #  Choose samples corresonding to those indices,
        #  and reshape the array to avoid the problem when dim=1
        self.x_batch = torch.reshape(self.X_vec[self.active_indices], (self.batch_size, self.tot_dim))

        # This is needed in order to compute spatial gradients
        self.x_batch.requires_grad = True

    def pre_processing_layer(self) :
        """
        Process data before it is sent to neural networks.
        This function returns the mini-batch itself.
        It can be overrided in child class (i.e., aligning data wrt a reference).
        """
        return self.x_batch


class MD_data_set(data_set) :
    """
    This class is a child class of :class:`data_set`, used for data of
    molecular system.
    """

    def __init__(self, xvec, weights) :
        super(MD_data_set, self).__init__(xvec, weights)
        # System in 3D 
        self.dim = 3
        # Number of atoms
        self.num_atoms = int(self.tot_dim / self.dim)

    @classmethod 
    def from_file(cls, states_filename) :
        """
        Initize the data set from file.
        """
        self = super(MD_data_set, cls).from_file(states_filename) 
        return self

    def load_ref_state(self) :
        """
        Load a reference configuration from the file `./data/ref_state.txt`.
        It is used to align the data.
        """
        ref_state_filename = './data/ref_state.txt' 
        ref_state_file= open(ref_state_filename)
        contents = ref_state_file.readlines()
        self.ref_num_atoms = int(contents[0].rstrip())
        self.ref_index = [int(x) for x in contents[1].rstrip().split()]
        ref_tmp = np.loadtxt(ref_state_filename, skiprows=2)
        self.ref_x = torch.from_numpy(ref_tmp).double().reshape((self.ref_num_atoms, self.dim))
        return self

    def align(self) :
        """
        Align all states in the data set with respect to the reference
        configuration. 
        
        This function implements the Kabash's algorithm, which minimizes the 
        the root mean squared deviation of states with respect to the reference configuration.
        """
        x = self.x_batch.reshape((-1, self.num_atoms, self.dim))
        x_angle_atoms = x[:,self.ref_index, :]

        # center
        x_c = torch.mean(x_angle_atoms, 1, True)
        # translation
        x_notran = x_angle_atoms - x_c 

        xtmp = x_notran.permute((0,2,1)).reshape((-1,self.ref_num_atoms))
        prod = torch.matmul(xtmp, self.ref_x).reshape((-1, self.dim, self.dim))
        u, s, vh = torch.linalg.svd(prod)

        diag_mat = torch.diag(torch.ones(3)).double().unsqueeze(0).repeat(self.batch_size, 1, 1)

        sign_vec = torch.sign(torch.linalg.det(torch.matmul(u, vh))).detach()
        diag_mat[:,2,2] = sign_vec

        rotate_mat = torch.bmm(torch.bmm(u, diag_mat), vh)

        return torch.matmul(x-x_c, rotate_mat).reshape((-1, self.tot_dim) )

    def pre_processing_layer(self) :
        """
        Return the aligned data.
        """

        return self.align()

