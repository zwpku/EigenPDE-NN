import torch
import numpy as np
import random

class feature_tuple():
    """
    Class containing features.

    :param features: user-defined features. 
    :type features: string

    Currently supported types of features: 
      dihedral angle, angle, bond.

    """
    def __init__(self, features):
        if features == None :
            self.num_features = 0
        else :
            self.f_dim = 0 
            self.f_tuples = []
            for feature_str in features.split(';') :
                feature_str_split = feature_str.strip("{( )}\n").split(',')
                # diheral angle, angle and bond length are considered in the current implementation
                f_name = feature_str_split[0]
                if f_name == 'dihedral' : 
                    span = 2
                if f_name == 'bond' : 
                    span = 1
                if f_name == 'angle' : 
                    span = 1
                ag = [int(xx) for xx in feature_str_split[1:]]
                self.f_tuples.append([f_name, ag, list(range(self.f_dim, self.f_dim + span))])
                self.f_dim += span

            self.num_features = len(self.f_tuples) 
            self.f_tuples = tuple(self.f_tuples)

    def show_features(self) :
        """
        Display information of features.
        """

        print ('[Info]  No. of features: ', self.num_features)

        if self.num_features > 0 :
            print ('\tFeature dimension: ', self.f_dim)

            for i in range(self.num_features): 
              print ('\t%dth feature: ' % (i+1), self.f_tuples[i]) 

    def convert_atom_ix_by_file(self, ids_filename) :
        """
        Convert indices in features according to the index file.
        """

        if self.num_features == 0 :
            return 
        self.ix_array = np.loadtxt(ids_filename, skiprows=1, dtype=int)
        K = len(self.ix_array)
        idx_vec = np.ones(int(np.max(self.ix_array))+1, dtype=int) * -1 
        for idx in range(K) :
            g_idx = self.ix_array[idx] 
            idx_vec[g_idx] = idx

        tmp_f_tuples = []
        for f_idx in range(self.num_features) :
            ag = [idx_vec[idx] for idx in self.f_tuples[f_idx][1]]
            tmp_f_tuples.append([self.f_tuples[f_idx][0], ag, self.f_tuples[f_idx][2]])

        self.f_tuples = tuple(tmp_f_tuples)

        print ('[Info] Feature-tuple converted to local indices: ')
        for f_id in range(len(self.f_tuples)) :
            print ('\t%dth feature:' % (f_id+1), self.f_tuples[f_id])

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
    :ivar features: the tuple of features (the state is first transformed to
        features, before it is passed to neural networks).
    :ivar num_features: number of features 
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
        
        self.features = feature_tuple(None)
        self.num_features = 0 

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

    def set_features(self, features) :
        """
        Set the features of the data set.
        :param features: list of features.
        :type features: 
        """
        # List of features
        self.features = features
        # Number of features used
        self.num_features = self.features.num_features
        self.f_dim = self.features.f_dim

    def dim_of_features(self) :
        """
        Number of features. 
        """
        return self.f_dim

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
        This function returns the mini-batch itself (i.e., no features is used
        to transform data, in base class). It can be overrided in child class (i.e., mapping data to features).
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

    def map_to_feature(self, idx):
        """
        Map the states in the mini-batch to one feature.

        :param idx: idx of the feature to which states are mapped.

        """
        x = self.x_batch.reshape((-1, self.num_atoms, self.dim))
        f_name = self.features.f_tuples[idx][0]
        ag = self.features.f_tuples[idx][1]
        if f_name == 'dihedral': 
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r34 = x[:, ag[3], :] - x[:, ag[2], :]
            n1 = torch.cross(r12, r23)
            n2 = torch.cross(r23, r34)
            cos_phi = (n1*n2).sum(dim=1, keepdim=True)
            sin_phi = (n1 * r34).sum(dim=1, keepdim=True) * torch.norm(r23, dim=1, keepdim=True)
            radius = torch.sqrt(cos_phi**2 + sin_phi**2)
            return torch.cat((cos_phi / radius, sin_phi / radius), dim=1)

        if f_name == 'angle': 
            r21 = x[:, ag[0], :] - x[:, ag[1], :]
            r23 = x[:, ag[2], :] - x[:, ag[1], :]
            r21l = torch.norm(r21, dim=1, keepdim=True)
            r23l = torch.norm(r23, dim=1, keepdim=True)
            cos_angle = (r21 * r23).sum(dim=1, keepdim=True) / (r21l * r23l)
            return cos_angle

        if f_name == 'bond': 
            r12 = x[:, ag[1], :] - x[:, ag[0], :]
            return torch.norm(r12, dim=1, keepdim=True)

    def map_to_all_features(self):
        """
        Map the states in the mini-batch to all features, by calling
        :py:meth:`map_to_feature` for each feature.

        :return: 2d torch tensor, whose dimensions are (batch_size, num_features).
        """
        for i in range(self.num_features) :
            if i == 0 :
                xf = self.map_to_feature(i) 
            else :
                # Each column corresponds to one feature 
                xf = torch.cat((xf, self.map_to_feature(i)), dim=1)
        return xf

    def write_all_features_file(self, feature_filename) :
        """
        Map all states to features, and write the features to file.

        :param feature_filename: filename to output features.
        """
        self.generate_minibatch(self.K, False)
        # Append weights to last column (after features) 
        xf_w = torch.cat((self.map_to_all_features(), self.weights.reshape((-1,1))), dim=1).detach().numpy() 
        np.savetxt(feature_filename, xf_w, header='%d %d' % (self.K, self.f_dim + 1), comments="", fmt="%.10f")

    def pre_processing_layer(self) :
        """
        If features are defined, then map data to features. Otherwise, return
        the aligned data.

        """
        if self.num_features > 0 :
            return self.map_to_all_features() 
        else :
            return self.align()

