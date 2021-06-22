import torch
import numpy as np
import random

class feature_tuple():
    def __init__(self, features):
        if features == None :
            self.num_features = 0
        else :
            self.f_dim = 0 
            self.f_tuples = []
            for feature_str in features.split(';') :
                feature_str_split = feature_str.strip("{( )}").split(',')
                # diheral angle, angle and bond length are considered in the current implementation
                f_name = feature_str_split[0]
                if f_name == 'dihedral' : 
                    self.f_dim += 2
                if f_name == 'bond' : 
                    self.f_dim += 1
                if f_name == 'angle' : 
                    self.f_dim += 1
                ag = [int(xx) for xx in feature_str_split[1:]]
                self.f_tuples.append([f_name, ag])

            self.num_features = len(self.f_tuples) 
            self.f_tuples = tuple(self.f_tuples)

    def show_features(self) :
        print ('No. of features: ', self.num_features)

        if self.num_features > 0 :
            print ('Dim of features: ', self.f_dim)

            for i in range(self.num_features): 
              print ('%dth feature:' % (i+1), self.f_tuples[i]) 

class data_set():
    def __init__(self, xvec, weights) :
        self.X_vec = torch.from_numpy(xvec)
        self.weights = torch.from_numpy(weights)
        # Number of states 
        self.K = self.X_xvec.shape[0]
        # Dimension of state
        self.tot_dim = self.X_vec.shape[1]
        # Indices of states that will appear in mini-batch
        self.active_index = range(self.K)
        self.batch_size = self.K

        self.batch_uniform_weight = True 
        self.cum_weights = np.arange(1, self.K+1)
        
        self.features = feature_tuple(None)
        self.num_features = 0 

        print ('[Info] Range of weights: [%.3e, %.ee]' % (self.weights.min(), self.weights.max()) )

    def __init__(self, states_filename ) :

        # Reads states from file 
        state_weight_vec = np.loadtxt(states_filename, skiprows=1)

        print ("[Info] loaded data from file: %s\n\t%d states" % (states_filename, state_weight_vec.shape[0]), flush=True)
        self.__init__(state_weight_vec[:,:-1], state_weight_vec[:,-1])


    def set_nonuniform_batch_weight(self) :
        self.batch_uniform_weight = False
        self.cum_weights = np.cumsum(self.weights).numpy()

    def weights_minbatch(self) :
        if self.batch_uniform_weight == True : 
            return self.weights[self.active_indices]
        else :
            return torch.ones(self.batch_size, dtype=torch.float64)

    def set_features(self, features) :
        # List of features
        self.features = features
        # Number of features used
        self.num_features = self.features.num_features
        self.f_dim = self.features.f_dim

    def dim_of_features(self) :
        return self.f_dim

    def generate_minbatch(self, batch_size, minbatch_flag=True) :

        if minbatch_flag == True :
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

    def map_to_feature(self, idx):
        pass

    def map_to_all_features(self):
        if self.num_features > 0 :
            for i in range(self.num_features) :
                if i == 0 :
                    xf = self.map_to_feature(i) 
                else :
                    xf = torch.cat((xf, self.map_to_feature(i)), dim=1)
            return xf
        else :
            return self.x_batch

    def write_all_features_file(self, feature_filename) :
        # Use all data
        self.generate_minbatch(self.K, False)
        xf = self.map_to_all_features().detach().numpy() 
        np.savetxt(feature_filename, xf, header='%d %d' % (self.K, self.f_dim), comments="", fmt="%.10f")

# data of molecular system
class MD_data_set(data_set) :
    def __init__(self, states_filename) :

        super(MD_data_set, self).__init__(states_filename)
        # System in 3D 
        self.dim = 3
        # Number of atoms
        self.num_atoms = int(self.tot_dim / self.dim)

    # features of data in mini-batch
    def map_to_feature(self, idx):
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

