import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os   

class Full2Layer(nn.Module): ## 2 layers nn with 2 Relu
    # two fully connected sigmoid+relu layers 
    def __init__(self):
        super(Full2Layer, self).__init__()
        self.in2hid1 = nn.Linear(32, 16)
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None
        self.hids = nn.Linear(16,8)
        self.hidss = nn.Linear(8,4)
        self.hid2out = nn.Linear(4, 2)

    def forward(self, input):
        self.hid1 = F.leaky_relu(self.in2hid1(input))
        self.hid2 = F.relu(self.hids(self.hid1))
        self.hid3 = F.leaky_relu(self.hidss(self.hid2))
        out = F.leaky_relu(self.hid2out(self.hid3))
        return out


class NN4PD(nn.Module):
    def __init__(self, mean, std):
        super(NN4PD, self).__init__()
        # Store mean and std for normalization
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

        # Define the layers
        self.in2hid1 = nn.Linear(4, 100)
        self.hid_fc1 = nn.Linear(100, 64)
        self.hid_fc2 = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 2)

    def normalise(self, input):
        # Normalize input using stored mean and std
        return (input - self.mean.to(input.device)) / self.std.to(input.device)

    def forward(self, input):
        # Normalize input
        normed_input = self.normalise(input)
        # Forward pass through the network
        hid1 = F.leaky_relu(self.in2hid1(normed_input))
        hid2 = F.leaky_relu(self.hid_fc1(hid1))
        hid3 = F.leaky_relu(self.hid_fc2(hid2), 0.5)
        out = self.hid2out(hid3)
        return out


class DetecDataset(Dataset):
    def __init__(self, raw_data_path, set_size,isTest=False):
        self.file_path = raw_data_path        
        self.set_size = set_size
        self.is_test = isTest        
        self.pd_loc = np.array([[250., 381.],
            [266., 325.],
            [282., 405.],
            [302., 371.],
            [318., 290.],
            [346., 341.],
            [386., 336.],
            [376., 297.],
            [386., 245.],
            [330., 229.],
            [410., 213.],
            [376., 193.],
            [295., 177.],
            [346., 149.],
            [341., 109.],
            [302., 119.],
            [250., 109.],
            [234., 165.],
            [218.,  85.],
            [198., 119.],
            [182., 200.],
            [154., 149.],
            [114., 154.],
            [124., 193.],
            [114., 245.],
            [170., 261.],
            [ 90., 277.],
            [124., 297.],
            [205., 313.],
            [154., 341.],
            [159., 381.],
            [198., 371.]])
        self.in_set, self.tar_set = self.MakeSet()

    def MakeSet(self):        
        root_dir = self.file_path
        trans_mats = []
        # get diod readings in each image, get all the transfer function to the nominal image and store in th elist
        ## combine them in order x,y,suncave,sunflower
        # sample the reading with scattered diodes
        center_image = np.loadtxt("nominal.csv", delimiter=",")
        center_reads = self.DiodReads(center_image)
        for foldername, subfolders, filenames in os.walk(root_dir):    
            for filename in filenames:
                if filename.endswith('.csv'):  # Check if the file is a CSV
                    #print(filename)
                    filepath = os.path.join(foldername, filename)
                    image = np.loadtxt(filepath, delimiter=",")
                    diod_read = self.DiodReads(image)
                    T = np.diag((diod_read / center_reads))
                    trans_mats.append(T)
        train_set,tar_set = self.Stackup(center_reads,trans_mats)
        train_set = torch.tensor(np.array(train_set), dtype=torch.float16)
        tar_set = torch.tensor(np.array(tar_set), dtype=torch.float16)
        return train_set,tar_set
    
    ## read from profiler image
    def DiodReads(self,prof_image):
        diod_read = []
        for diod in self.pd_loc:    
            diod_read.append(prof_image[int(diod[0]),int(diod[1])])
        return np.array(diod_read)

    ## stack things up by x,y, suncave
    def Stackup(self,nominal,T):
        #nominal*x traansform*y transform * sunflower_angle * suncave angle
        stack_in = []
        stack_tar = []
        num = 0
        for i in range(0,7):
            for j in range(7,14):
                for k in range(14,21):                
                    if k == 17:
                        x_off = -1
                    elif k>17:
                        x_off = -2
                    else:
                        x_off = 0
                    for l in range(21,28):                    
                        if l==24:
                            y_off = -1
                        elif l>24:
                            y_off = -2
                        else:
                            y_off = 0
                        if num >= self.set_size:
                            if self.is_test:
                                stack_in.clear()
                                stack_tar.clear()
                                num = 0
                            else:
                                break         
                        stack_in.append(T[i] @ T[j] @ T[k] @ T[l]@ nominal)
                        stack_tar.append([(k-14)*3-8+x_off,(l-21)*3-8+y_off])
                        num += 1
        return stack_in,stack_tar    


    def __len__(self):
        return len(self.tar_set)

    def __getitem__(self, idx):
        return self.in_set[idx], self.tar_set[idx]     


class Diods4DatasetAll(Dataset):
    def __init__(self, root_path, preprocess_path=None):
        self.root_dir = root_path
        self.preprocess_path = preprocess_path

        # If preprocessed data exists, load it; otherwise, preprocess and save
        if preprocess_path and os.path.exists(preprocess_path):
            self.in_set, self.tar_set = torch.load(preprocess_path)
        else:
            self.in_set, self.tar_set = self._load_and_preprocess_data()
            if preprocess_path:
                torch.save((self.in_set, self.tar_set), preprocess_path)

    def _load_and_preprocess_data(self):
        # Preprocess and load data
        in_set = []
        out_set = []
        for data_file in os.listdir(self.root_dir):
            if data_file.endswith(".csv"):
                file_path = os.path.join(self.root_dir, data_file)
                file_data_in = np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=1024, usecols=(2, 3, 4, 5))
                file_data_out = np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=1024, usecols=(0, 1))
                in_set.append(file_data_in)
                out_set.append(file_data_out)

        in_set = torch.tensor(np.vstack(in_set), dtype=torch.float32)
        tar_set = torch.tensor(np.vstack(out_set), dtype=torch.float32)
        return in_set, tar_set

    def __len__(self):
        return len(self.tar_set)

    def __getitem__(self, idx):
        return self.in_set[idx], self.tar_set[idx]  

class LSTM_model(nn.Module):
    def __init__(self,num_input,num_hid,num_out,batch_size=1,num_layers=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid), 
                        torch.zeros(batch_size,self.num_hid))
        else:
            h_t, c_t = init_states
         
        NH = self.num_hid
        for t in range(seq_size):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :NH]),     # input gate
                torch.sigmoid(gates[:, NH:NH*2]), # forget gate
                torch.tanh(gates[:, NH*2:NH*3]),  # new values
                torch.sigmoid(gates[:, NH*3:]),   # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output

