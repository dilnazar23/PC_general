import argparse
import torch
import torch.onnx
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from nn_models import Full2Layer, NN4PD, DetecDataset, Diods4DatasetAll
from pathlib import Path
from datetime import datetime

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()        # zero the gradients
        output = model(data)         # apply network
        loss = F.huber_loss(output, target) #loss fucntion as mse/huber  
        loss.backward()              # compute gradients
        optimizer.step()             # update weights
        
        if epoch % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(args, model, device, test_loader):
    test_loss = 0
    with torch.no_grad():   # suppress updating of gradients
        model.eval()        # toggle batch norm, dropout
        for batch_id, (data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.huber_loss(output, target, reduction='sum').item()  ## 
        model.train() # toggle batch norm, dropout back again
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

def export_model(model, device, save_path="trained_model", input_shape=(1,4)):

    # add time stamp to save path
    time_stamp= datetime.now().strftime('%m%d_%H%M')
    save_path = save_path + "_" + time_stamp

    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, f"{save_path}.pth")
    
    # Prepare model for ONNX export
    model.eval()    
    # Create dummy input tensor
    dummy_input = torch.randn(input_shape, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model saved as {save_path}.pth and {save_path}.onnx")

def check_cuda():
    print(f"CUDA is available: {torch.cuda.is_available()}")

    # Check CUDA version
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        
    # Check which device PyTorch is using
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is using: {device}")

    # Get the name of the CUDA device
    if torch.cuda.is_available():
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")

def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',type=str,default="4diod_test",help='filepath to sim data')
    parser.add_argument('--preprocess',type=bool,default=True,help='is data preprocessed')
    parser.add_argument('--batch',type=str,default=256,help='training set size')
    parser.add_argument('--net',type=str,default='shallow',help='shallow or pd4')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--mom',type=float,default=0.05,help='momentum')
    parser.add_argument('--init', type=float, default=0.1, help='initial weight size')
    parser.add_argument('--epochs',type=int,default=100,help='number of training epochs')
    args = parser.parse_args()

    check_cuda()
    device = torch.device('cuda')

    #kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # choose network architecture
    if args.net == 'shallow':
        net = Full2Layer().to(device)
        # load train set
        train_set = DetecDataset(args.src,2000)        
        train_set = torch.utils.data.TensorDataset(train_set.in_set,train_set.tar_set)
        train_loader = torch.utils.data.DataLoader(train_set,
                                    batch_size=args.batch)      
        
        # load test set
        test_set = DetecDataset(args.src,2000,True)
        test_set = torch.utils.data.TensorDataset(test_set.in_set,test_set.tar_set)
        test_loader = torch.utils.data.DataLoader(test_set,
                                    batch_size=test_set.__len__())
    elif args.net == 'pd4':
        net = NN4PD(4700,250).to(device)
        #load train set:
        root_path= Path(args.src)
        train_set = Diods4DatasetAll(root_path/"TrainSet",root_path/"TrainSet/data.pt")
        normaalise(train_set)
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=len(train_set))
        test_set = Diods4DatasetAll(root_path/"TestSet")
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=len(test_set))
    print('done')
 

    if list(net.parameters()):
        # weight initialization
        for m in list(net.parameters()):
            m.data.normal_(0,args.init)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)
    #adam
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,
                            weight_decay=0.0008)
    # training and testing loop
    for epoch in range(1, args.epochs * 100):
        train(args, net, device, train_loader, optimizer, epoch)
        if epoch%1000 == 0:
            test(args, net, device, test_loader)

    torch.save(net.state_dict(), "model_state.pth")
    input_shape = (1,4)
    export_model(net, device, save_path="2101", input_shape=input_shape)


if __name__ == '__main__':
    main()



    
