#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tonic
import torchvision
import sys
parameters_file = sys.argv[1]
out_file_path = sys.argv[2]
with open(r'./parameters/'+parameters_file) as file:

    parameters_list = yaml.load(file, Loader=yaml.FullLoader)
    print(parameters_list)

LR = parameters_list["LR"] 
HIDDEN_FEATURES = parameters_list["HIDDEN_FEATURES"] 
tau_syn_inv=torch.tensor(parameters_list["tau_syn_inv"] ) 
tau_mem_inv=torch.tensor(parameters_list["tau_mem_inv"] )
alpha=parameters_list["alpha"] 
v_th=torch.tensor(parameters_list["v_th"] )
BATCH_SIZE = parameters_list["BATCH_SIZE"] 
filter_time = parameters_list["filter_time"] 
drop_probability = parameters_list["drop_probability"] 
time_factor = parameters_list["time_factor"] 
spatial_factor= parameters_list["spatial_factor"] 
EPOCHS  = parameters_list["EPOCHS"] 

transform = tonic.transforms.Compose(
    [

        tonic.transforms.DropEvent(drop_probability=drop_probability),
        tonic.transforms.Downsample(time_factor = time_factor,spatial_factor = spatial_factor),
        tonic.transforms.ToSparseTensor(merge_polarities=True),
    ]
)

download = True
trainset = tonic.datasets.DVSGesture(save_to='./dvsgesture/',download=False,train=True)
testset = tonic.datasets.DVSGesture(save_to='./dvsgesture/',download=False,transform=transform,train=False)


# We can have a look at how a sample of one digit looks like. The event camera's output is encoded as events that have x/y coordinates, a timestamp and a polarity that indicates whether the lighting increased or decreased at that event. The events are provided in an (NxE) array. Let's have a look at the first example in the dataset. Every row in the array represents one event of timestamp, x, y, and polarity.

events = trainset[0][0]
#print(events.shape)

#tonic.utils.plot_event_grid(events, trainset.ordering)
# And this one is the target class:
#trainset[0][1]


# We wrap the training and testing sets in PyTorch DataLoaders that facilitate file loading. Note also the custom collate function __pad_tensors__ , which makes sure that all sparse tensors in the batch have the same dimensions


# add sparse transform to trainset, previously omitted because we wanted to look at raw events
trainset.transform = transform

train_loader = torch.utils.data.DataLoader(trainset,
                                        batch_size=BATCH_SIZE,
                                        collate_fn=tonic.utils.pad_tensors,
                                        shuffle=True
)

test_loader = torch.utils.data.DataLoader(testset,
                                        batch_size=BATCH_SIZE,
                                        collate_fn=tonic.utils.pad_tensors,
                                        shuffle=False
)


# ## Defining a Network
# 
# Once the data is encoded into spikes, a spiking neural network can be constructed in the same way as a one would construct a recurrent neural network.
# Here we define a spiking neural network with one recurrently connected layer
# with `hidden_features` LIF neurons and a readout layer with `output_features` and leaky-integrators. As you can see, we can freely combine spiking neural network primitives with ordinary `torch.nn.Module` layers.
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState
from typing import NamedTuple

class SNNState(NamedTuple):
    lif0 : LIFState
    readout : LIState


class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, tau_syn_inv, tau_mem_inv, record=False, dt=1e-6):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=alpha, 
                            v_th=v_th,
                            tau_syn_inv=tau_syn_inv,
                            tau_mem_inv=tau_syn_inv,
                           ),
            dt=dt                     
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _,_ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
              LIFState(
                z = torch.zeros(seq_length, batch_size, self.hidden_features),
                v = torch.zeros(seq_length, batch_size, self.hidden_features),
                i = torch.zeros(seq_length, batch_size, self.hidden_features)
              ),
              LIState(
                v = torch.zeros(seq_length, batch_size, self.output_features),
                i = torch.zeros(seq_length, batch_size, self.output_features)
              )
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts,:] = s1.z
                self.recording.lif0.v[ts,:] = s1.v
                self.recording.lif0.i[ts,:] = s1.i
                self.recording.readout.v[ts,:] = so.v
                self.recording.readout.i[ts,:] = so.i
            voltages += [vo]

        return torch.stack(voltages)




# ## Training the Network
# 
# The final model is then simply the sequential composition of our network and a decoding step.

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y


INPUT_FEATURES = np.product(trainset.sensor_size)

OUTPUT_FEATURES = len(trainset.classes)


if torch.cuda.is_available():
    
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    snn=SNN(
      input_features=int(INPUT_FEATURES*(spatial_factor**2)),
      hidden_features=HIDDEN_FEATURES,
      output_features=OUTPUT_FEATURES,
      tau_syn_inv=tau_syn_inv, 
      tau_mem_inv=tau_mem_inv
    ),
    decoder=decode
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model


# What remains to do is to setup training and test code. This code is completely independent of the fact that we are training a spiking neural network and in fact has been largely copied from the pytorch tutorials.
from tqdm import tqdm, trange



def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader):
        data, target = data.to(device).to_dense().permute([1,0,2,3,4]), torch.LongTensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


# Just like the training function, the test function is standard boilerplate, common with any other supervised learning task.

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    true_all = []

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device).to_dense().permute([1,0,2,3,4]), torch.LongTensor(target).to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            true_all.append(target.cpu().numpy())
            pred_all.append(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy, pred_all, true_all



training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(False)

for epoch in range(EPOCHS):
    print("Epoch:",epoch)
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
    test_loss, accuracy, pred_all, true_all = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    np.save(out_file_path+'/test_acc.npy', accuracies)
    np.save(out_file_path+'/test_loss.npy', test_losses)
    np.save(out_file_path+'/total_loss.npy', mean_losses)
    np.save(out_file_path+'/pred.npy', pred_all)
    np.save(out_file_path+'/true.npy', true_all)
    print("Accuracies:",accuracies)

print(f"final accuracy: {accuracies[-1]}")




