from torch_geometric.data import DataLoader
from model import *
import random
from tqdm import tqdm
from rotation import augmentation
from torch import optim
from arg_helper import *
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parse_arguments()
config = get_config(args.config_file)

np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

data = torch.load('sulc')
random.shuffle(data)

train_set = data[0:25]
valid_set = data[25:29]
test_set = data[29:]

train_set = augmentation(train_set, config.num)

train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = 1)
test_loader = DataLoader(test_set, batch_size = 1)

if config.model == 'ProbGraphUnet':
    model = ProbGraphUnet(config)
else:
    model = GraphUnet(config)



device = config.gpu if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,100,200], gamma=0.5)

train_loss_history=[]
valid_loss_history=[]

best_loss = 10e10
for epoch in tqdm(range(500)):
    model.train()
    train_loss = 0
    valid_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        weight = torch.bincount(data.y) / len(data.y)
        weight = 1 / weight
        weight = weight / weight.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss = criterion(out[0], data.y)
        loss.backward()
        train_loss += loss

        optimizer.step()

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)

            weight = torch.bincount(data.y) / len(data.y)
            weight = 1 / weight
            weight = weight / weight.sum()
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            for i in range(config.num_samples):
                loss = criterion(out[i], data.y)
                valid_loss += loss

    train_loss = train_loss / len(train_set)
    valid_loss = valid_loss / (len(valid_set) * config.num_samples)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)



    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pt'))

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')

torch.save(train_loss_history, os.path.join(config.save_dir, 'train_loss'))
torch.save(valid_loss_history, os.path.join(config.save_dir, 'valid_loss'))
