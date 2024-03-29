from torch_geometric.data import DataLoader
from model import *
import random
from tqdm import tqdm
from rotation import augmentation
from torch import optim
import numpy as np
from arg_helper import *

args = parse_arguments()
config = get_config(args.config_file)

seed = config.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if config.data == 'mind':
    data = torch.load('mind')
    random.shuffle(data)
    train_set = data[0:71]
    valid_set = data[71:81]
    test_set = data[81:]
elif config.data == 'sphere6':
    data = torch.load('sphere6')
    random.shuffle(data)
    train_set = data[0:71]
    valid_set = data[71:81]
    test_set = data[81:]
elif config.data == 'sphere5':
    data = torch.load('sphere5')
    random.shuffle(data)
    train_set = data[0:71]
    valid_set = data[71:81]
    test_set = data[81:]
elif config.data == 'sulc':
    data = torch.load('sulc')
    random.shuffle(data)
    train_set = data[0:25]
    valid_set = data[25:29]
    test_set = data[29:]

train_loader = DataLoader(train_set, batch_size = 1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = 1)
test_loader = DataLoader(test_set, batch_size = 1)

device = config.device
#
if config.model == 'edge':
    model = EdgeUnet(config)
elif config.model == 'trans':
    model = TransUnet(config)
# model = PaTransUnet(in_channels=7, hidden_channels=[32,64,128,256], out_channels=32,
#                  num_classes=32, pool_ratios = 0.5, sum_res=False)

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

train_loss_history=[]
valid_loss_history=[]
best_loss = 10e10
for epoch in tqdm(range(500)):
    model.train()
    train_loss = 0
    valid_loss = 0

    for data in train_loader:
        data = data
        optimizer.zero_grad()
        out = model(data)

        y = data.y.to('cuda:3')
        weight = torch.bincount(y) / len(y)
        weight = 1 / weight
        weight = weight / weight.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data
            out = model(data)

            y = data.y.to('cuda:3')
            weight = torch.bincount(y) / len(y)
            weight = 1 / weight
            weight = weight / weight.sum()
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

            loss = criterion(out, y)
            valid_loss += loss

    train_loss = train_loss / len(train_set)
    valid_loss = valid_loss / len(valid_set)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model'))

    print(f'Epoch: {epoch:03d} Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}')

torch.save(train_loss_history, os.path.join(config.save_dir, 'train_loss'))
torch.save(valid_loss_history, os.path.join(config.save_dir, 'valid_loss'))

