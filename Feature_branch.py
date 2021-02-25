
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os
from glob import glob
import imageio


# In[2]:

weight_save_root = "logs"
if not os.path.exists(weight_save_root):
    os.mkdir(weight_save_root)

# In[3]:


# Note: normalize mean and std are standardized for ImageNet
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
train_transform = transforms.Compose([
        transforms.Resize(size=(128,128)),
        transforms.ToTensor()
])

test_transform = transforms.Compose([
        transforms.Resize(size=(128,128)),
        transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    'Arch_train/',
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    'Arch_test/',
    transform=train_transform
)

print(train_dataset.class_to_idx)
print(test_dataset.class_to_idx)

batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
)


# In[4]:


for inputs,tars in train_loader:
    print(inputs[0].shape)
    plt.imshow(inputs[0].permute(1,2,0))
    print(tars[0])
    break
K=0
print(os.getcwd())
for i in os.listdir("Arch_test/"):
    K+=1
print(K)


# In[5]:


# Define the model
pre_model = models.vgg16(pretrained=True)
features=pre_model.classifier[0].in_features

# pre_model
lower_branch=pre_model.features[:17] ### 16,16-- 2
middle_branch=pre_model.features[:24] ### 8,8-- 4
higher_branch=pre_model.features ### 4,4-- 8
class_head=pre_model.classifier
for param in lower_branch.parameters():
  param.requires_grad = False
for param in middle_branch.parameters():
  param.requires_grad = False
for param in higher_branch.parameters():
  param.requires_grad = False
class_head


# In[6]:


conv_1_dim = int(sys.argv[1])
conv_m_dim = int(sys.argv[2])
conv_h_dim = int(sys.argv[3])
class Denser_Net(nn.Module):
  def __init__(self,K,h_b,m_b,l_b):
    super(Denser_Net,self).__init__()
    ### add batch norm
    self.conv_1=nn.Sequential(
        lower_branch,
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(256, conv_1_dim, kernel_size=1)### 384  # 64
    )
    self.conv_m=nn.Sequential(
        middle_branch,
        nn.UpsamplingNearest2d(scale_factor=4),
        nn.Conv2d(512, conv_m_dim, kernel_size=1) ### 512  # 64
)
    self.conv_h=nn.Sequential(
        higher_branch,
        nn.UpsamplingNearest2d(scale_factor=8),
        nn.Conv2d(512, conv_h_dim, kernel_size=1)  ### 256  # 64
)
    self.fc=nn.Sequential(
          nn.Linear(in_features=1024*(conv_1_dim+conv_m_dim+conv_h_dim), out_features=4096, bias=True),  ### 196608
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=4096, bias=True),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False),
          nn.Linear(in_features=4096, out_features=1000, bias=True)   ### 1000
)
  def forward(self, x):
    # print(x.shape)
    h_x = self.conv_1(x)
    m_x = self.conv_m(x)
    l_x = self.conv_h(x)
    
    # print(h_x.shape)
    # ## concat features
    out = torch.cat((l_x,m_x,h_x), 1)
    out=F.relu(out)
    out = out.view(out.size(0), -1)# flatten
    # print(out.shape)
    out=self.fc(out)
    return out



# In[7]:


denser_net = Denser_Net(lower_branch,middle_branch,higher_branch,K)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
denser_net.to(device)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# denser_net = denser_net.cuda()
# device_ids = [0, 1, 2, 3]
# denser_net = torch.nn.DataParallel(denser_net, device_ids=device_ids)

# torch.cuda.get_device_name(1),torch.cuda.device_count(),torch.cuda.current_device()


# In[8]:


# from torchsummary import summary
# summary(denser_net, (3, 128, 128))


# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, denser_net.parameters()), lr=0.0001,     betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)


# In[10]:


def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
  train_losses = np.zeros(epochs)
  test_losses = np.zeros(epochs)
    
  best_test_acc = 0

  for it in range(epochs):
    t0 = datetime.now()
    train_loss = []
    model.train()
    for inputs, targets in train_loader:
      # move data to GPU
      inputs, targets = inputs.to(device), targets.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets)
        
      # Backward and optimize
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

    # Get train loss and test loss
    train_loss = np.mean(train_loss) # a little misleading
    
    model.eval()
    test_loss = []
    n_test_correct = 0.
    n_test_total = 0.
    n_train_correct = 0.
    n_train_total = 0.
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      loss = criterion(outputs, targets)
      test_loss.append(loss.item())
      n_test_correct += (predictions == targets).sum().item()
      n_test_total+= targets.shape[0]
    
    test_acc = n_test_correct / n_test_total
    test_loss = np.mean(test_loss)
    
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      _, predictions = torch.max(outputs, 1)
      n_train_correct += (predictions == targets).sum().item()
      n_train_total+= targets.shape[0]
    # Save losses
    
    train_acc = n_train_correct / n_train_total

    
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    
    dt = datetime.now() - t0
    # print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc:{train_acc:.4f},\
    # Test Loss: {test_loss:.4f}, Test Acc:{test_acc:.4f}')
    print('Epoch %d/%d, Train Loss: %f, Train Acc:%f,    Test Loss: %f, Test Acc:%f' % (it+1, epochs, train_loss, train_acc, test_loss, test_acc))
    
    # torch.save(model.state_dict(), os.path.join(weight_save_path, "model_%d.pth" % (it+1)))
    if test_acc > best_test_acc:
      # torch.save(model.module.conv_1.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_1_dim-%d.pth" % conv_1_dim))
      # torch.save(model.module.conv_m.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_m_dim-%d.pth" % conv_m_dim))
      # torch.save(model.module.conv_h.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_h_dim-%d.pth" % conv_h_dim))
      torch.save(model.conv_1.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_1_dim-%d.pth" % conv_1_dim))
      torch.save(model.conv_m.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_m_dim-%d.pth" % conv_m_dim))
      torch.save(model.conv_h.state_dict(), os.path.join(weight_save_root, "densernet_vgg16_conv_h_dim-%d.pth" % conv_h_dim))
      best_test_acc = test_acc
  
  return train_losses, test_losses


# In[11]:


train_losses, test_losses = batch_gd(
    denser_net,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=100,
)


# In[12]:


c=0
for i,t in train_loader:
    plt.imshow(i[3].permute(1,2,0))
    outs=cnn_model(i[3].unsqueeze(0).to(device))
    _,pred=torch.max(outs,1)
    print(pred == t[3])
    plt.title(f'Pred:{pred.cpu().numpy()}---Label:{t[3]}')
    break
    


# In[ ]:


plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()


# In[ ]:


torch.save(cnn_model.state_dict(),"trained_arc.pt")

