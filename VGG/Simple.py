import torch as tc
import torchvision as tv
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

batch_sz = 128
nChannel = 1
epoch_sz = 100


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#dataset = datasets.ImageFolder('D:/Deep/PyCharm/GAN_PyTorch/DCGAN/img_align_celeba',trans)
dataset = datasets.MNIST(root='./MNIST_Data', train=True, download=True, transform=trans)
#dataset = datasets.CIFAR10(root='./CIFAR10_Data', train=True, download=True, transform=trans)
dataloader = tc.utils.data.DataLoader(dataset=dataset, batch_size= batch_sz, shuffle= True)

def img_range(x):
    out = (x+1)/2
    out = out.clamp(0, 1)
    return out

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(nChannel, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 2, 1)

        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 2, 1)

        self.bn2 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn10 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*2*2, 1024)
        self.fc2 = nn.Linear(1024, 10)


    def forward(self, input, batch):
        x = F.relu(self.bn2(self.conv2(self.conv1(input))))
        x = F.relu(self.bn4(self.conv4(self.conv3(x))))
        x = F.relu(self.bn7(self.conv7(self.conv6(self.conv5(x)))))
        x = F.relu(self.bn10(self.conv10(self.conv9(self.conv8(x)))))
        x = x.reshape(batch, -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = F.softmax(x)

        return x


vgg = VGGNet()

device =tc.device('cuda' if tc.cuda.is_available() else 'cpu')
vgg.to(device)

loss_func = tc.nn.MSELoss()
opt = tc.optim.Adam(vgg.parameters(), lr=0.01 ) #0.999

print("Processing Start")
for ep in range(epoch_sz):
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        mini_batch = images.size()[0]

        vgg_result = vgg(images, mini_batch)

        loss = F.nll_loss(vgg_result, labels)

        vgg.zero_grad()
        loss.backward()
        opt.step()

        if step%200 ==0:
            print('epoch {}/{}, step {}, loss {}, label {}'.format(ep, epoch_sz, step, loss.item(), labels[0]))
            pred = vgg_result.max(1, keepdim=True)[1]
            print('pred {}'.format(pred))

            # if ep + 1 == 1:
    #     out = images
    #     out = image_range(out)
    #     save_image(out, os.path.join(result_path, 'real_img.png'))
    #
    # out = fake_images
    # out = image_range(out)
    # save_image(out, os.path.join(result_path, 'fake_img {}.png'.format(ep)))

    tc.save(vgg.state_dict(), 'vgg.ckpt')