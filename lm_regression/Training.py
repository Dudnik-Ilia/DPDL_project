import numpy as np
from sklearn.model_selection import train_test_split
import time
import copy
import json
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision

from DataAugmentation import ToPILImage, Normalize, Rescale, RandomCrop, ToTensor, ColorJitter
from FaceLandmarksDataset import FaceLandmarksDataset


def train_model(model, criterion, optimizer, scheduler, writer, dataloaders, num_epochs=25):
    since = time.time()
    
    # TODO: EXPERIMENTAL:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000
    last_activation = torch.nn.Identity() # not a great way to program this
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            epoch_iter = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample['image'].float().to(device)
                labels = sample['landmarks'].float().to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = last_activation(outputs) #TODO: Fixme
                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_iter += 1
                running_loss += loss.item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/epoch_iter
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
            writer.flush()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # visualize_model(model)  # you can activate this line if you want to see an example of how the predictions progress
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main():

    writer = SummaryWriter(log_dir='./output/logs/')

    # Load data from disk
    for p in ['./data/image_{}.json'.format(i) for i in [1]]:
        with open(p, 'r') as f:
            json_info = json.load(f)

    # tranfer the relevant information into a pandas frame:
    my_list = []
    for id in range(1, 101):

        with open('./data/image_{}.json'.format(id), 'r') as f:
            json_info = json.load(f)

        for entry in json_info['shapes']:
            if entry['label'] == 'eye_right':
                eye_r = np.asarray(entry['points'][0]).astype('float').round(1)
            if entry['label'] == 'eye_left':
                eye_l = np.asarray(entry['points'][0]).astype('float').round(1)

        my_list.append([id, './data/image_{}.jpg'.format(id), eye_r[0], eye_r[1], eye_l[0], eye_l[1]])

    landmarks_frame = pd.DataFrame(my_list, columns = ['id', 'image_path', 'eye right x', 'eye right y', 'eye left x', 'eye left y'])
    
    transform_objs = transforms.Compose([ToPILImage(), RandomCrop(0.6), Rescale(244), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)])

    transformed_dataset = FaceLandmarksDataset(data_frame=landmarks_frame, transform=transform_objs, output_shape=(-1, 2))

    train_val, test = train_test_split(landmarks_frame, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    transform_objs = transforms.Compose([ToPILImage(), RandomCrop(0.6), Rescale(128), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), ToTensor()])
    transform_val = transforms.Compose([ToPILImage(), Rescale(128), ToTensor()])  # no data augmentation for the validation set (can make sense)

    transformed_dataset = FaceLandmarksDataset(data_frame=train, transform=transform_objs)
    transformed_dataset_val = FaceLandmarksDataset(data_frame=val, transform=transform_val)
    transformed_dataset_test = FaceLandmarksDataset(data_frame=test, transform=transform_val)


    dataloader = DataLoader(transformed_dataset, batch_size=10, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(transformed_dataset_val, batch_size=10, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(transformed_dataset_test, batch_size=1, shuffle=False, num_workers=0)

    dataloaders = {'train': dataloader, 'val': dataloader_val}

    model_conv = torchvision.models.resnet34(pretrained=True)
    for param in model_conv.parameters(): # Parameters of newly constructed modules have requires_grad=True by default, if you want to train fewer parameters, you can set this to false
        param.requires_grad = True


    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, 4)
    
    criterion = torch.nn.MSELoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
    
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, writer, dataloaders,
                         num_epochs=30)
    torch.save(model_conv.state_dict(), './output/checkpoints/checkpoint.pth')
    
if __name__ == '__main__':
	main()
