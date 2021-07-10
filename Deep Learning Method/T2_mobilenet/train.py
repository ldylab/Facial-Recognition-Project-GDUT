import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model_v3 import mobilenet_v3_large
import pandas as pd
import time
from my_dataset import MyDataSet
from utils import read_split_data


def main():
    image_root = lambda x: "/Dataset/all/dataset_expression/" if x == "E" else "/Dataset/all/dataset_gender/"
    image_path = os.path.dirname(os.getcwd()) + image_root(input("Gender(G) or Expression(E) = "))  # 表情识别

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    epoch_times = int(input("Epoch Times = "))
    csv_note = input("Notes for csv = ")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(image_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = 16

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 到这里
    # create model
    net = mobilenet_v3_large(num_classes=2)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v3_large.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights 权重的修改
    # for param in net.features.parameters():
    #    param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    train_csv = pd.DataFrame(columns=['Loss', 'Acc'])
    epochs = epoch_times
    best_acc = 0.0
    save_path = './MobileNetV3_new.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        train_csv = train_csv.append({'Loss': running_loss / step, 'Acc': val_accurate}, ignore_index=True)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    csv_path = os.getcwd() + "/DataSave/CSV/"
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    save_name = csv_path + now + csv_note + r"--MobileNetV3.csv"
    train_csv.to_csv(save_name, mode='a', header=True, index=True)
    print('Finished Training')


if __name__ == '__main__':
    main()
