import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
import datetime
import random
import time
import warnings
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import logging
from logger import setup_logger

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')


def test(loader, base_net, classifier1, classifier2, test_10crop=True, class_avg=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                outputs_c1 = []
                for j in range(10):
                    feature = base_net(inputs[j])
                    predict_out1 = classifier1(feature)
                    predict_out2 = classifier2(feature)
                    predict_out = predict_out1 + predict_out2
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                feature = base_net(inputs)
                outputs1 = classifier1(feature)
                outputs2 = classifier2(feature)
                outputs = outputs1 + outputs2
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if class_avg:
        # for VisDA class-wise acc
        dset_classes = ['aeroplane',
                        'bicycle',
                        'bus',
                        'car',
                        'horse',
                        'knife',
                        'motorcycle',
                        'person',
                        'plant',
                        'skateboard',
                        'train',
                        'truck']
        classes_acc = {}
        for i in dset_classes:
            classes_acc[i] = []
            classes_acc[i].append(0)
            classes_acc[i].append(0)
        for i in range(len(all_label)):
            key_label = dset_classes[all_label.long()[i].item()]
            key_pred = dset_classes[predict.long()[i].item()]
            classes_acc[key_label][1] += 1
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1
        avg_for_class = []
        for i in dset_classes:
            print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                                   100. * classes_acc[i][0] / classes_acc[i][1]))
            avg_for_class.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
        class_acc = np.average(avg_for_class)
        print('\t class average:', class_acc)
        return accuracy, class_acc

    return accuracy


def train(config):
    logger = logging.getLogger("CAF-D.trainer")

    # set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    # prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(),
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(),
                                       transform=prep_dict["test"][i])
                             for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset,
                                               batch_size=test_bs,
                                               shuffle=False, num_workers=4)
                                    for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"],
                                          batch_size=test_bs,
                                          shuffle=False,
                                          num_workers=4)

    class_num = config["network"]["params"]["class_num"]
    K = config["K"]

    # set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    # set classifier
    classifier1 = network.Classifier(num_classes=class_num)
    network.init_weights(classifier1)
    classifier1 = classifier1.cuda()

    classifier2 = network.Classifier(num_classes=class_num)
    network.init_weights(classifier2)
    classifier2 = classifier2.cuda()

    # set optimizer
    optimizer_config = config["optimizer"]

    # for generator
    optimizer_g = optimizer_config["type"](base_network.get_parameters(), \
                                           **(optimizer_config["optim_params"]))
    param_g_lr = []
    for param_group in optimizer_g.param_groups:
        param_g_lr.append(param_group["lr"])
    # for classifier
    optimizer_f = optimizer_config["type"](classifier1.get_parameters() + classifier2.get_parameters(), \
                                           **(optimizer_config["optim_params"]))
    param_f_lr = []
    for param_group in optimizer_f.param_groups:
        param_f_lr.append(param_group["lr"])

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    best_model = None
    start = time.time()
    for i in tqdm(range(config["num_iterations"])):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            classifier1.train(False)
            classifier2.train(False)
            if config["dataset"] == "visda2017":
                temp_acc, avg_for_class = test(dset_loaders, base_network, classifier1, classifier2,
                                               test_10crop=prep_config["test_10crop"], class_avg=True)
                if avg_for_class > best_acc:
                    best_acc = avg_for_class
                    best_model = nn.Sequential(base_network, classifier1, classifier2)

                logger.info("iter: {:05d}, precision: {:.5f},\tbest_class_acc: {:.5f}\tall_acc:{:.5f}\ttime: {:.2f}".format(
                    i,
                    avg_for_class,
                    best_acc,
                    temp_acc,
                    time.time() - start))
            else:
                temp_acc = test(dset_loaders, base_network, classifier1, classifier2,
                                test_10crop=prep_config["test_10crop"], class_avg=False)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_model = nn.Sequential(base_network, classifier1, classifier2)

                logger.info("iter: {:05d}, precision: {:.5f},\tbest_acc:{:.5f}\ttime: {:.2f}".format(
                    i,
                    temp_acc,
                    best_acc,
                    time.time() - start))

        # train one iter
        base_network.train(True)
        classifier1.train(True)
        classifier2.train(True)
        optimizer_g = lr_scheduler(optimizer_g, i, **schedule_param)
        optimizer_f = lr_scheduler(optimizer_f, i, **schedule_param)
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        # step A train all networks to minimize the loss
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        features_source = base_network(inputs_source)
        outputs_source1 = classifier1(features_source)
        outputs_source2 = classifier2(features_source)
        features_target = base_network(inputs_target)
        outputs_target1 = classifier1(features_target)
        outputs_target2 = classifier2(features_target)

        transfer_loss = loss.sliced_wasserstein_distance(features_source, features_target, embed_dim=256)
        classifier_loss = (nn.CrossEntropyLoss()(outputs_source1, labels_source) + nn.CrossEntropyLoss()(
            outputs_source2, labels_source)) * 0.5
        outputs_target1_logit = nn.Softmax(dim=1)(outputs_target1)
        outputs_target2_logit = nn.Softmax(dim=1)(outputs_target2)
        entropy_loss = loss.Entropy_loss(outputs_target1_logit) + loss.Entropy_loss(outputs_target2_logit)

        total_loss = classifier_loss + transfer_loss + 0.01 * entropy_loss

        total_loss.backward()
        optimizer_f.step()
        optimizer_g.step()

        # step B train F1,F2 to max the discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        features_source = base_network(inputs_source)
        outputs_source1 = classifier1(features_source)
        outputs_source2 = classifier2(features_source)
        features_target = base_network(inputs_target)
        outputs_target1 = classifier1(features_target)
        outputs_target2 = classifier2(features_target)

        classifier_loss = (nn.CrossEntropyLoss()(outputs_source1, labels_source) + nn.CrossEntropyLoss()(
            outputs_source2, labels_source)) * 0.5
        distance_loss = loss.sliced_wasserstein_distance(outputs_target1, outputs_target2, embed_dim=class_num)
        outputs_target1_logit = nn.Softmax(dim=1)(outputs_target1)
        outputs_target2_logit = nn.Softmax(dim=1)(outputs_target2)
        entropy_loss = loss.Entropy_loss(outputs_target1_logit) + loss.Entropy_loss(outputs_target2_logit)

        total_loss = classifier_loss - distance_loss + 0.01 * entropy_loss
        total_loss.backward()
        optimizer_f.step()

        # step C train G to min the discrepancy, to max the transfer_loss
        for k in range(K):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            features_source = base_network(inputs_source)
            features_target = base_network(inputs_target)
            outputs_target1 = classifier1(features_target)
            outputs_target2 = classifier2(features_target)

            transfer_loss = loss.sliced_wasserstein_distance(features_source, features_target, embed_dim=256)
            distance_loss = loss.sliced_wasserstein_distance(outputs_target1, outputs_target2, embed_dim=class_num)
            outputs_target1_logit = nn.Softmax(dim=1)(outputs_target1)
            outputs_target2_logit = nn.Softmax(dim=1)(outputs_target2)
            entropy_loss = loss.Entropy_loss(outputs_target1_logit) + loss.Entropy_loss(outputs_target2_logit)

            total_loss = distance_loss + transfer_loss + 0.01 * entropy_loss
            total_loss.backward()
            optimizer_g.step()

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth"))
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collaborative Alignment Framework Distance matching')
    parser.add_argument('--method', type=str, default='CAF-A')
    parser.add_argument('--net', type=str, default='ResNet101',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])
    parser.add_argument('--dset', type=str, default='domainnet',
                        choices=['office31', 'imageCLEF', 'visda2017', 'domainnet'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/list/domainnet/clipart_train.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/list/domainnet/infograph_train.txt',
                        help="The target dataset path list")
    parser.add_argument('--t_test_path', type=str, default='data/list/domainnet/infograph_test.txt',
                        help="The target dataset path list")
    parser.add_argument('--max_interval', type=int, default=30001, help="max iteration for training")
    parser.add_argument('--test_interval', type=int, default=1000, help="interval of two continuous test phase")
    parser.add_argument('--output_dir', type=str, default='result/CAF-D',
                        help="output directory of our model (in result directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--SEED', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)

    # train config
    config = {}
    config['method'] = args.method
    config["num_iterations"] = args.max_interval
    config["K"] = args.K
    config["test_interval"] = args.test_interval
    config["output_for_test"] = True
    config["output_path"] = os.path.join(args.output_dir, 'CAF-D', args.dset + f'_seed-{args.SEED}')
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])

    try:
        source_name = args.s_dset_path.split('/')[-1].split('.')[0]
        target_name = args.t_test_path.split('/')[-1].split('.')[0]
    except:
        source_name = '0'
        target_name = '1'
    file_name = f'{source_name}_2_{target_name}.txt'
    logger = setup_logger("CAF-D", config["output_path"], 0, filename=file_name)

    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": True,
                      'params': {"resize_size": 256, "crop_size": 224}}
    config["loss"] = {"trade_off": 1.0}
    config["network"] = {"name": network.ResNetFc,
                         "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                    "new_cls": True}
                         }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD,
                           "optim_params": {'lr': args.lr, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv",
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}
                           }

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36},
                      "target": {"list_path": args.t_dset_path, "batch_size": 36},
                      "test": {"list_path": args.t_dset_path, "batch_size": 4}}

    if config["dataset"] == "office31":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "imageCLEF":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda2017":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "domainnet":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 345
        config["num_iterations"] = 30001
        config["test_interval"] = 5000
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    logger.info(str(config))
    train(config)
