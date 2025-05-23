# 运行时间: 16:48
import os
import numpy as np
import torch
import dgl
import torch.optim as optim
import scipy.io as scio
from model import *
from utils import *
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.cuda)
    args.device = device
    # dataset_path = args.data_path+args.dataset+'.dgl'
    dataset_path = r'D:\FraudDetector\FDetector\data\toy_dataset.dgl'
    model_path = args.result_path+args.dataset+'_model.pt'


    dataset1_path = '../data/'
    toy_path = dataset1_path+'toy_dataset.mat'
    toy = scio.loadmat(toy_path)
    lbs = toy['label'][0]
    label = torch.from_numpy(lbs)
    labels = torch.from_numpy(lbs).long()
    num_samples = labels.size(0)
    one_hot_labels = torch.zeros(num_samples, args.num_classes)
    one_hot_labels[torch.arange(num_samples), labels] = 1

    results = {'F1-macro':[],'AUC':[],'G-Mean':[],'recall':[]}
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    '''
    # load dataset and normalize feature
    '''
    dataset = dgl.load_graphs(dataset_path)[0][0]
    features = dataset.ndata['feature'].numpy()
    features = normalize(features)
    dataset.ndata['feature'] = torch.from_numpy(features).float()
    dataset = dataset.to(device)

    '''
    # train model
    '''
    print('Start training model...')
    model = DualHFDNet(args, dataset, features, one_hot_labels)
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stop = EarlyStop(args.early_stop)

    for e in range(args.epoch):
    #for e in range(1):
        model.train()
        loss = model.loss(dataset, features, one_hot_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            '''
            # valid
            '''
            model.eval()
            valid_mask = dataset.ndata['valid_mask'].bool()
            valid_labels = dataset.ndata['label'][valid_mask].cpu().numpy()
            valid_logits = model(dataset)[valid_mask]
            valid_preds = valid_logits.argmax(1).cpu().numpy()
            f1_macro, auc, gmean, recall = evaluate(valid_labels, valid_logits)

            if args.log:
                print(f'{e}: Best Epoch:{early_stop.best_epoch}, Best valid AUC:{early_stop.best_eval}, Loss:{loss.item()}, Current valid: Recall:{recall}, F1_macro:{f1_macro}, G-Mean:{gmean}, AUC:{auc}')
            do_store, do_stop = early_stop.step(auc, e)
            if do_store:
                torch.save(model, model_path)
            if do_stop:
                break
    print('End training')
    '''
    # test model
    '''
    print('Test model...')
    model = torch.load(model_path)
    with torch.no_grad():
        model.eval()
        test_mask = dataset.ndata['test_mask'].bool()
        test_labels = dataset.ndata['label'][test_mask]
        test_labels = test_labels.cpu().numpy()
        logits = model(dataset)[test_mask]
        logits = logits.cpu()
        test_result_path = args.result_path+args.dataset
        f1_macro, auc, gmean, recall = evaluate(test_labels, logits, test_result_path)
        results['F1-macro'].append(f1_macro)
        results['AUC'].append(auc)
        results['G-Mean'].append(gmean)
        results['recall'].append(recall)
        print(f'Test: F1-macro:{f1_macro}, AUC:{auc}, G-Mean:{gmean}, Recall:{recall}')
