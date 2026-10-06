import os

from configs import Config
from sign_dataset import Sign_Dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from tgcn_model import GCN_muti_att


def test(model, test_loader):
    # set model as testing mode
    model.eval()

    val_loss = []
    all_y = []
    all_y_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print('starting batch: {}'.format(batch_idx))
            # distribute data to device
            X, y, video_ids = data
            X, y = X.to(next(model.parameters()).device), y.to(next(model.parameters()).device).view(-1, )

            all_output = []

            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i + 1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # log down incorrectly labelled instances
    incorrect_indices = torch.nonzero(all_y - all_y_pred).squeeze().data
    incorrect_video_ids = [(vid, int(all_y_pred[i].data)) for i, vid in enumerate(all_video_ids) if
                           i in incorrect_indices]

    all_y = all_y.cpu().data.numpy()
    all_y_pred = all_y_pred.cpu().data.numpy()

    # top-k accuracy
    # top1acc = accuracy_score(all_y, all_y_pred)
    # top3acc = compute_top_n_accuracy(all_y, all_pool_out, 3)
    # top5acc = compute_top_n_accuracy(all_y, all_pool_out, 5)
    # top10acc = compute_top_n_accuracy(all_y, all_pool_out, 10)
    # top30acc = compute_top_n_accuracy(all_y, all_pool_out, 30)

    print("Prediction:", all_y_pred)
    print("Actual:", all_y)

    # show information
    # print('\nVal. set ({:d} samples): top-1 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top1acc))
    # print('\nVal. set ({:d} samples): top-3 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top3acc))
    # print('\nVal. set ({:d} samples): top-5 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top5acc))
    # print('\nVal. set ({:d} samples): top-10 Accuracy: {:.2f}%\n'.format(len(all_y), 100 * top10acc))


def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]


if __name__ == '__main__':

    root = r'C:\Users\hi\OneDrive\Documents\ACM\Cue\WLASL'
    trained_on = 'asl100'
    #testing bc the training on model isnt asl 100 and im js checking
    
    hf_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'checkpoints',
    'asl100'
)

    config_file = os.path.join(hf_root, 'config.ini')
    checkpoint_file = os.path.join(hf_root, 'pytorch_model.bin')

    configs = Config(config_file)
    

    split_file = os.path.join(root, 'data', 'splits', '{}.json'.format(trained_on))
    pose_data_root = os.path.join(
    root, 'data', 'pose_per_individual_videos', 'pose_per_individual_videos')
    config_file = os.path.join(hf_root, 'config.ini')
    checkpoint_file = os.path.join(hf_root, 'pytorch_model.bin')

    configs = Config(config_file)

    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    batch_size = configs.batch_size

    dataset = Sign_Dataset(
        index_file_path=split_file,
        split='test',
        pose_root=pose_data_root,
        img_transforms=None,
        video_transforms=None,
        num_samples=num_samples,
        sample_strategy='k_copies',
        test_index_file=split_file
    )

    from torch.utils.data import Subset
    dataset = Subset(dataset, [0])

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCN_muti_att(
        input_feature=num_samples * 2,
        hidden_feature=hidden_size,
        num_class=int(trained_on[3:]),
        p_dropout=drop_p,
        num_stage=num_stages
    ).to(device)

    print('Loading model...')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print('Finish loading model!')
    print('Using device:', device)

    test(model, data_loader)
