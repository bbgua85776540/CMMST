import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from vrdataset_only_traj import Vrdataset
from evaldataset import Evaldataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import pickle
import sys
sys.path.append('/media/kemove/1A226EEF226ECEF7/work/pytorch_workplace/CMMST')
# BATCH_SIZE = 32
SEQ_LEN = 30
TAG_SIZE = 144
CUDA = True



class LSTMPredict(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, tag_size=TAG_SIZE
                 , use_cuda=CUDA, batch_size=12, look_ahead=25):
        super(LSTMPredict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tag_size = tag_size
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.look_ahead = look_ahead
        # self.in2lstm = nn.Linear(tag_size, input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.init_lstm()
        self.lstm2tag = nn.Linear(self.hidden_size, self.tag_size)
        # nn.init.normal(self.lstm2tag.weight)
        self.hidden = self.init_hidden()  # initial hidden state for LSTM network

    def init_lstm(self):
        for name, weights in self.lstm.named_parameters():
            if len(weights.data.shape) == 2:
                nn.init.kaiming_normal_(weights)
            if len(weights.data.shape) == 1:
                nn.init.normal_(weights)

    def init_hidden(self):
        hx = torch.nn.init.xavier_normal_(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        cx = torch.nn.init.xavier_normal_(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        if self.use_cuda:
            hx, cx = hx.cuda(), cx.cuda()
        hidden = (autograd.Variable(hx), autograd.Variable(cx))  # convert to Variable as late as possible
        return hidden

    def forward(self, orientations):
        # orientation_seq is a 3 dimensional tensor with shape [batch_size, seq_len, tag_size]
        # lstm_in is a 2 dimensional tensor with shape [seq_len, input_size]
        # inputs is a 3 dimensional tensor with shape [batch_size, seq_len, tag_size]
        lstm_out, self.hidden = self.lstm(orientations, self.hidden)
        # print('lstm_out.shape=', lstm_out.size())    # lstm_out.shape= torch.Size([1, 15, 256])
        # tag_scores = self.lstm2tag(lstm_out.contiguous().view(-1, self.hidden_size))
        tag_scores = self.lstm2tag(lstm_out)
        # print('tag_scores.size=', tag_scores.size())    # tag_scores.size= torch.Size([12, 15, 144])
        out = tag_scores[:, -1, :]
        # print('out=', out.shape)
        return out


def train_model(model, learning_rate, data_loader, epoch=10):
    use_cuda = torch.cuda.is_available()
    print('cuda: ' + str(use_cuda))
    process_frame_nums = 50
    model.train()  # for cuda speed up
    if use_cuda:
        model = model.cuda()

    loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, nesterov=True)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=args.lr_decay)
    for poch in range(epoch):
        count = 0
        loss_avg = 0.0
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # learning_rate *= 0.2
        for headmaps, labels in data_loader:
            # inputs = train_data[i: i+30]
            # if count == count_max:
            #     break
            headmaps = headmaps.float()
            labels = labels.float()
            nseries = headmaps.shape[1] - process_frame_nums - 1
            loss_series = 0

            for i in range(nseries):
                model.zero_grad()
                model.hidden = model.init_hidden()
                start_index = i
                end_index = i + process_frame_nums
                headmaps_segment = headmaps[:, start_index:start_index + 25]
                labels_segment = labels[:, end_index]

                inputs, label = headmaps_segment.cuda(), labels_segment.cuda()

                output = model(inputs)

                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()
                loss_series += loss.cpu().detach()
            learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            # print('loss=', loss_series / nseries, ' learning_rate=', learning_rate)
            loss_avg += loss_series
            count += 1
            scheduler.step()
            # if count % 100 == 0:
            # print('epoch=', poch, 'count=', count, 'loss_avg=', loss_avg / count, ' learning_rate=', learning_rate)
        print('epoch=', poch, 'count=', count, 'loss_avg=', loss_avg / (count * nseries), ' learning_rate=',
              learning_rate)


def eval_model(model, data_loader):
    use_cuda = torch.cuda.is_available()
    print('cuda: ' + str(use_cuda))
    if use_cuda:
        model = model.cuda()
    loss_function = nn.MSELoss()
    count = 0
    loss_avg = 0.0
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # learning_rate *= 0.2
    for inputs, label, user_index, topic_index in data_loader:
        inputs = inputs.float()
        label = label.float()
        user_index = int(user_index)
        topic_index = int(topic_index[0])
        # print('user_index=', user_index)
        # print('topic_index=', topic_index)
        nseries = inputs.shape[1]
        loss_series = 0
        outputs = []
        for i in range(nseries):
            model.zero_grad()
            model.hidden = model.init_hidden()
            series = inputs[:, i, :]
            ilabel = label[:, i, :]
            if use_cuda:
                series, ilabel = series.cuda(), ilabel.cuda()
            # print('series.shape=', series.shape)   # series.shape= torch.Size([1, 15, 144])
            # print('label.shape=', label[:, i, :].shape) # label.shape= torch.Size([1, 144])
            output = model(series)
            # print('output.shape=', output.shape) # output.shape= torch.Size([1, 144])
            # ilabel = label[:, i, :]
            # ilabel = torch.squeeze(ilabel, dim=0)
            loss = loss_function(output, ilabel)
            loss_series += loss.cpu().detach()
            outputs.append(output.detach().cpu().reshape(args.n_row, args.n_colum))
        # print('loss=', loss_series / nseries, ' learning_rate=', learning_rate)
        loss_avg += loss_series
        count += 1
        pickle.dump(outputs, open(PATH_PRED + 'ds{}_topic{}_user{}'.format(args.dataset, topic_index, user_index), 'wb'))
        # if count % 100 == 0:
        # print('epoch=', poch, 'count=', count, 'loss_avg=', loss_avg / count, ' learning_rate=', learning_rate)
    print('eval=', 'count=', count, 'loss_avg=', loss_avg / (count * nseries))


def metric():
    pass


def try_hyper_para(hidden_size_list, num_layer_list, data_loader, epoc, count_max, batch_size):
    for hidden_size in hidden_size_list:
        for num_layers in num_layer_list:
            model = LSTMPredict(input_size=144, hidden_size=hidden_size, num_layers=num_layers, tag_size=144,
                                batch_size=batch_size)
            train_model(model, learning_rate=0.001, data_loader=data_loader, epoch=epoc)
            print("finished training")
            model_name = 'sgd-lstm-' + str(hidden_size) + '-' + str(num_layers) + '.pth'
            # loss_name = 'loss-' + str(hidden_size) + '-' + str(num_layers) + '.dat'
            torch.save(model, model_name)
            print('saved model: ' + model_name)
            # save_loss(losses, loss_name)
            # print('saved loss data: ' + loss_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run PARIMA algorithm and calculate Average QoE of a video for all users')

    parser.add_argument('-D', '--dataset', type=int, default=2, help='Dataset ID (1 or 2)')
    parser.add_argument('-T', '--topic', default=0, help='Topic in the particular Dataset (video name)')
    parser.add_argument('--fps', type=int, default=29, help='fps of the video')
    parser.add_argument('-O', '--offset', type=int, default=0,
                        help='Offset for the start of the video in seconds (when the data was logged in the dataset) [default: 0]')
    parser.add_argument('--fpsfrac', type=float, default=1.0,
                        help='Fraction with which fps is to be multiplied to change the chunk size [default: 1.0]')
    parser.add_argument('-Q', '--quality', default='360p',
                        help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')
    parser.add_argument('-B', '--batch_size', type=int, default=24, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='training optimizer weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='training optimizer momentum')
    parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every epoch')
    # parser.add_argument("--resume", type=str, default=None, help='resume model param path')
    parser.add_argument("--resume", type=str, default='/media/kemove/1A226EEF226ECEF7/work/pytorch_workplace/PARIMA-master/AVpredict/sgd-lstm-256-1.pth', help='resume model param path')
    parser.add_argument("--n_colum", type=int, default=16, help='colum num of tiles')
    parser.add_argument("--n_row", type=int, default=9, help='row num of tiles')
    parser.add_argument("--look_back", type=int, default=30, help='use frames history')
    parser.add_argument("--look_ahead", type=int, default=30, help='predict future n frames')

    args = parser.parse_args()

    PATH_ACT = '../../Viewport/ds{}/'.format(args.dataset)
    PATH_PRED = './head_prediction/ds{}/'.format(args.dataset)

    train_dataset = Vrdataset()
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True, drop_last=True)
    val_dataset = Evaldataset(dataset_flag=args.dataset, topic=args.topic,
                              look_ahead=args.look_ahead, look_back=args.look_back)
    valLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    hidden_size_list = [256, 512]
    num_layer_list = [1, 2, 3]
    # train
    try_hyper_para(hidden_size_list, num_layer_list, trainLoader, epoc=30, count_max=10, batch_size=args.batch_size)

    # eval
    # model = LSTMPredict(input_size=144, hidden_size=256, num_layers=1, tag_size=144, batch_size=1)
    # if args.resume is not None:
    #     state_dict = torch.load(args.resume).state_dict()
    #     model.load_state_dict(state_dict)
    #     print("Checkpoint {} loaded!".format(args.resume))
    # model.eval()
    # val_dataset = Evaldataset(dataset_flag=args.dataset, topic=args.topic)
    # valLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # with torch.no_grad():
    #     eval_model(model, valLoader)

