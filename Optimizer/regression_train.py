'''
All of the details are available at:
https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/
'''

import  torch, os
import  numpy as np
import  argparse
import time

from meta_regression import Meta
from tensorboardX import SummaryWriter
from MLP import MLP

def generate_sinusoid_batch(args, input_idx=None):
    # input_idx is used during qualitative testing --the number of examples used for the grad update
    amp = np.random.uniform(args.amp_range[0], args.amp_range[1], [args.task_num])
    phase = np.random.uniform(args.phase_range[0], args.phase_range[1], [args.task_num])
    outputs = np.zeros([args.task_num, args.k_spt + args.k_qry, args.dim_output])
    init_inputs = np.zeros([args.task_num, args.k_spt + args.k_qry, args.dim_input])
    for func in range(args.task_num):
        init_inputs[func] = np.random.uniform(args.input_range[0], args.input_range[1], [args.k_spt + args.k_qry, 1])
        if input_idx is not None:
            init_inputs[:,input_idx:,0] = np.linspace(args.input_range[0], args.input_range[1], num=args.k_spt + args.k_qry-input_idx, retstep=False)
        outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
    return init_inputs, outputs, amp, phase

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))

    if args.network == '2hilayer':
        config = [
            ('linear', [40, 1]),
            ('relu', [True]),
            ('linear', [40, 40]),
            ('relu', [True]),
            ('linear', [1, 40]),
        ]
    elif args.network == '1hilayer':
        config = [
            ('linear', [200, 1]),
            ('relu', [True]),
            ('linear', [1, 200]),
        ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num_list = list(map(lambda x: np.prod(x.shape), tmp))
    num = sum(num_list)
    print('Total trainable tensors:', num)
    
    # Generate the whole gaussian vector for the whole epoch
    normal_vectors_matrix = []
    normal_vectors_tensor_matrix = []
    for i in range(args.task_num):
        nv3 = []
        nvt3 = []
        for j in range(args.update_step):
            nv2 = []
            nvt2 = []
            for m in range(args.gaus_num):
                nv1 = []
                nvt1 = []
                for k in maml.parameters():
                    if k.requires_grad:
                        normal_vector = torch.randn(list(k.size())).cuda()
                        nv1.append(normal_vector)
                nvt1 = torch.cat(list(torch.reshape(t, (-1,)) for t in nv1))
                nv2.append(nv1)
                nvt2.append(nvt1)
            nv3.append(nv2)
            nvt3.append(nvt2)
        normal_vectors_matrix.append(nv3)
        normal_vectors_tensor_matrix.append(nvt3)
     

    for step in range(args.epoch):

        x_batch, y_batch, amp, phase = generate_sinusoid_batch(args)
        x_spt = x_batch[:, :args.k_spt, :]
        y_spt = y_batch[:, :args.k_spt, :]
        x_qry = x_batch[:, args.k_spt:, :]
        y_qry = y_batch[:, args.k_spt:, :]
       
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).float().to(device), torch.from_numpy(y_spt).float().to(device), \
                                     torch.from_numpy(x_qry).float().to(device), torch.from_numpy(y_qry).float().to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        loss = maml(x_spt, y_spt, x_qry, y_qry, normal_vectors_matrix, normal_vectors_tensor_matrix, step)
        writer.add_scalar('losses', loss[-1], step)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3000)
    argparser.add_argument('--n_way', type=int, help='n way, the number of classes', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--k_spt', type=int, default=10, help='K-shot, number of examples used for inner gradient updates')
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--gaus_num', type=int, default=10, help='the number of gaussian vectors used')
    argparser.add_argument('--approx_delta', type=float, default=1e-2, help='parameter used in approximation')
    argparser.add_argument('--amp_range', default=[0.1, 5.0])
    argparser.add_argument('--phase_range', default=[0, 3.1415926])
    argparser.add_argument('--input_range', default=[-5.0, 5.0])
    argparser.add_argument('--dim_input', type=int, default=1)
    argparser.add_argument('--dim_output', type=int, default=1)
    argparser.add_argument('--task_num', type=int, default=25, help='task_num, number of tasks sampled per meta-updates')
    argparser.add_argument('--approx_method', default='maml', choices=['maml', 'ggs', 'first_order', 'zero_order'])
    argparser.add_argument('--network', default='2hilayer', choices=['2hilayer', '1hilayer'])
    argparser.add_argument('--output_folder', default='', help='save the tfevent file')
    args = argparser.parse_args()
    args.output_folder += args.approx_method + '_' + args.network + '_innerstep_' + str(args.update_step) + '_metalr_'+str(args.meta_lr)+'_updatelr_'
    args.output_folder += (str(args.update_lr)+'_gaus_'+str(args.gaus_num)+'_delta_'+str(args.approx_delta))
    main(args)
