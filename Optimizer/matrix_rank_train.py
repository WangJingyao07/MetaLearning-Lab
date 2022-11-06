'''
All of the details are available at:
https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/
'''

import  torch, os
import  numpy as np
import  argparse

from meta_matrix_rank import Meta
from tensorboardX import SummaryWriter
from linear_matrix_rank import Linear

def generate_matfactor_batch(args):
    outputs = np.zeros([args.task_num, args.dim*2])
    variance = np.random.uniform(0, args.variance_amp, args.task_num)
    for func in range(args.task_num):
        outputs[func] = np.random.normal(loc=0.0, scale=variance[func], size=args.dim*2)
    return outputs

def main(args):

    torch.manual_seed(2722)
    torch.cuda.manual_seed_all(22672)
    np.random.seed(2267892)

    print(args)
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))

    # linear node
    config = args.dim
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    print(maml)
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
        y_batch = generate_matfactor_batch(args)
        y_spt = y_batch[:, :args.dim]
        y_qry = y_batch[:, args.dim:]
        y_spt, y_qry = torch.from_numpy(y_spt).float().to(device), torch.from_numpy(y_qry).float().to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        loss = maml(y_spt, y_qry, normal_vectors_matrix, normal_vectors_tensor_matrix, step)
        writer.add_scalar('losses', loss[-1], step)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=2000)
    argparser.add_argument('--n_way', type=int, help='n way, the number of classes', default=50)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=3e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--gaus_num', type=int, default=10, help='the number of gaussian vectors used')
    argparser.add_argument('--approx_delta', type=float, default=1e-2, help='parameter used in approximation')
    argparser.add_argument('--dim', type=int, default=20, help='the dimension of x vector')
    argparser.add_argument('--variance_amp', type=float, default=0.1)
    argparser.add_argument('--task_num', type=int, default=10, help='task_num, number of tasks sampled per meta-updates')
    argparser.add_argument('--approx_method', default='maml', choices=['maml', 'ggs', 'first_order', 'zero_order'])
    argparser.add_argument('--output_folder', default='', help='save the tfevent file')
    
    args = argparser.parse_args()
    args.output_folder += 'generate__'+ args.approx_method + '_innerstep_' + str(args.update_step) + '_metalr_'+str(args.meta_lr)+'_updatelr_'
    args.output_folder += (str(args.update_lr)+'_gaus_'+str(args.gaus_num)+'_delta_'+str(args.approx_delta))+'_variance_'+str(args.variance_amp)
    main(args)
