'''
All of the details are available at:
https://wangjingyao07.github.io/Awesome-Meta-Learning-Platform/
'''

import  torch
import  math
import  numpy as np

from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch import optim
from    MLP import MLP
from    copy import deepcopy, copy
from    learner import Learner


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()
        self.delta = 0.01
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.U = args.gaus_num
        self.delta = args.approx_delta
        self.init_scale = 1.0
        self.approx_method = args.approx_method

        self.config = config
        # copy the n_units from the parameter
        self.net = Learner(config, 1, 1)
        # initialize the layers of MLP
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def inner_update(self, grad_diff_matrix, normal_matrix, grad_final_tensor):
        # parameter update based on Hessian vector and grad vectors
        if len(grad_diff_matrix[0]) >= 1:
            grad_right = torch.unsqueeze(grad_final_tensor, 1)
            for j in range(self.update_step):
                H_u = torch.matmul(torch.stack(normal_matrix[self.update_step-j-1]), grad_right)
                H_u = torch.matmul(torch.t(torch.stack(grad_diff_matrix[self.update_step-j-1])), H_u)
                H_u = torch.div(H_u, self.U)
                grad_right = grad_right - self.update_lr*H_u
            para_update = torch.squeeze(grad_right)
            return para_update
        else:
            return grad_final_tensor

    def inner_update_zero(self, loss_diff_list, normal_matrix, grad_final_tensor):
        with torch.no_grad():
            grad_right = torch.unsqueeze(grad_final_tensor, 1)
            for j in range(self.update_step):
                temp = torch.ones(normal_matrix[0][0].size()[0], 1).cuda()
                loss_diff_list_temp = torch.matmul(temp, 
                    torch.stack(loss_diff_list[self.update_step-j-1]).unsqueeze(0))
                nv_tensor = torch.stack(normal_matrix[self.update_step-j-1])
                loss_normal = torch.t(nv_tensor)*loss_diff_list_temp
                H_u = torch.matmul(nv_tensor, grad_right)
                H_u = torch.matmul(loss_normal, H_u)
                H_u = torch.div(H_u, self.U)
                grad_right = grad_right - self.update_lr*H_u
            para_update = torch.squeeze(grad_right)
            return para_update

    def first_order(self, u, optimizer_inner, grad_list, nv_matrix_i_k, net_inner, x_spt_i, y_spt_i):
        if u==self.U:
            logits = net_inner(x_spt_i, net_inner.parameters(), bn_training=True)
            loss = F.mse_loss(logits, y_spt_i)
            grad = torch.autograd.grad(loss, net_inner.parameters(), retain_graph=True)
            grad_list.append(grad)
            optimizer_inner.zero_grad()
            loss.backward()
            optimizer_inner.step()
           
        else:     
           
            fast_weight_delta = list(map(lambda q: q[0] + self.delta * q[1], zip(net_inner.parameters(), 
                                                nv_matrix_i_k[u])))
            logits_u = net_inner(x_spt_i, fast_weight_delta, bn_training=True)
            loss = F.mse_loss(logits_u, y_spt_i)
            # Gradient of Gaussian vector list form and tensor form
            with torch.no_grad():
                grad_u = torch.autograd.grad(loss, fast_weight_delta)
            grad_list.append(grad_u)
          

    def forward(self, x_spt, y_spt, x_qry, y_qry, nv_matrix, nv_tensor_matrix, step):
        """
        :param x_spt:   [b, setsz, c_, h, w] inner loop update
        :param y_spt:   [b, setsz] inner loop update
        :param x_qry:   [b, querysz, c_, h, w] outer loop update
        :param y_qry:   [b, querysz] outer loop update
        :num_list:      List contain split size of network parameters
        :return:
        """
       
        task_num, setsz, c_ = x_spt.size()
    
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
       
        para_update_matrix = []
        tmp = filter(lambda x: x.requires_grad, self.net.parameters())
        num_list = list(map(lambda x: np.prod(x.shape), tmp))
        
        if self.approx_method == 'ggs':
            net_inner = deepcopy(self.net)
            optimizer_inner = torch.optim.SGD(net_inner.parameters(), self.update_lr)
            para_original = self.net.state_dict()          
            
            for i in range(task_num):
                net_inner.load_state_dict(para_original)
                grad_diff_matrix = []
                for k in range(self.update_step):
                    grad_list = [] 
                    grad_diff_list = []   
                    for u in range(self.U+1):
                        self.first_order(u, optimizer_inner, grad_list, nv_matrix[i][k], net_inner, x_spt[i], y_spt[i])
                    
                    grad = grad_list[self.U]
                    for u in range(self.U):
                        grad_u = grad_list[u]
                        grad_diff = torch.cat(list(map(lambda p: torch.div(torch.reshape(p[0], (-1,)) - 
                        torch.reshape(p[1], (-1,)), self.delta), zip(grad_u, grad))))
                        grad_diff_list.append(grad_diff)
                    grad_diff_matrix.append(grad_diff_list)

                # compute the gradient of final parameter
                logits_q = net_inner(x_qry[i], net_inner.parameters())
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k+1] += loss_q
                grad_final = torch.autograd.grad(loss_q, net_inner.parameters())
                grad_final_tensor = torch.cat(list(map(lambda x:torch.reshape(x, (-1,)), grad_final)))
                # compute the update for the parameter
                with torch.no_grad():  
                    para_update = self.inner_update(grad_diff_matrix, nv_tensor_matrix[i], grad_final_tensor)        
                    para_update_matrix.append(para_update)
            with torch.no_grad():
                grad_update = torch.mean(torch.stack(para_update_matrix), dim=0)
                grad_update = torch.split(grad_update, num_list)
                for para, update in zip(self.net.parameters(), grad_update):
                    para.grad = torch.reshape(update, para.size())
            self.meta_optim.step()
            if step % 50 == 0:
                print('step:', step, '\ttraining loss:', losses_q)
            return losses_q
      

        elif self.approx_method == 'first_order':
            net_inner = deepcopy(self.net)
            optimizer_inner = torch.optim.SGD(net_inner.parameters(), self.update_lr)
            para_original = self.net.state_dict()
            for i in range(task_num):
                net_inner.load_state_dict(para_original)
                for k in range(self.update_step):
                    logits = net_inner(x_spt[i], net_inner.parameters(), bn_training=True)
                    loss = F.mse_loss(logits, y_spt[i])
                    optimizer_inner.zero_grad()
                    loss.backward()
                    optimizer_inner.step()
                 
                logits_q = net_inner(x_qry[i], net_inner.parameters())
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k+1] += loss_q
                with torch.no_grad():
                    grad_q = torch.autograd.grad(loss_q, net_inner.parameters())
                    if i==0:
                        grad_update = copy(grad_q)
                    else:
                        grad_update = list(map(lambda p: p[0]+p[1], zip(grad_update, grad_q)))
            
            with torch.no_grad():
                for para, update in zip(self.net.parameters(), grad_update):
                    para.grad = update/self.task_num
            self.meta_optim.step()
            if step % 50 == 0:
                print('step:', step, '\ttraining loss:', losses_q)
            return losses_q
        
        elif self.approx_method == 'maml':
            for i in range(task_num):
                # 1. run the i-th task and compute loss for k=0
                logits = self.net(x_spt[i])
                loss = F.mse_loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.mse_loss(logits_q, y_qry[i])
                    losses_q[0] += loss_q

                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.mse_loss(logits_q, y_qry[i])
                    losses_q[1] += loss_q

                for k in range(1, self.update_step):
                    # inner loop update steps
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)
                    loss = F.mse_loss(logits, y_spt[i])
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.mse_loss(logits_q, y_qry[i])
                    losses_q[k + 1] += loss_q

            # end of all tasks
            # sum over all losses on query set across all tasks (last parameters of all inner loop tasks: losses_q[-1])
            loss_q = losses_q[-1] / task_num

            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()
           
            if step % 50 == 0:
                losses_q = list(a.detach().cpu().numpy() for a in losses_q)
                print('step:', step, '\ttraining loss:', losses_q)
        
            return losses_q

        elif self.approx_method == 'zero_order':
            net_inner = deepcopy(self.net)
            optimizer_inner = torch.optim.SGD(net_inner.parameters(), self.update_lr)
            para_original = self.net.state_dict()          
            
            for i in range(task_num):
                net_inner.load_state_dict(para_original)
                func_diff_matrix = []
                for k in range(self.update_step):
                    loss_list = [] 
                    for u in range(self.U+1):
                        self.zero_order(u, optimizer_inner, loss_list, nv_matrix[i][k], net_inner, x_spt[i], y_spt[i])
                    loss_original = loss_list[self.U]
                    func_diff_list = list(map(lambda p: torch.div((p[0]-2*loss_original), 
                        2*self.delta*self.delta), zip(loss_list[0:self.U])))
                    func_diff_matrix.append(func_diff_list)

                # compute the gradient of final parameter
                logits_q = net_inner(x_qry[i], net_inner.parameters())
                loss_q = F.mse_loss(logits_q, y_qry[i])
                losses_q[k+1] += loss_q
                grad_final = torch.autograd.grad(loss_q, net_inner.parameters())
                grad_final_tensor = torch.cat(list(map(lambda x:torch.reshape(x, (-1,)), grad_final)))
                # compute the update for the parameter
                with torch.no_grad():  
                    para_update = self.inner_update_zero(func_diff_matrix, nv_tensor_matrix[i], grad_final_tensor)        
                    para_update_matrix.append(para_update)
            with torch.no_grad():
                grad_update = torch.mean(torch.stack(para_update_matrix), dim=0)
                grad_update = torch.split(grad_update, num_list)
                for para, update in zip(self.net.parameters(), grad_update):
                    para.grad = torch.reshape(update, para.size())
            self.meta_optim.step()
            if step % 50 == 0:
                print('step:', step, '\ttraining loss:', losses_q)
            return losses_q


    def zero_order(self, u, optimizer_inner, loss_list, nv_matrix_i_k, net_inner, x_spt_i, y_spt_i):
        if u==self.U:
            logits = net_inner(x_spt_i, net_inner.parameters(), bn_training=True)
            loss = F.mse_loss(logits, y_spt_i)
            loss_list.append(loss)
            optimizer_inner.zero_grad()
            loss.backward()
            optimizer_inner.step()
        else:     
            with torch.no_grad():
                fast_weight_delta_add = list(map(lambda q: q[0] + self.delta * q[1], zip(net_inner.parameters(), 
                                                    nv_matrix_i_k[u])))
                fast_weight_delta_minus = list(map(lambda q: q[0] - self.delta * q[1], zip(net_inner.parameters(), 
                                                    nv_matrix_i_k[u])))
                logits_add = net_inner(x_spt_i, fast_weight_delta_add, bn_training=True)
                loss_add = F.mse_loss(logits_add, y_spt_i)
                logits_minus = net_inner(x_spt_i, fast_weight_delta_minus, bn_training=True)
                loss_minus = F.mse_loss(logits_minus, y_spt_i)
                loss_total = (loss_add + loss_minus).detach()
                loss_list.append(loss_total)

def main():
    pass


if __name__ == '__main__':
    main()
