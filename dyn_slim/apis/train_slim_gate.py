import logging
import time
import random
import pickle
import os
from collections import OrderedDict

from dyn_slim.models.dyn_slim_blocks import MultiHeadGate
from dyn_slim.models.dyn_slim_ops import DSBatchNorm2d
from dyn_slim.utils import add_flops, accuracy

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False
from timm.utils import AverageMeter, reduce_tensor

import numpy as np
import torch
import torch.nn as nn

import datetime
from jtop import jtop,JtopException
import csv
from multiprocessing import Process
from dyn_slim.utils.os_metrics import Log_device



model_mac_hooks = []

#ADD PATH
time_filename=datetime.datetime.now().strftime("%Y%M%d-%H%M%S")
#path='/home/sharon/Documents/Research/Slimmable/DS-Net/sys-data/'+time_filename+'/'
path_metrics='/home/iasl/DS-Net-V2/sys-data/'+time_filename+'/'
#path='/home/sharon/Documents/Research/Slimmable/DS-Net/sys-data/image-model-accuracy-mapping/'
os.mkdir(path_metrics)
delay=0.01
logger=Log_device(delay)

#  METRICS PROCESSES
try:
    p1 = Process(target=logger.start_log, args=(path_metrics+'orin_logs_cpu_gpu_'+time_filename+'.csv',))
    p1.start()
    print("PROCESS 1 STARTED")
    #p2 = Process(target=logger.start_log_net, args=(path_metrics+'orin_logs_net_'+time_filename+'.csv',))
    #p2.start()
except JtopException as e:
    print(e)
except KeyboardInterrupt:
    print("Closed with CTRL-C")
except IOError:
    print("I/O error")

def end_processes(self):
    p1.join()
    #p2.join()

def generate_gate_labels(model, loader, output_filename=''):
    print("GENERATE GATE LABELS")
    for n, m in model.named_modules():  # Freeze bn
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, DSBatchNorm2d):
            m.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 18:
            m.in_channels_list = m.in_channels_list[0:18]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 18:
            m.out_channels_list = m.out_channels_list[0:18]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()
            
    is_one_gate = False
    list_count = (1 if is_one_gate else 5)
    gate_labels = {}

    #stop value to prevent force gates at a uniform length
    reduce = False

    with torch.no_grad():
        #this should be a batch size of 1, so we can generate a gate for each input
        for batch_idx, (input, target, path) in enumerate(loader):
            gate_state = [0] * list_count
            cannot_predict_or_is_smallest = True
                
            #find uniform width that can handle image
            channels_to_explore = 18
            for channel in range(channels_to_explore):
                if channel == 0:
                    set_model_mode(model, 'smallest')
                else:
                    set_model_mode(model, 'uniform', None, channel)

                output = model(input)
                conf_s, correct_s = accuracy(output, target, no_reduce=True)
                #print('first gate correct; ', correct_s)

                if correct_s[0][0]:
                    cannot_predict_or_is_smallest = False if channel > 0 else True
                    gate_state = [channel] * list_count
                    break

            #if if could no predict set to smallest width
            #or if there is only one activate gate set, save the uniform width
            #else attempt to reduce gates until any reduction fails
            # if not cannot_predict_or_is_smallest or list_count > 1:
            #     target_found = True
            #     temp_gate_state = gate_state

            #     while target_found and reduce:
            #         target_found = False
            #         gate_indeces = [0,1,2,3,4] #hardcoding for a set of 5 gates for a quick implementaion

            #         for i in range(len(gate_state)):
            #             random_idx = random.randint(0, len(gate_indeces)-1)
            #             i_idx = gate_indeces.pop(random_idx)

            #             if temp_gate_state[i_idx] > 0:
            #                 temp_gate_state[i_idx] -= 1 if temp_gate_state[i_idx] > 0 else 0

            #                 #print(temp_gate_state)
            #                 set_model_mode(model, 'multi-choice', None, temp_gate_state)

            #                 output = model(input)
            #                 conf_s, correct_s = accuracy(output, target, no_reduce=True)

            #                 #print('secondary gate correct; ', correct_s)
            #                 if correct_s[0][0]:
            #                     gate_state = temp_gate_state
            #                     target_found = True
            #                     break
                        
            #                 temp_gate_state[i_idx] += 1 if gate_state[i_idx] != 0 else 0

            #save gate
            gate_labels[path[0]] = gate_state
            
            print(gate_state)
            if batch_idx%1000 == 0:
                logging.info('Currently found - {} - gates'.format(batch_idx))
        print("FINISHED")
        pickle_file = open(output_filename, 'wb+')
        pickle.dump(gate_labels, pickle_file)
        pickle_file.close()
                

def train_epoch_slim_gate(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None,
        optimizer_step=1, first_epoch=False):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    flops_m = AverageMeter()
    ce_loss_m = AverageMeter()
    flops_loss_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    gate_loss_m_l = [AverageMeter() for i in range(num_gate)]
    model.train()
    for n, m in model.named_modules():  # Freeze bn
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, DSBatchNorm2d):
            m.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 4:
            m.in_channels_list = m.in_channels_list[start_chn_idx:18]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 4:
            m.out_channels_list = m.out_channels_list[start_chn_idx:18]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    model.apply(lambda m: add_mac_hooks(m))
    #print(model)

    filename = os.getcwd() + "gate_train_reduced_dict.p"
    print(filename)
    if first_epoch and not os.path.isfile(filename):
        generate_gate_labels(model, loader, filename)

    pickle_file = open(filename, 'rb')
    gate_val_dict = pickle.load(pickle_file)
    pickle_file.close()
    end = time.time()

    gate_loss_avg = 0

    flop_list = [133067936.0, 174799072.0, 223956256.0, 280539488.0, 344548768.0, 415984096.0, 494845472.0, 581132928.0,
                 581132928.0, 674846336.0, 775985920.0, 884551424.0, 1000543104.0, 1123960704.0, 1254804480.0,
                 1393074176.0, 1538770048.0, 1691891840.0, 1852439808.0]

    model.train()
    for batch_idx, (input, target, path) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        #TODO:change this later (from current hardcoded prefetch fix)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()

        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.zero_grad()

        # generate online labels
        #Currently hardcoding for batch size of 1 
        gate_targets = []
        gate_targets = gate_val_dict[path[0]] #gate_targets.append(gate_val_dict[path[0]])
        # =============
        set_model_mode(model, 'dynamic')
        # GET_LATENCY
        START_TIME = time.time()
        output = model(input)
        # print("end dynamic")
        torch.cuda.synchronize()
        END_TIME = time.time()
        # conf_s, correct_s = accuracy(output, target, no_reduce=True)
        dict_res = model.get_stats()
        dict_res["FILENAME"] = sample_fname.split("/")[-1]
        dict_res["LABEL"] = target.clone().detach().cpu().numpy()[0]
        dict_res["CLASSIFIED_AS"] = np.argmax(output.clone().detach().cpu().numpy())
        dict_res["START_TIME"] = START_TIME
        dict_res["END_TIME"] = END_TIME

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        #  SGS Loss
        gate_loss = 0
        gate_num = 0
        gate_loss_l = [0.0]
        gate_acc_l = []
        keep_gate_sum = None
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    #print(m.keep_gate, torch.tensor([gate_targets[gate_num]]))
                    g_loss = loss_fn(m.keep_gate, torch.LongTensor([gate_targets[gate_num]]).cuda())
                    gate_loss += g_loss
                    gate_loss_l.append(g_loss)
                    gate_acc_l.append(accuracy(m.keep_gate, torch.LongTensor([gate_targets[gate_num]]).cuda(), topk=(1,))[0])
                    gate_num += 1
                    keep_gate_sum = m.keep_gate if keep_gate_sum == None else (m.keep_gate + keep_gate_sum)

        
        g_comb_loss = loss_fn(keep_gate_sum, torch.LongTensor([gate_targets[0]]).cuda())
        gate_loss /= gate_num
        gate_loss_avg += gate_loss.item()

        #  MAdds Loss ====> Current not considering but keeping in place for implementation testing
        running_flops = add_flops(model)
        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
        else:
            running_flops = torch.FloatTensor([running_flops]).cuda()
        flops_loss = ((torch.tensor(flop_list[gate_targets[0]]).cuda() - running_flops) / 1e8) ** 2

        #====Testing floop control====
        #print(running_flops)
        #flops_loss = ((running_flops/1e7)-torch.tensor([18]).cuda()) **2 #(torch.exp((running_flops/1e7)-torch.tensor([18]).cuda())) 
        #print(running_flops, flops_loss)

        #  Target Loss, back-propagate through gumbel-softmax
        #print(output.shape, target.shape)
        ce_loss = loss_fn(output, target)

        loss =  gate_loss + (0.1 * ce_loss)  + (flops_loss) ##+ (0.5*gate_loss) #+g_comb_loss
        acc1 = accuracy(output, target, topk=(1,))[0]

        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if last_batch or (batch_idx + 1) % optimizer_step == 0:
            optimizer.step()

            updated_gates = []
            for n, m in model.named_modules():
                if isinstance(m, MultiHeadGate) and m.has_gate:
                    updated_gates.append(m.gate)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            acc_m.update(acc1.item(), input.size(0))
            flops_m.update(running_flops.item(), input.size(0))
            ce_loss_m.update(ce_loss.item(), input.size(0))
            flops_loss_m.update(flops_loss.item(), input.size(0))
        else:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_acc = reduce_tensor(acc1, args.world_size)
            reduced_flops = reduce_tensor(running_flops, args.world_size)
            reduced_loss_flops = reduce_tensor(flops_loss, args.world_size)
            reduced_ce_loss = reduce_tensor(ce_loss, args.world_size)
            reduced_acc_gate_l = reduce_list_tensor(gate_acc_l, args.world_size)
            reduced_gate_loss_l = reduce_list_tensor(gate_loss_l, args.world_size)
            losses_m.update(reduced_loss.item(), input.size(0))
            acc_m.update(reduced_acc.item(), input.size(0))
            flops_m.update(reduced_flops.item(), input.size(0))
            flops_loss_m.update(reduced_loss_flops.item(), input.size(0))
            ce_loss_m.update(reduced_ce_loss.item(), input.size(0))
            for i in range(num_gate):
                acc_gate_m_l[i].update(reduced_acc_gate_l[i].item(), input.size(0))
                gate_loss_m_l[i].update(reduced_gate_loss_l[i].item(), input.size(0))
        batch_time_m.update(time.time() - end)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print(gate_loss.item())
            print_gate_stats(model)
            logging.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'CELoss: {celoss.val:>9.6f} ({celoss.avg:>6.4f})  '
                #'GateLoss: {gate_loss[0].val:>6.4f} ({gate_loss[0].avg:>6.4f})  '
                'FlopsLoss: {flopsloss.val:>9.6f} ({flopsloss.avg:>6.4f})  '
                'TrainAcc: {acc.val:>9.6f} ({acc.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'LR: {lr:.3e}  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    epoch,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    flopsloss=flops_loss_m,
                    acc=acc_m,
                    flops=flops_m,
                    celoss=ce_loss_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m,
                    #gate_loss=[gate_loss],
                    #gate_avg = gate_loss_avg/50,
                    acc_gate=acc_gate_m_l
                )
            )
            print('AVG Gate loss:', gate_loss_avg/50, 'Current Gate Target: ', gate_targets)
            gate_loss_avg = 0

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


@torch.no_grad()
def validate_gate(model, loader, loss_fn, args, log_suffix='', first_epoch=False):
    start_chn_idx = args.start_chn_idx
    num_gate = 1

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    flops_m = AverageMeter()
    acc_gate_m_l = [AverageMeter() for i in range(num_gate)]
    model.eval()

    for n, m in model.named_modules():
        if len(getattr(m, 'in_channels_list', [])) > 18:
            m.in_channels_list = m.in_channels_list[start_chn_idx:18]
            m.in_channels_list_tensor = torch.from_numpy(
                np.array(m.in_channels_list)).float().cuda()
        if len(getattr(m, 'out_channels_list', [])) > 18:
            m.out_channels_list = m.out_channels_list[start_chn_idx:18]
            m.out_channels_list_tensor = torch.from_numpy(
                np.array(m.out_channels_list)).float().cuda()

    end = time.time()
    last_idx = len(loader) - 1
    model.apply(lambda m: add_mac_hooks(m))

    filename = os.getcwd() + "gate_val_reduced_dict.p"
    if not os.path.isfile(filename):
        generate_gate_labels(model, loader, filename)

    pickle_file = open(filename, 'rb')
    gate_val_dict = pickle.load(pickle_file)
    pickle_file.close()

    #channel = 0
    print("BEFORE LOADER")
    for batch_idx, (input, target, path) in enumerate(loader):
        sample_fname, _ = loader.dataset.samples[batch_idx]  # GET THE FILENAME
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
        # generate online labels
        #Currently hardcoding for batch size of 1 
        gate_targets = []
        gate_targets = gate_val_dict[path[0]] #gate_targets.append(gate_val_dict[path[0]])
        # =============
        set_model_mode(model, 'dynamic')
        # GET_LATENCY
        print("START MEASURING LATENCY")
        START_TIME = time.time()
        output = model(input)
        # print("end dynamic")
        torch.cuda.synchronize()
        END_TIME = time.time()
        # conf_s, correct_s = accuracy(output, target, no_reduce=True)
        dict_res = model.get_stats()
        dict_res["FILENAME"] = sample_fname.split("/")[-1]
        dict_res["LABEL"] = target.clone().detach().cpu().numpy()[0]
        dict_res["CLASSIFIED_AS"] = np.argmax(output.clone().detach().cpu().numpy())
        dict_res["START_TIME"] = START_TIME
        dict_res["END_TIME"] = END_TIME

        if hasattr(model, 'module'):
            model_ = model.module
        else:
            model_ = model

        gate_num = 0
        gate_acc_l = []
        for n, m in model_.named_modules():
            if isinstance(m, MultiHeadGate):
                if getattr(m, 'keep_gate', None) is not None:
                    gate_acc_l.append(accuracy(m.keep_gate, torch.LongTensor([gate_targets[gate_num]]).cuda(), topk=(1,))[0])
                    gate_num += 1

        running_flops = add_flops(model)

        if isinstance(running_flops, torch.Tensor):
            running_flops = running_flops.float().mean().cuda()
            dict_res["NUM_FLOPS"] = running_flops.item()
        else:
            dict_res["NUM_FLOPS"] = running_flops
            running_flops = torch.FloatTensor([running_flops]).cuda()
        # Writing metrics dictionary
        print("PATH: ",path_metrics)
        print("NUM FLOPS: ", dict_res["NUM_FLOPS"])
        print("time_filename: ", time_filename)
        if batch_idx > 0:
            filename = path_metrics + 'orin_model_data_' + str(int(dict_res["NUM_FLOPS"])) + '_gpu_' + time_filename+'.csv'
            file_exists = os.path.isfile(filename)
            with open(filename, 'a') as csvfile:
                fieldnames = list(dict_res.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(dict_res)
        time.sleep(3)

        loss = loss_fn(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            prec1_m.update(prec1.item(), input.size(0))
            prec5_m.update(prec5.item(), input.size(0))
            flops_m.update(running_flops.item(), input.size(0))
        else:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_prec1 = reduce_tensor(prec1, args.world_size)
            reduced_prec5 = reduce_tensor(prec5, args.world_size)
            reduced_flops = reduce_tensor(running_flops, args.world_size)
            reduced_acc_gate_l = reduce_list_tensor(gate_acc_l, args.world_size)
            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(reduced_prec1.item(), input.size(0))
            prec5_m.update(reduced_prec5.item(), input.size(0))
            flops_m.update(reduced_flops.item(), input.size(0))
            for i in range(num_gate):
                acc_gate_m_l[i].update(reduced_acc_gate_l[i].item(), input.size(0))
        batch_time_m.update(time.time() - end)
        if (last_batch or batch_idx % args.log_interval == 0) and args.local_rank == 0 and batch_idx != 0:
            print_gate_stats(model)
            log_name = 'Test' + log_suffix
            logging.info(
                '{}: [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {prec1.val:>9.6f} ({prec1.avg:>6.4f})  '
                'Acc@5: {prec5.val:>9.6f} ({prec5.avg:>6.4f})  '
                'GateAcc: {acc_gate[0].val:>6.4f}({acc_gate[0].avg:>6.4f})  '
                'Flops: {flops.val:>6.0f} ({flops.avg:>6.0f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'DataTime: {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                    log_name,
                    batch_idx, last_idx,
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    prec1=prec1_m,
                    prec5=prec5_m,
                    flops=flops_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    data_time=data_time_m,
                    acc_gate=acc_gate_m_l
                )
            )
            #print('Gate: -', channel, '- \tFLOPs: -', running_flops.item())

        end = time.time()
        # end for
    metrics = OrderedDict(
        [('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg), ('flops', flops_m.avg)])

    return metrics


def reduce_list_tensor(tensor_l, world_size):
    ret_l = []
    for tensor in tensor_l:
        ret_l.append(reduce_tensor(tensor, world_size))
    return ret_l


def set_gate(m, gate=None):
    if gate is not None:
        gate = gate.cuda()
    if hasattr(m, 'gate'):
        setattr(m, 'gate', gate)


def module_mac(self, input, output):
    if isinstance(input[0], tuple):
        if isinstance(input[0][0], list):
            ins = input[0][0][3].size()
        else:
            ins = input[0][0].size()
    else:
        ins = input[0].size()
    if isinstance(output, tuple):
        if isinstance(output[0], list):
            outs = output[0][3].size()
        else:
            outs = output[0].size()
    else:
        outs = output.size()
    if isinstance(self, (nn.Conv2d, nn.ConvTranspose2d)):
        # print(type(self.running_inc), type(self.running_outc), type(self.running_kernel_size), type(outs[2]), type(self.running_groups))
        self.running_flops = (self.running_inc * self.running_outc *
                              self.running_kernel_size * self.running_kernel_size *
                              outs[2] * outs[3] / self.running_groups)
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, nn.Linear):
        if hasattr(self, 'running_inc'):
            self.running_flops = self.running_inc * self.running_outc
        else:
            self.running_flops = 0
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    elif isinstance(self, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        # NOTE: this function is correct only when stride == kernel size
        self.running_flops = self.running_inc * ins[2] * ins[3]
        # print(type(self), self.running_flops.mean().item() if isinstance(self.running_flops, torch.Tensor) else self.running_flops)
    return


def add_mac_hooks(m):
    global model_mac_hooks
    model_mac_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_mac(
            m, input, output)))


def remove_mac_hooks():
    global model_mac_hooks
    for h in model_mac_hooks:
        h.remove()
    model_mac_hooks = []


def set_model_mode(model, mode, seed=None, choice=None):
    if hasattr(model, 'module'):
        model.module.set_mode(mode, seed, choice)
    else:
        model.set_mode(mode, seed, choice)


def print_gate_stats(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    for n, m in model.named_modules():
        if isinstance(m, MultiHeadGate) and getattr(m, 'print_gate', None) is not None:
            logging.info('{}: {}'.format(n, m.print_gate.sum(0)))
