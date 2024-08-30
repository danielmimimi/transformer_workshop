
from math import ceil, floor
from pathlib import Path
from matplotlib.lines import Line2D
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import math

class VcaTensorboardLogger(object):
    def __init__(self,summary_writer:SummaryWriter) -> None:
        self.summary_writer = summary_writer
        # logging.getLogger('matplotlib.font_manager').disabled = True

    # def write_hyperparameter(self,config_reader:VcaJsonConfigReader):
    #     """Write hyperparamters/metrics directly from config"""
    #     hparams_or_none = config_reader("Logger hyper_parameter")
    #     self.hparams = None
    #     if hparams_or_none is not None:
    #         hparams_with_values = {}
    #         metrics_paths = {}
    #         for hparam in hparams_or_none:
    #             hparams_with_values[hparam.split(' ')[-1]] = config_reader(hparam)
            
    #         metrics_or_none = config_reader("Logger metric_paths")
    #         if metrics_or_none is not None:
    #             for metric in metrics_or_none:
    #                 metrics_paths[metric] = 0

    #             self.summary_writer.add_hparams(hparams_with_values,metrics_paths)
    #     self.hparams = hparams_with_values
                

    def write_scalars(self,path,scalars,iteration:int):
        self.summary_writer.add_scalars(path,scalars,iteration)
        
    def write_scalar(self,path,scalar,iteration:int):
        """path can be Loss/train"""
        self.summary_writer.add_scalar(path, scalar, iteration)
        

    def write_graph(self,model,model_input):
        """ add the model and a dummy input, they must be on the same device!"""
        self.summary_writer.add_graph(model,model_input)

    def write_image(self,path,image,iteration:int):
        """path can be Loss/train, be aware that the image must be HWC! """
        self.summary_writer.add_image(path, image, iteration,dataformats='HWC')
        
    def write_histogramm(self,path, values, iteration):
        """plots values as histogramm"""
        self.summary_writer.add_histogram(path,values,iteration)

    def flush_messages(self):
        self.summary_writer.flush()

    def write_conv_results(self, path, tensor,iteration:int):
        """ Plots max 64 feature maps and one image """
        tensor = tensor.cpu()
        num_images, num_subplots, height, width = tensor.size()
        height_of_subplot = 8
        side_of_subplot = min(math.floor(num_subplots/height_of_subplot),8)
        # PLOTS MAX 64 LAYERS
        
        # Plot each image
        fig = plt.figure(figsize=(30, 30))
        for image_index in range(1):# TAKE ONLY THE FIRST
            # fig, axs = plt.subplots(height_of_subplot, side_of_subplot)
            fig.suptitle(f'Image {image_index+1}')
            # REMOVE BORDERS
            plt.subplots_adjust(wspace=0, hspace=0)
            for j in range(height_of_subplot):
                for k in range(side_of_subplot):
                    current_index = j*side_of_subplot + k
                    if current_index == 64: # we will visualize only 8x8 blocks from each layer
                        break
                    plt.subplot(8, 8, current_index + 1)
                    plt.imshow(tensor[image_index, current_index].detach().numpy(), cmap='gray')
                    plt.axis("off")
                    # axs[j, k].imshow(tensor[image_index, current_index].detach().numpy(), cmap='gray')
                    # axs[j, k].axis('off')
            fig.canvas.draw()
            # convert to a NumPy array
            buf = fig.canvas.buffer_rgba()   
            image_from_plot = np.asarray(buf)
            plt.close()
            self.write_image(path, image_from_plot, iteration)


    def write_detections(self,path,detections,targets,image_path:Path,iteration:int):
        """Pass in detection xyxy,conf,class
        targets with xywh,class and the 
        image with CWH"""
        image = cv2.imread(image_path.as_posix())
        image = cv2.resize(image,(640,640))
        h,w,c = image.shape
        # scale detections

        # detections[:, :4] *= torch.tensor([w, h, w, h])
        targets[:, 1:5] *= torch.tensor([w, h, w, h])

        thickness = 2
        color_blue = (255, 255, 0) 
        color_red = (0, 0, 255) 
        
        for detection in detections:
            probability = detection[4]
            
            x1,y1 = detection[0:2]
            x2,y2 = detection[2:4]
            start_point = (floor(x1),floor(y1))
            end_point = (ceil(x2),ceil(y2))
            if x1 > x2:
                continue
            if y1 > y2:
                continue
            image = cv2.rectangle(image, start_point, end_point, color_blue, thickness) 
            
        for ground_truth in targets:
            cx,cy = ground_truth[1:3].type(torch.int16)
            width,height = ground_truth[3:5].type(torch.int16)
            start_point = (int(cx-(width/2)),int(cy-(height/2)))
            end_point = (int(cx+(width/2)),int(cy+(height/2)))
    
            image = cv2.rectangle(image, start_point, end_point, color_red, thickness) 
        self.write_image(path,image,iteration)


    def write_weights(self,model,iteration:int):
        for tag, value in model.named_parameters():
            if value is not None and ("bias" not in tag):
                values = value.ravel()
                self.summary_writer.add_histogram(tag+"/weights",values.detach().cpu(),global_step=iteration,bins='auto')
                # DOES NOT WORK
                # self.summary_writer.add_scalars(tag+"/weights",{
                #                                                 '0.25q':values.detach().cpu().quantile(0.25),
                #                                                 '0.5q':values.detach().cpu().quantile(0.5),
                #                                                 '0.75':values.detach().cpu().quantile(0.75),
                #                                                 '0.95':values.detach().cpu().quantile(0.05),
                #                                                 },global_step=iteration)
    

    def write_gradients(self,model,iteration:int):
        for tag, value in model.named_parameters():
            if value.grad is not None:
                if(value.requires_grad) and ("bias" not in tag) and not value.grad.isnan().any() and not value.grad.isinf().any():
                    self.write_histogramm(tag + "/grad", value.grad.cpu(), iteration)

    def write_gradient_flow(self,path,named_parameters,iteration:int):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        fig = plt.figure(figsize=(200,4))
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        fig.canvas.draw()
        # convert to a NumPy array
        buf = fig.canvas.buffer_rgba()   
        image_from_plot = np.asarray(buf)
        plt.close()

        self.write_image(path, image_from_plot, iteration)
            
            
    @torch.no_grad()
    def log_attentionmaps(self,path,image_index, input_image, attention_maps,iteration:int):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        labels = ['Input Image']
        heads = [input_image.squeeze().cpu().detach().numpy()]
        
        for encoder_index,attention_map in enumerate(attention_maps):   
            attention_map_image = attention_map[image_index]  
            if len(attention_map_image.shape) != 3:
                attention_map_image.unsqueeze(dim=0)
            meaned_attention_map = torch.mean(attention_map_image, dim=0)
            residual_att = torch.eye(meaned_attention_map.size(0)).to(device)
            aug_att_mat = meaned_attention_map + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            v = joint_attentions[0]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[ 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
            mask = cv2.resize(mask / mask.max(), input_image.shape[1:])[..., np.newaxis]
            # result = (mask.squeeze() * im.squeeze().cpu().detach().numpy()).astype("uint8")
            heads.append(mask.squeeze())
            labels.append("Mean Attention Map Encoder {}".format(str(encoder_index)))


        fig, axs = plt.subplots(1, len(heads),figsize=(20, 12))
        plt.subplots_adjust(wspace=0, hspace=0)
        for index, image in enumerate(heads):
            if image.shape[0] == 3:
                axs[index].imshow(np.transpose(image, (1, 2, 0)))
            else:
                axs[index].imshow(image)
            axs[index].set_title(labels[index])

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()   
        image_from_plot = np.asarray(buf)
        plt.close()

        self.write_image(path, image_from_plot, iteration)
        
        
    def log_position_encoding(self,path,grid_h, grid_w, pos_embeddings,iteration:int):
        
        seq_length = pos_embeddings.shape[0]
        x = pos_embeddings.repeat_interleave(seq_length, 0) 
        y = pos_embeddings.tile((seq_length, 1)) 
        similarities = nn.CosineSimilarity(dim=-1)(x, y)
    
        similarity_heatmaps = similarities.view((grid_h, grid_w, grid_h, grid_w)).numpy()
        
        
        fig, ax = plt.subplots(grid_h, grid_w, figsize=(15, 15))
        fontsize = 24

        for i in range(grid_h):
            for j in range(grid_w):
                im = ax[i, j].imshow(similarity_heatmaps[i][j], vmin=-1, vmax=1)
                if i == grid_h - 1:
                    ax[i, j].set_xlabel(j + 1, fontsize=fontsize, rotation='horizontal')
                if j == 0:
                    ax[i, j].set_ylabel(i + 1, fontsize=fontsize)
                ax[i, j].tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), aspect=25, ticks=[-1, 1])
        cbar.set_label('Cosine similarity', rotation=-270, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        fig.text(0.45, 0.04, 'Input patch row', ha='center', fontsize=fontsize)
        fig.text(0.04, 0.5, 'Input patch column', va='center', rotation='vertical', fontsize=fontsize)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()   
        image_from_plot = np.asarray(buf)
        plt.close()

        self.write_image(path, image_from_plot, iteration)