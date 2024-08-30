


from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.vca_scalar_collector import VcaScalarCollector
from utils.vca_tensorboard_logger import VcaTensorboardLogger
from utils.vca_logger import create_logger

class VitTrainer:
    def __init__(self,args,model,train_dataset,valid_dataset) -> None:
        
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.console_logger = create_logger(__file__)
        
        self.scalar_collector = VcaScalarCollector()
        self.logger = VcaTensorboardLogger(SummaryWriter(log_dir="runs_vit"+"/"+datetime.now().strftime("%m_%d_%Y_%H_%M")))
        
        self.reset_flag = False
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)
    
        self.epoch = 0
        
    def train(self):


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for epoch in range(0, self.args.epochs):
            
            self.console_logger.info("Train Epoch {}".format(epoch))
            
            self.epoch = epoch
            # TRAIN LOOP
            self.optimizer.zero_grad()
            self.__train_one_epoch()
            
            # EVAL LOOP
            self.__validate_one_epoch()

            # TENSORBOARD WRITING
            self.scalar_collector.publish(self.logger.write_scalars,self.epoch)
            self.logger.flush_messages()

    def __train_one_epoch(self):
        self.model.train()
        
        reset_flag = False
        for batch_idx, (img, labels) in enumerate(self.train_dataset):
            img = img.to(self.device)
            labels = labels.to(self.device)
            
            preds,attention_maps = self.model(img)
            
            loss = self.loss_function(preds, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            result_preds = preds.detach().argmax(dim=-1).tolist()
            result_gt = labels.detach().tolist()
            total_correct = len([True for x, y in zip(result_preds, result_gt) if x==y])
            total = len(result_gt)
            self.scalar_collector.add_scalar("training/loss/loss",loss.cpu().item())
            self.scalar_collector.add_scalar("training/classification/accuracy",total_correct * 100 / total)
            
            if reset_flag is False and self.args.enable_logging_image:
                reset_flag = True
                learned_embeddings = self.model.patch_embedding.pos_embedding[0].detach()
                image_pos_embeddings = learned_embeddings.cpu()
                self.logger.log_position_encoding("training/positionencoding/cosinesimilarity",self.args.amount_of_patches,self.args.amount_of_patches, image_pos_embeddings,self.epoch)
        
        
    def __validate_one_epoch(self):
        
        with torch.no_grad():
            self.model.eval()
            reset_flag = False
            for batch_idx, (img, labels) in enumerate(self.valid_dataset):
                img = img.to(self.device)
                labels = labels.to(self.device)
                
                preds,attention_maps = self.model(img)
                
                loss = self.loss_function(preds, labels)          
                
                result_preds = preds.detach().argmax(dim=-1).tolist()
                result_gt = labels.detach().tolist()
                total_correct = len([True for x, y in zip(result_preds, result_gt) if x==y])
                total = len(result_gt)
                self.scalar_collector.add_scalar("validation/loss/loss",loss.cpu().item())
                self.scalar_collector.add_scalar("validation/classification/accuracy",total_correct * 100 / total)
                
                if reset_flag is False and self.args.enable_logging_image:
                    reset_flag = True
                    # Plot only 10
                    for sample_index in range(0,10):
                        input_image = img[sample_index]
                        current_iteration = 0
                        self.logger.log_attentionmaps("validation/attentionmaps/{}".format(sample_index),sample_index,input_image,attention_maps,current_iteration)
