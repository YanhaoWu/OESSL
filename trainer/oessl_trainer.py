from tabnanny import check
from typing_extensions import assert_type
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist

from data_utils.collations import *
from numpy import inf, pi, cos, mean
import sys
from tools.box_generate import *
# from utils import make_dir
class oessl_trainer(pl.LightningModule):
    def __init__(self, model, criterion, train_loader, params):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.params = params
        self.writer = SummaryWriter(f'{params["log_dir"]}')
        self.iter_log = 1  # 100
        self.loss_eval = []
        self.loss_cc = []
        self.loss_pd = []
        self.train_step = 0
        self.val_step = 0
        self.pix_deal = self.params["pix_deal"]
        self.mix_loss = True

        self.scene_id = []
        self.object_id = []
        self.stage = 0

        if self.params['load_path'] is not None:
            self.load_checkpoint()



    ############################################################################################################################################
    # FORWARD                                                                                                                                  #
    ############################################################################################################################################
    def forward(self, xi, xj=None, s=None, exan=None, pix_deal=False):
        return self.model(pcd_q=xi, pcd_k=xj, segments=s, pix_deal=pix_deal, exchange_anno=exan)


    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################

    def pre_training_segment_step_mix_loss(self, batch, batch_nb): 
        
        

        (xi_coord_1, xi_feats_1, xi_exchange_1), (xj_coord_1, xj_feats_1), (xi_coord_2, xi_feats_2, xi_exchange_2), (xj_coord_2, xj_feats_2),  (S_i1j1, S_i1j2, S_i2j2, S_i2j1) = batch
       
       
        xi1, xj1 = collate_points_to_sparse_tensor(xi_coord_1, xi_feats_1, xj_coord_1, xj_feats_1)
        xi2, xj2 = collate_points_to_sparse_tensor(xi_coord_2, xi_feats_2, xj_coord_2, xj_feats_2)
        exchange_anno1 = ME.utils.batched_coordinates(array_to_torch_sequence(xi_exchange_1), dtype=torch.float32, device='cuda')[:, 1:]
        exchange_anno2 = ME.utils.batched_coordinates(array_to_torch_sequence(xi_exchange_2), dtype=torch.float32, device='cuda')[:, 1:]


        loss_remain, loss_swap, loss_PD= self.forward([xi1,xi2], [xj1,xj2], [S_i1j1, S_i1j2, S_i2j2, S_i2j1], [exchange_anno1, exchange_anno2], True)
        loss = (loss_remain + loss_swap + 2 * loss_PD) 



                
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            self.contrastive_iter_callback(loss.item(), (loss_remain).item(), loss_PD.item(),)

                
        return {'loss': loss}
        


    def training_step(self, batch, batch_nb):

        self.train_step += 1
        torch.cuda.empty_cache()
        loss =  self.pre_training_segment_step_mix_loss(batch, batch_nb)
        return loss



    def training_epoch_end(self, outputs):
        # print(timestamp)
        if dist.get_rank() == 0:
            self.checkpoint_callback()





    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################




    def checkpoint_callback(self):

        index_save = 0
        if (self.current_epoch + index_save ) % 50 == 0:          # +2 是前面跑错了， 所以临时+2，其他情况下应该是+1
            self.save_checkpoint(f'epoch{self.current_epoch}')

        # if ((self.current_epoch + 1 ) == self.params["training"]["max_epochs"]) or  ((self.current_epoch + 1 ) == self.params.stop_epoch):
        if ((self.current_epoch + 1 ) == self.params["training"]["max_epochs"]):

            self.save_checkpoint(f'epoch{self.current_epoch}')

    def contrastive_iter_callback(self, batch_loss, batch_loss1, batch_loss2, batch_pcd_loss=None, batch_segment_loss=None):
        # after each iteration we log the losses on tensorboard
        self.loss_eval.append(batch_loss)
        self.loss_cc.append(batch_loss1)
        self.loss_pd.append(batch_loss2)

        if self.train_step % self.iter_log == 0:
            self.write_summary(
                'training/loss_cc',
                mean(self.loss_cc),
                self.train_step,
            )

            self.write_summary(
                'training/loss_pd',
                mean(self.loss_pd),
                self.train_step,
            )

            self.write_summary(
                'training/learning_rate',
                self.scheduler.get_lr()[0],
                self.train_step,
            )

            # loss
            self.write_summary(
                'training/loss',
                mean(self.loss_eval),
                self.train_step,
            )

            self.loss_eval = []
            self.loss_cc = []
            self.loss_pd = []



    ############################################################################################################################################

    ############################################################################################################################################
    # SUMMARY WRITERS                                                                                                                          #
    ############################################################################################################################################

    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)


    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        self.configure_optimizers()


        #原本的代码
        # load model, best loss and optimizer
        file_name_1 = self.params['load_path']
        # checkpoint = torch.load(file_name)
        checkpoint = torch.load(file_name_1, map_location='cpu')

        if 'MSC' in self.params['load_path']:
            state_dict = checkpoint['state_dict']
            new_state = {}
            for k, v in state_dict.items():

                if ('module.backbone' in k):
                    new_name = k[len('module.backbone.'):]
                    new_state[new_name] = v
                    
            self.model.model_q.load_state_dict(new_state)
            self.model.model_k.load_state_dict(new_state)

        else:
            self.model.model_q.load_state_dict(checkpoint['model'])
            self.model.model_k.load_state_dict(checkpoint['model'])
        txt_path = self.params["save_dir"] + '/' + 'param.txt'
        file = open(txt_path ,'a')
        oldstdout = sys.stdout
        sys.stdout = file
        print("loaded the modle")
        print("loaded", file_name_1)
        file.close()
        sys.stdout = oldstdout

        # print("loaded", file_name_2)

        print("loaded the model")


    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.model.model_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params["save_dir"]}/{checkpoint_id}_model.pt'

        torch.save(state, file_name)


        torch.save(self.state_dict(), f'{self.params["save_dir"]}/{checkpoint_id}_full_model.pt')

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        print("get opt")
        # define optimizers
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params["optimizer_params"]["lr"], momentum=0.9, weight_decay=self.params["optimizer_params"]["decay_lr"], nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params["training"]["max_epochs"], eta_min=self.params["optimizer_params"]["lr"] / 1000)

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]

    ############################################################################################################################################

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        pass


    #@pl.data_loader
    def test_dataloader(self):
        pass
