from typing_extensions import assert_type
import torch
import torch.nn as nn
from data_utils.collations import *
latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNet256': 256,         # 
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}

class oessl_model(nn.Module):
    def __init__(self, model, model_preject, model_preject_pix, mode_predict, Exchange_predction_Layer, args, K=65536, m=0.999, T=0.1):
        super(oessl_model, self).__init__()

        self.K = K
        self.m = m
        self.T = T



        # online 部分
        self.model_q = model(in_channels=3, out_channels=latent_features[args["network"]["backbone"]])
        self.head_q = model_preject_pix(in_channels=latent_features[args["network"]["backbone"]], out_channels=args["network"]["feature_size"], batch_nor=True, pix_level=True)
        self.predict_q  = mode_predict()

        # target 部分
        self.model_k = model(in_channels=3, out_channels=latent_features[args["network"]["backbone"]])
        self.head_k = model_preject(in_channels=latent_features[args["network"]["backbone"]], out_channels=args["network"]["feature_size"], batch_nor=True)

        self.exchange_precdtion = Exchange_predction_Layer()
        self.expd_criterion = nn.CrossEntropyLoss(ignore_index=255)

        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize headection head k and q
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if torch.cuda.device_count() > 1:
            self.model_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_q)
            self.head_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_q)
            self.predict_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.predict_q)

            self.model_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_k)
            self.head_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_k)



    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)



    def compute_cross_loss(self,representation_q, representation_target_k, pix_deal, segments):


        h_qs = list_segments_points(representation_q.C, representation_q.F, segments[0])
        project_q = self.head_q(h_qs, pix_deal)
        predict_q = self.predict_q(project_q)
        q_seg_1 = nn.functional.normalize(predict_q, dim=1, p=2)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            try:
              h_ks, number_k = list_segments_points(representation_target_k.C, representation_target_k.F, segments[1], collect_numbers=True)
            except:
              print("after crop, no remain same clusters")
              return 0
            number_q = list_segments_number(segments[0])

            try:
              project_target_k = self.head_k(h_ks)        # 就只有投影部分，没有预测部分了 / 对结果进行Pooling
            except:
            
              print("h_ks shape",h_ks.shape)
              print("only one segment")
              return 0 
            
            if pix_deal:
                assert len(number_k) == project_target_k.shape[0]
                assert len(number_q) == project_target_k.shape[0]

                feature_size =  project_target_k.shape[1]

                k_seg_2_points = project_target_k[0].expand(number_q[0], feature_size)

                for idx in range(len(number_q)-1):  # 注意，q_seg_2应该用的是number_k  k_seg_2应该用的是number_q。因为最终是向另外一个变换对齐
                    number_ori_points = number_q[idx+1]
                    temp_tensor = project_target_k[idx+1].expand(number_ori_points, feature_size)
                    k_seg_2_points = torch.cat((k_seg_2_points, temp_tensor))

                k_seg_2 = nn.functional.normalize(k_seg_2_points, dim=1, p=2)
            
            else:
                k_seg_2 = nn.functional.normalize(project_target_k, dim=1, p=2)

        # 注意，这里地方必须保留梯度
        l_1 = (q_seg_1 * k_seg_2.detach()).sum(dim=-1).mean()
        loss = 1 - l_1   
        return loss


    def compute_P2C_C2C(self, representation_q_0, representation_q_1, representation_target_k_0, representation_target_k_1, pix_deal, segments):
        try:
          loss1 = self.compute_cross_loss(representation_q_0, representation_target_k_0, pix_deal, segments[0])
        except:
          loss1 = 0
          print("empty segments0")
        try:
          loss2 = self.compute_cross_loss(representation_q_0, representation_target_k_1, pix_deal, segments[1])
        except:
          loss2 = 0
          print("empty segments1")
        try:
          loss3 = self.compute_cross_loss(representation_q_1, representation_target_k_1, pix_deal, segments[2])
        except:
          loss3 = 0
          print("empty segments2")
        try:
          loss4 = self.compute_cross_loss(representation_q_1, representation_target_k_0, pix_deal, segments[3])
        except:
          loss4 = 0
          print("empty segments3")
        
        return loss1, loss2, loss3, loss4



    # def compute_exchange_loss(self, representation_q_0, representation_q_1, exchange_anno):
      


    def forward(self, pcd_q, pcd_k=None, segments=None,  exchange_anno=None, pix_deal=True, mix_loss=False, update_t=True):
        # 传进来的是[xi1,xi2], [xj1,xj2], [S_i1j1, S_i1j2, S_i2j2, S_i2j1]
        
        
        # 先计算四个PCD的Point-wise features
        representation_q_0 = self.model_q(pcd_q[0]) # xi1
        representation_q_1 = self.model_q(pcd_q[1]) # xi2
        
        

        
        if exchange_anno is not None:
          
          exchange_pd_1 =  self.exchange_precdtion(representation_q_0)
          exchange_pd_2 =  self.exchange_precdtion(representation_q_1)

          loss_pd_1 = self.expd_criterion(exchange_pd_1, exchange_anno[0].flatten().long())
          loss_pd_2 = self.expd_criterion(exchange_pd_2, exchange_anno[1].flatten().long())
          loss_pd = loss_pd_1 + loss_pd_2 


        with torch.no_grad():  # no gradient to keys
          if update_t:
            self._momentum_update_target_encoder()  # update the target encoder
          representation_target_k_0 = self.model_k(pcd_k[0])  # xj1
          representation_target_k_1 = self.model_k(pcd_k[1])  # xj2

        lossP2C1, lossP2C2, lossP2C3, lossP2C4 = self.compute_P2C_C2C(representation_q_0, representation_q_1, representation_target_k_0, representation_target_k_1, True, segments)
        lossC2C1, lossC2C2, lossC2C3, lossC2C4 = self.compute_P2C_C2C(representation_q_0, representation_q_1, representation_target_k_0, representation_target_k_1, False, segments)



        if exchange_anno is not None:

          loss_remain = lossP2C1 + lossP2C3 + lossC2C1 + lossC2C3 
          loss_swap   = lossP2C2 + lossP2C4 + lossC2C2 + lossC2C4


          return loss_remain, loss_swap, loss_pd

        else:

          loss_remain = lossP2C1 + lossP2C3 + lossC2C1 + lossC2C3 
          loss_swap   = lossP2C2 + lossP2C4 + lossC2C2 + lossC2C4

          return loss_remain, loss_swap, 0



