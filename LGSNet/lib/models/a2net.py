import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
INF = 1e8

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class TemporalMultiAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout=0., dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.heads = heads
        # self.scale = dim_head ** -0.25
        self.scale = 1
        self.attn_drop = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
            ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        q = k = v = x
        qkv = torch.cat([q,k,v],dim=-1).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

############### Postbackbone ##############
class BaseFeatureNet(nn.Module):
    '''
    Calculate basic feature
    PreBackbobn -> Backbone
    CAS(ME)^2:
    input: [batch_size, 2048, 64]
    output: [batch_size, 512, 16]
    SAMM:
    input: [batch_size, 2048, 256]
    output: [batch_size, 512, 64]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.dataset = cfg.DATASET.DATASET_NAME
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)     
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mish = Mish()

    def forward(self, x):
        feat1 = self.mish(self.conv1(x))
        feat1 = self.max_pooling(feat1)
        feat2 = self.mish(self.conv2(feat1))
        feat2 = self.max_pooling(feat2)
        return feat1, feat2


############### Neck ##############
class FeatNet(nn.Module):
    '''
    Main network
    Backbone -> Neck
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.inhibition = cfg.MODEL.RECEPTIVE_FUSION
        self.base_feature_net = BaseFeatureNet(cfg)
        self.convs = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer - 1]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]
            conv = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[layer], padding=1), Mish())
            self.convs.append(conv)
        for i in range(len(self.inhibition)):
            self.fuse.append(Inhibit_Fuse(cfg, self.inhibition[i]))

    def forward(self, x):
        results = []
        feat_base, feat = self.base_feature_net(x)
        for conv in self.convs:
            feat = conv(feat)
            results.append(feat)
        j = 0
        for i in range(len(results)):  
            if results[i].size(-1) in self.inhibition:
                fuse = self.fuse[j]
                results[i] = fuse(results[i], feat_base)
                feat_base = results[i]
                j = j + 1
        return tuple(results)


class Inhibit_Fuse(nn.Module):
    def __init__(self, cfg, inhibit):
        super(Inhibit_Fuse, self).__init__()
        self.enhance = nn.Sequential(Conv_branch(cfg.MODEL.BASE_FEAT_DIM, inhibit, 1), nn.GroupNorm(8, cfg.MODEL.BASE_FEAT_DIM))
        self.base = nn.Sequential(Conv_branch(cfg.MODEL.BASE_FEAT_DIM, 2 * inhibit ,1, Down=True), nn.GroupNorm(8, cfg.MODEL.BASE_FEAT_DIM))
        self.fuse = SeFuse(2*cfg.MODEL.BASE_FEAT_DIM, inhibit, Concate=True)

    def forward(self, x, base):
        feat_enhance = self.enhance(x)
        feat_base = self.base(base)
        out = torch.cat((feat_base, feat_enhance), dim=1)
        return self.fuse(out)


class SSE(nn.Module):
    def __init__(self, gn_c, se_channel, down=False):
        super(SSE, self).__init__()
        out_c = se_channel//2 if down else se_channel
        self.bn = nn.GroupNorm(8, gn_c)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv2 = nn.Sequential(nn.Conv1d(se_channel, out_c, kernel_size=1, stride=1), nn.Sigmoid())
    
    def forward(self, x, y = None):
        x = x if y is not None else self.bn(x)
        feat = self.pool(x.permute(0, 2, 1))
        feat = self.conv2(feat).permute(0, 2, 1)
        out =  y * feat.expand_as(y) if y is not None else  x * feat.expand_as(x)
        return out


class Newpool(nn.Module):
    def __init__(self, kenerl_shape, strides, pad):
        super(Newpool, self).__init__()
        self.ks = kenerl_shape
        self.stride = strides
        self.pad = pad
        self.pool = nn.MaxPool1d(kernel_size=self.ks, stride=self.stride)

    def compute_pad(self, tem):
        tmp = (tem - self.ks + 2*self.pad) % self.stride + 1
        if tmp == tem:
            return 0, 0
        else:
            return 0, 1

    def forward(self, x):
        l, r = self.compute_pad(x.size(-1))
        x = F.pad(x, (l,r))
        return self.pool(x)
        

class SeFuse(nn.Module):
    def __init__(self, in_c, se_channel, Down = False, Concate = False):
        super(SeFuse, self).__init__()
        self.down = Down
        self.cat = Concate
        stride = 2 if self.down else 1
        out_c = in_c//2 if self.cat else in_c
        self.conv3 = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1), nn.GroupNorm(8, out_c))
        self.se = SSE(in_c, se_channel, self.down)
        self.pool1 = Newpool(kenerl_shape=2, strides=stride, pad=0)
        self.conv1 = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, out_c))
        self.act = Mish()
    
    def forward(self, x):
        feat_3 = self.conv3(x)
        out = self.act(self.se(x, feat_3 + self.conv1(self.pool1(x)))) if self.down or self.cat else self.act(feat_3 + self.conv1(x) + self.se(x))
        return out


class Conv_branch(nn.Module):
    def __init__(self, in_c, se_channel, t, Down=False):
        super(Conv_branch, self).__init__()
        self.cb = nn.ModuleList()
        for _ in range(t):
            self.cb.append(SeFuse(in_c, se_channel))
        if Down:
            self.cb.append(SeFuse(in_c, se_channel, Down))
        
    def forward(self, x):
        feat = x
        for se in self.cb:
            feat = se(feat)
        return feat


# Postbackbone -> Neck
class GlobalLocalBlock(nn.Module):
    def __init__(self, cfg):
        super(GlobalLocalBlock, self).__init__()
        self.dim_in = cfg.MODEL.REDU_CHA_DIM
        self.dim_out = cfg.MODEL.REDU_CHA_DIM
        self.ws = cfg.DATASET.WINDOW_SIZE
        self.drop_threshold = cfg.MODEL.DROP_THRESHOLD
        self.ss = cfg.DATASET.SAMPLE_STRIDE
        self.mish = Mish()
        
        self.theta = nn.Conv1d(self.dim_in, self.dim_out, kernel_size=1, stride=1)
        self.phi = nn.Conv1d(self.dim_in, self.dim_out, kernel_size=1, stride=1)
        self.g = nn.Conv1d(self.dim_in, self.dim_out, kernel_size=1, stride=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.wl = nn.Conv1d(1, 1, kernel_size=1)
        self.act = nn.Sigmoid()
        nn.init.constant_(self.wl.weight, 0)
        nn.init.constant_(self.wl.bias, 0)
        self.wg = nn.Conv1d(self.dim_in, self.dim_in, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.wg.weight, 0)
        nn.init.constant_(self.wg.bias, 0)
        
        #Fuse
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2) 

        # MLP
        self.conv1 = nn.Linear(self.dim_out, 4*self.dim_out)
        self.ln1 = nn.LayerNorm(self.dim_out)
        self.conv2 = nn.Linear(4*self.dim_out, self.dim_out)
        self.ln2 = nn.LayerNorm(self.dim_out)

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        channels = x.shape[1]
        ori_length = x.shape[2]
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)
        length_temp = ori_length//4
        # local channel enhancer
        all_tmp = torch.zeros([ori_length, batch_size, channels, length_temp]).cuda()
        all_temp_g = all_tmp
        for j in range(theta.size(2)):
            # Sometimes
            # temp1: BS * Channels * length_temp(temporal neighbour, e.g. 4)
            temp = torch.zeros([batch_size, channels, length_temp]).cuda() # BS*channels*length_temp
            temp_g = temp
            if j < length_temp//2:
                temp[:,:,length_temp//2-j:] = theta[:,:,:j+length_temp//2]
                temp_g[:,:,length_temp//2-j:] = g[:,:,:j+length_temp//2]
            elif length_temp//2 <= j <= theta.size(2)-length_temp//2:
                if length_temp%2 == 0:
                    temp = theta[:,:,j-length_temp//2:j+length_temp//2]
                    temp_g= g[:,:,j-length_temp//2:j+length_temp//2]
                else:
                    temp= theta[:,:,j-length_temp//2:j-length_temp//2+1]
                    temp_g= g[:,:,j-length_temp//2:j+length_temp//2+1]
            else:
                temp[:,:,:length_temp-(j%length_temp-length_temp//2)] = theta[:,:,j-length_temp//2:]
                temp_g[:,:,:length_temp-(j%length_temp-length_temp//2)] = g[:,:,j-length_temp//2:]
            all_tmp[j:j+1,:,:,:]= temp
            all_temp_g[j:j+1,:,:,:] = temp_g
        all_tmp_phi = phi.permute(0,2,1).unsqueeze(dim=2)
        local_theta_phi = torch.matmul(all_tmp_phi, all_tmp.permute(1, 0, 2, 3))
        local_theta_phi_sc = local_theta_phi * (channels**-.5)
        local_p = F.softmax(local_theta_phi_sc, dim=-1)  
        local_p = local_p.expand(-1, -1, channels, -1)
        # local_p = torch.where(local_p > torch.tensor(self.drop_threshold).float().cuda(), local_p, torch.tensor(0).float().cuda())
        all_temp_g = all_temp_g.permute(1, 0, 2, 3)
        local_temp = torch.sum(self.drop1(local_p) * all_temp_g, dim=-1) 
        local_temp = self.pool(local_temp) # BS * length * channel => BS * length * 1
        local_temp = self.wl(local_temp.permute(0, 2, 1)) # BS * 1 * length => BS * 1 * length
        local_temp = self.act(local_temp) 

        # global channel enhancer
        # e.g. (BS, 16, 512) * (BS, 512, 16) => (BS, 16, 16)
        global_theta_phi = torch.bmm(torch.transpose(phi, 2, 1), theta)
        global_theta_phi_sc = global_theta_phi * (channels**-.5)
        global_p = F.softmax(global_theta_phi_sc, dim=-1)
        global_temp = torch.bmm(self.drop1(global_p), g.permute(0, 2, 1))
        global_temp = global_temp.permute(0, 2, 1) 
        global_temp = self.wg(global_temp)

        # local&global fusion
        local_global = global_temp * local_temp.expand_as(global_temp) 
        
        # MLP
        local_global = local_global.permute(2, 0, 1)
        residual = residual.permute(2, 0, 1)
        out_temp_ln = self.ln1(self.drop2(local_global) + residual)
        out_mlp_act = self.drop2(self.mish(self.drop2(self.conv1(out_temp_ln))))
        out_mlp_conv2 = self.conv2(out_mlp_act)
        out = self.ln2(self.drop2(out_mlp_conv2) + out_temp_ln)
        out = out.permute(1, 2, 0)
        return out


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim_out = cfg.MODEL.REDU_CHA_DIM
        self.drop = nn.Dropout(0.2) 
        self.conv1 = nn.Linear(self.dim_out, 4*self.dim_out, bias=False)
        self.ln1 = nn.LayerNorm(self.dim_out)
        self.conv2 = nn.Linear(4*self.dim_out, self.dim_out, bias=False)
        self.ln2 = nn.LayerNorm(self.dim_out)
        self.mish = nn.GELU()
    
    def forward(self, global_temp, residual):
        global_temp =  global_temp.permute(2, 0, 1)
        residual = residual.permute(2, 0, 1)
        out_temp_ln = self.ln1(self.drop(global_temp) + residual)
        out_mlp_act = self.drop(self.mish(self.drop(self.conv1(out_temp_ln))))
        out_mlp_conv2 = self.conv2(out_mlp_act)
        out = self.ln2(self.drop(out_mlp_conv2) + out_temp_ln)
        out = out.permute(1, 2, 0)
        return out

class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim_in = cfg.MODEL.REDU_CHA_DIM
        self.dim_out = cfg.MODEL.REDU_CHA_DIM
        self.heads = cfg.MODEL.HEADS
        self.att = TemporalMultiAttention(self.dim_in, self.heads, self.dim_in//self.heads, 0.1, 0.1)
        self.mlp = MLP(cfg)
    
    def forward(self, x):
        global_temp = (self.att(x.permute(0, 2, 1))).permute(0, 2, 1)
        out = self.mlp(global_temp, x)
        return out

############### Postneck ##############
class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)
            self.convs.append(conv)
        # self.relu = nn.ReLU(inplace=True)
        self.mish = Mish()

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for conv, feat in zip(self.convs, feat_list):
           results.append(self.mish(conv(feat)))
           # results.append(self.relu(conv(feat)))
        return tuple(results)

############### Head ##############
class PredHeadBranch(nn.Module):
    '''
    From ReduceChannel Module
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    output: Channels reduced into 256
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    output: Channels reduced into 256
    '''
    def __init__(self, cfg):
        super(PredHeadBranch, self).__init__()
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS  # 2
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.convs.append(conv)
        self.mish = Mish()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = x
        for conv in self.convs:
            feat = self.mish(conv(feat))
            # feat = self.relu(conv(feat))
        return feat


############### Details of Prediction ##############
class PredHead(nn.Module):
    '''
    CAS(ME)^2:
    input: [batch_size, 512, (16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    SAMM:
    input: [batch_size, 512, (128,64,32,16,8,4,2)]
    input_tmp: to PredHeadBranch Module
    output: Channels reduced into number of classes or boundaries
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.head_branches = nn.ModuleList()
        self.lgf = GlobalLocalBlock(cfg)
        self.ge = nn.ModuleList()
        self.enhanment = cfg.MODEL.GLOBAL_ENHANCEMENT
        for _ in range(len(self.enhanment)):
            self.ge.append(GlobalMultiHeadAttention(cfg))
        self.inhibition = cfg.MODEL.INHIBITION_INTERVAL
        for _ in range(4):
            self.head_branches.append(PredHeadBranch(cfg))
        num_class = cfg.DATASET.NUM_CLASSES  # 2
        num_box = len(cfg.MODEL.ASPECT_RATIOS)  # 5

        # [batch_size, 256, (16,8,4,2)] -> [batch_size, _, (16,8,4,2)]
        af_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_class, kernel_size=3, padding=1)
        af_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, 2, kernel_size=3, padding=1)
        ab_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * num_class, kernel_size=3, padding=1)
        ab_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * 2, kernel_size=3, padding=1)
        self.pred_heads = nn.ModuleList([af_cls, af_reg, ab_cls, ab_reg])

    def forward(self, x):
        preds = []
        lgf_out = x
        if lgf_out.size(-1) in self.inhibition:
            lgf_out = self.lgf(lgf_out)
        elif lgf_out.size(-1) in self.enhanment:
            lgf_out = self.ge[self.enhanment.index(x.size(-1))](lgf_out)
        for pred_branch, pred_head in zip(self.head_branches, self.pred_heads):
            feat = pred_branch(lgf_out)
            preds.append(pred_head(feat))
        # print(list(self.pred_heads[0].parameters())[0][0][0])
        return tuple(preds)


############### Prediction ##############
class LocNet(nn.Module):
    '''
    Predict expression boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.ab_pred_value = cfg.DATASET.NUM_CLASSES + 2

    def _layer_cal(self, feat_list):
        af_cls = list()
        af_reg = list()
        ab_pred = list()
        for feat in feat_list:
            cls_af, reg_af, cls_ab, reg_ab = self.pred(feat)
            af_cls.append(cls_af.permute(0, 2, 1).contiguous())
            af_reg.append(reg_af.permute(0, 2, 1).contiguous())
            ab_pred.append(self.tensor_view(cls_ab, reg_ab))

        af_cls = torch.cat(af_cls, dim=1)  # bs, sum(t_i), n_class+1
        af_reg = torch.cat(af_reg, dim=1)  # bs, sum(t_i), 2
        af_reg = F.relu(af_reg)
        return (af_cls, af_reg), tuple(ab_pred)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.ab_pred_value)
        return data

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)
        return self._layer_cal(features_list)


############### All processing ##############
class A2Net(nn.Module):
    def __init__(self, cfg):
        super(A2Net, self).__init__()
        self.features = FeatNet(cfg)
        self.loc_net = LocNet(cfg)

    def forward(self, x):
        features = self.features(x)
        out_af, out_ab = self.loc_net(features)
        return out_af, out_ab

if __name__ == '__main__':
    import sys
    sys.path.append('/home/yww/1_spot/MSA-Net/lib')
    from config import cfg, update_config
    cfg_file = '/home/yww/1_spot/MSA-Net/experiments/samm.yaml'
    update_config(cfg_file)

    model = A2Net(cfg).cuda()
    data = torch.randn((8, 2048, 64)).cuda()
    output = model(data)
