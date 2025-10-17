_base_ = './pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'
train_dataloader = dict(batch_size=1, num_workers=1)
# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=4096.)
