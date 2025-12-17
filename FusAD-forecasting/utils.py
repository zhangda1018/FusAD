import torch
import os
import shutil
import inspect
import argparse


def save_copy_of_files(checkpoint_callback):
    """保存当前脚本文件的副本到检查点目录"""
    caller_frame = inspect.currentframe().f_back
    caller_filename = caller_frame.f_globals["__file__"]
    caller_script_path = os.path.abspath(caller_filename)
    destination_directory = checkpoint_callback.dirpath
    os.makedirs(destination_directory, exist_ok=True)
    shutil.copy(caller_script_path, destination_directory)


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def random_masking_3D(xb, mask_ratio):
    """
    随机掩码3D张量，用于预训练任务
    Args:
        xb: [bs x num_patch x dim] 输入张量
        mask_ratio: 掩码比例
    Returns:
        x_masked: 掩码后的张量
        x_kept: 保留的张量
        mask: 二值掩码
        ids_restore: 恢复索引
    """
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(bs, L, device=xb.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, x_kept, mask, ids_restore
