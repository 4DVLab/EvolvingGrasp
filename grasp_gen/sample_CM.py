import os
import hydra
from scripts.grasp_gen_ur.test_copy import evaluation_grasp, evaluation_grasp_all
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import pdb
from utils.misc import timestamp_str, compute_model_dim
from utils.io import mkdir_if_not_exists
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer
from models.visualizer_val import create_visualizer_val
from models.visualizer_CM import create_visualizer_CM
import peft
from models.dm.schedule import make_schedule_ddpm

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()
    
    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            # logger.info(f'Load parameter {key} for current model.')
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            # logger.info(f'Load parameter {key} for current model [Trained on multi GPUs].')
    
    model.load_state_dict(model_state_dict)
    for k, v in make_schedule_ddpm(model.timesteps, **model.schedule_cfg).items():
        model.register_buffer(k, v)


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
    
    ## set output dir
    eval_dir = os.path.join(cfg.exp_dir, 'eval')
    mkdir_if_not_exists(eval_dir)
    vis_dir = os.path.join(eval_dir, 
        'series' if cfg.task.visualizer.vis_denoising else 'final', timestamp_str())

    logger.add(vis_dir + '/sample.log') # set logger file
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg)) # record configuration

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    ## prepare dataset for visual evaluation
    ## only load scene
    # import pdb; pdb.set_trace()
    datasets = {
        'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True),
    }
    for subset, dataset in datasets.items():
        logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    # pdb.set_trace()
    dataloaders = {
        'test': datasets['test'].get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    }
    
    ## create model and load ckpt
    # cfg.save_RL_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/ckpts_RL_epoch/multidex_N_ft_10/4/'             ## DPO
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/DDPM_pre_all_para/multidex/ckpts'
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-01-16_23-00-35_/ckpts'         ## multidex
    cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/outputs/grasp_generation/train/2025-02-07_20-15-16_/ckpts'           ## multidex + guidance_loss
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-02-05_22-58-04_/ckpts'         ## realdex
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-02-06_22-39-13_/ckpts'           ## realdex + guidance_loss
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-02-16_20-47-13_/ckpts'           ## dexgrab + guidance_loss
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-02-17_18-03-12_/ckpts'             ## dexgraspnet + guidance_loss
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-02-28_16-56-19_/ckpts'             ## multidex (degraded) + guidance_loss
    # cfg.load_ckpt_dir = '/inspurfs/group/mayuexin/zhuyufei/graps_gen/outputs/grasp_generation/train/2025-03-02_15-20-09_/ckpts'             ## multidex (degraded ++) + guidance_loss
    # logger.info(f'training on dexgrab dataset, testing on dexgraspnet.')
    cfg.diffuser.name = 'Consistency_model'
    model, diffuser = create_model(cfg, slurm=cfg.slurm, device=device)
    
    ## if your models are seperately saved in each epoch, you need to change the model path manually
    
    ## train
    # pdb.set_trace()
    # ckpt_path = os.path.join(cfg.ckpt_dir, 'model_.pth')
    # if not os.path.exists(ckpt_path):
    #     checkpoint_files = [f for f in os.listdir(cfg.ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
    #     latest_ckpt = max(checkpoint_files, key=lambda f: int(f.split('_')[1].replace('.pth', '')))
    #     ckpt_path = os.path.join(cfg.ckpt_dir, latest_ckpt)
    #     logger.info(f"Using the latest checkpoint: {ckpt_path}")
    # load_ckpt(model, path=ckpt_path)
    
    peft_config = peft.LoraConfig( 
        r=cfg.lora_rank, 
        target_modules=["to_q", "to_k", "to_v", "linear_q", "linear_k", "linear_v"], 
        lora_dropout=0.01, )
    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device)

    ## create visualizer and visualize
    # train
    # visualizer = create_visualizer(cfg.task.visualizer)
    # visualizer.visualize(cfg, model, dataloaders['test'], vis_dir, evaluation_grasp=evaluation_grasp, evaluation_grasp_all=evaluation_grasp_all)
    
    # visualizer_1 = create_visualizer_1(cfg.task.visualizer)
    # visualizer_1.visualize(cfg, model, dataloaders['test'], vis_dir, evaluation_grasp=evaluation_grasp)

    # visualizer_CM = create_visualizer_CM(cfg.task.visualizer)
    # visualizer_CM.visualize(cfg, model, dataloaders['test'], vis_dir, evaluation_grasp=evaluation_grasp)

    # val
    # peft.set_peft_model_state_dict(model, torch.load(cfg.save_RL_dir + '/model_RL_4.pth')['model'])
    # import pdb; pdb.set_trace()
    visualizer_val = create_visualizer_val(cfg.task.visualizer)
    visualizer_val.visualize(cfg, model, dataloaders['test'], vis_dir, evaluation_grasp=evaluation_grasp, evaluation_grasp_all=evaluation_grasp_all)
    # print('cosmetics')

if __name__ == '__main__':
    # set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()