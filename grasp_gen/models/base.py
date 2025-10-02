from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner
import pdb, copy, torch, os
from loguru import logger

MODEL = Registry('Model')
DIFFUSER = Registry('Diffuser')
OPTIMIZER = Registry('Optimizer')
PLANNER = Registry('Planner')

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
            logger.info(f'Load parameter {key} for current model.')
        ## model is trained with ddm
        if 'eps_model.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['eps_model.'+key]
            logger.info(f'Load parameter eps_model.{key} for current model.')
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter module.{key} for current model [Trained on multi GPUs].')
    
    model.load_state_dict(model_state_dict)

def create_model(cfg: DictConfig, device=None, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a generative model and return it.
    If 'diffuser' in cfg, this function will call `create_diffuser` function to create a diffusion model.
    Otherwise, this function will create other generative models, e.g., cvae.

    Args:
        cfg: configuration object, the global configuration
    
    Return:
        A generative model
    """
    if 'diffuser' in cfg:
        return create_diffuser(cfg, device=device, *args, **kwargs)

    return MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)

def create_diffuser(cfg: DictConfig, device=None, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a diffuser model, first create a eps_model from model config,
    then create a diffusion model and use the eps_model as input.

    Args:
        cfg: configuration object
    
    Return:
        A diffusion model
    """
    ## use diffusion model, the model is a eps model
    # eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
    ## if the task has observation, then pass it to diffuser
    # has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False
    
    if cfg.diffuser.name == 'DDPM':
        eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
        has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False
        diffuser  = DIFFUSER.get(cfg.diffuser.name)(eps_model, cfg.diffuser, has_obser, *args, **kwargs)

        ## if optimizer is in cfg, then load it and pass it to diffuser
        if 'optimizer' in cfg:
            optimizer = create_optimizer(cfg.optimizer, device=device, *args, **kwargs)
            diffuser.set_optimizer(optimizer)
        
        ## if planner is in cfg, then load it and pass it to diffuser
        if 'planner' in cfg:
            planner = create_planner(cfg.planner, *args, **kwargs)
            diffuser.set_planner(planner)

        return diffuser
    
    elif cfg.diffuser.name == 'Consistency_model':
        eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
        has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False
        print(DIFFUSER)
        print(MODEL)
        
        eps_model.to(device=device)
        teacher_model = copy.deepcopy(eps_model)
        target_model = copy.deepcopy(eps_model)
        model_diffuser = DIFFUSER.get('DDPM')(eps_model, cfg.diffuser, has_obser, *args, **kwargs)
        teacher_model_diffuser = DIFFUSER.get('DDPM')(teacher_model, cfg.diffuser, has_obser, *args, **kwargs)
        target_model_diffuser = DIFFUSER.get('DDPM')(target_model, cfg.diffuser, has_obser, *args, **kwargs)

        ckpt_path = os.path.join(cfg.load_ckpt_dir, 'model_.pth')
        # pdb.set_trace()
        if not os.path.exists(ckpt_path):
            checkpoint_files = [f for f in os.listdir(cfg.load_ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
            latest_ckpt = max(checkpoint_files, key=lambda f: int(f.split('_')[1].replace('.pth', '')))
            ckpt_path = os.path.join(cfg.load_ckpt_dir, latest_ckpt)
            logger.info(f"Using the latest checkpoint: {ckpt_path}")
        
        load_ckpt(teacher_model_diffuser, path=ckpt_path)
        load_ckpt(model_diffuser, path=ckpt_path)
        load_ckpt(target_model_diffuser, path=ckpt_path)
        # pdb.set_trace()
        # 
        '''
        for dst, src in zip(eps_model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)
        for dst, src in zip(target_model.parameters(), eps_model.parameters()):
            dst.data.copy_(src.data)
        '''
        diffuser = DIFFUSER.get(cfg.diffuser.name)(cfg.diffuser, has_obser, model_diffuser, teacher_model_diffuser, target_model_diffuser, *args, **kwargs)
        
        ## if optimizer is in cfg, then load it and pass it to diffuser
        if 'optimizer' in cfg:
            optimizer = create_optimizer(cfg.optimizer, device=device, *args, **kwargs)
            model_diffuser.set_optimizer(optimizer)
        
        ## if planner is in cfg, then load it and pass it to diffuser
        if 'planner' in cfg:
            planner = create_planner(cfg.planner, *args, **kwargs)
            model_diffuser.set_planner(planner)
        
        return model_diffuser, diffuser
    elif cfg.diffuser.name == 'short_cut':
        eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)
        has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False

        diffuser  = DIFFUSER.get('DDPM')(eps_model, cfg.diffuser, has_obser, *args, **kwargs)
        ## if optimizer is in cfg, then load it and pass it to diffuser
        if 'optimizer' in cfg:
            optimizer = create_optimizer(cfg.optimizer, device=device, *args, **kwargs)
            diffuser.set_optimizer(optimizer)
        
        ## if planner is in cfg, then load it and pass it to diffuser
        if 'planner' in cfg:
            planner = create_planner(cfg.planner, *args, **kwargs)
            diffuser.set_planner(planner)

        return diffuser
    else:
        raise ValueError(f"Unknown diffuser {cfg.diffuser.name}")

def create_optimizer(cfg: DictConfig, device=None, *args: List, **kwargs: Dict) -> Optimizer:
    """ Create a optimizer for constrained sampling

    Args:
        cfg: configuration object
    
    Return:
        A optimizer used for guided sampling
    """
    if cfg is None:
        return None
    
    return OPTIMIZER.get(cfg.name)(cfg, device=device, *args, **kwargs)

def create_planner(cfg: DictConfig, *args: List, **kwargs: Dict) -> Planner:
    """ Create a planner for constrained sampling

    Args:
        cfg: configuration object
        
    Return:
        A planner used for guided sampling
    """
    if cfg is None:
        return None
    
    return PLANNER.get(cfg.name)(cfg, *args, **kwargs)
