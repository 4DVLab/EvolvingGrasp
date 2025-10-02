import os
import sys
sys.path.append(os.getcwd())
import ipdb
import gc
import yaml
import pickle
import argparse
from loguru import logger

from isaacgym import gymapi, gymutil, gymtorch
import torch
import random
import numpy as np

import trimesh as tm
from utils.handmodel import get_handmodel, compute_collision
from envs.tasks.grasp_test_force_shadowhand import IsaacGraspTestForce_shadowhand as IsaacGraspTestForce


def set_global_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test Scripts of Grasp Generation')
    parser.add_argument('--stability_config', type=str,
                        default='envs/tasks/grasp_test_force.yaml',
                        help='stability config file path')
    parser.add_argument('--eval_dir', type=str, required=True, 
                        help='evaluation directory path (e.g.,\
                             "outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/eval/final/2023-04-20_13-06-44")')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='dataset directory path (e.g.,\
                             ("/path/to/MultiDex_UR/")')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed')
    parser.add_argument('--cpu', action='store_true', default=False, 
                        help='run all on cpu')
    parser.add_argument('--onscreen', action='store_true', default=False,
                        help='run simulator onscreen')
    
    return parser.parse_args()


def get_sim_param():
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 0
    return sim_params


def stability_tester(args: argparse.Namespace) -> dict:
    with open(args.stability_config) as f:
        stability_config = yaml.safe_load(f)
    sim_params = get_sim_param()
    sim_headless = not args.onscreen

    # load generated grasp results here
    grasps = pickle.load(open(os.path.join('/inspurfs/group/mayuexin/zym/diffusion+hand/ugg/results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/allours_realdex5/all_hand_param_out.pkl'), 'rb'))
    isaac_env = None
    results = {}
    across_all_cases = 0
    across_all_succ = 0
    results1 = {}
    across_all_cases1 = 0
    across_all_succ1 = 0

    for object_name in grasps['sample_qpos'].keys():
        logger.info(f'Stability test for [{object_name}]')
        q_generated = grasps['sample_qpos'][object_name]
        q_generated = torch.tensor(q_generated, device=args.device).to(torch.float32)

        # load object mesh
        #multidex
        # object_mesh_path = f'./assets/object/{object_name.split("+")[0]}/{object_name.split("+")[1]}/{object_name.split("+")[1]}.stl'
        #realdex
        object_mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/Realdex/meshdata', f'{object_name}.obj')
        #grasp_anyting
        # object_mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/mesh', f'{object_name}.obj')
        #unidex 
        # object_mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/obj_scale_urdf', f'{object_name}.obj')
        #dexgrasp 
        # object_mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGraspNet/obj_scale_urdf', f'{object_name}.obj')
        #DexGRAB
        # object_mesh_path = f'/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes/{object_name}.ply'

        object_mesh = tm.load(object_mesh_path)
        object_volume = object_mesh.volume
        isaac_env = IsaacGraspTestForce(stability_config, sim_params, gymapi.SIM_PHYSX, 
                                        args.device, 0, headless=sim_headless, init_opt_q=q_generated,
                                        object_name=object_name, object_volume=object_volume, fix_object=False)
        succ_grasp_object ,succ_grasp_object1 = isaac_env.push_object()
        results[object_name] = {'total': int(succ_grasp_object.shape[0]),
                                'succ': int(succ_grasp_object.sum()),
                                'case_list': succ_grasp_object.tolist()}
        results1[object_name] = {'total': int(succ_grasp_object1.shape[0]),
                        'succ': int(succ_grasp_object1.sum()),
                        'case_list': succ_grasp_object1.tolist()}
        logger.info(f'all 6dir Success rate of [{object_name}]: {int(succ_grasp_object.sum())} / {int(succ_grasp_object.shape[0])}')
        logger.info(f'one 6dir Success rate of [{object_name}]: {int(succ_grasp_object1.sum())} / {int(succ_grasp_object1.shape[0])}')
        across_all_succ += int(succ_grasp_object.sum())
        across_all_cases += int(succ_grasp_object.shape[0])
        across_all_succ1 += int(succ_grasp_object1.sum())
        across_all_cases1 += int(succ_grasp_object1.shape[0])
        if isaac_env is not None:
            del isaac_env
            gc.collect()
    logger.info(f'**all 6dir Success Rate** across all objects: {across_all_succ} / {across_all_cases}')
    logger.info(f'**one 6dir Success Rate** across all objects: {across_all_succ1} / {across_all_cases1}')
    
    return results


def diversity_tester(args: argparse.Namespace, stability_results: dict) -> None:    
    grasps = pickle.load(open(os.path.join('/inspurfs/group/mayuexin/zym/diffusion+hand/ugg/results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/allours_realdex5/all_hand_param_out.pkl'), 'rb'))

    qpos_std = []
    for object_name in grasps['sample_qpos'].keys():
        i_qpos = grasps['sample_qpos'][object_name][:, 9:]
        i_qpos = i_qpos[stability_results[object_name]['case_list'], :]
        if i_qpos.shape[0]:
            i_qpos = np.sqrt(i_qpos.var(axis=0))
            qpos_std.append(i_qpos)

    qpos_std = np.stack(qpos_std, axis=0)
    qpos_std = qpos_std.mean(axis=0).mean()
    logger.info(f'**Diversity** (std: rad.) across all success grasps: {qpos_std}')


def collision_tester(args: argparse.Namespace, stability_results: dict) -> None:
    _BATCHSIZE = 8 #NOTE: adjust this batchsize to fit your GPU memory && need to be divided by generated grasps per object
    _NPOINTS = 4096 #NOTE: number of surface points sampled from a object

    grasps = pickle.load(open(os.path.join('/inspurfs/group/mayuexin/zym/diffusion+hand/ugg/results/ugg_generation/ugg_bs8x256_lr0002_l2_cosine/allours_realdex5/all_hand_param_out.pkl'), 'rb'))
    #multi
    # obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/zym/diffusion+hand/Scene-Diffuser/MultiDex_UR/object_pcds_nors.pkl', 'rb'))
    #realdex
    obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/datasets/Realdex/object_pcds_nors.pkl', 'rb'))
    #grasp_anyting
    # obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/datasets/scaled_PUnetdata/object_pcds_nors.pkl', 'rb'))
    #unidex
    # obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/object_pcds_nors.pkl', 'rb'))
    #dexgrasp
    # obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/datasets/DexGraspNet/object_pcds_nors.pkl', 'rb'))
    #DexGRAB
    # obj_pcds_nors_dict = pickle.load(open('/inspurfs/group/mayuexin/datasets/DexGRAB/object_pcds_nors.pkl', 'rb'))
    
    hand_model = get_handmodel(batch_size=_BATCHSIZE, device=args.device)

    collisions_dict = {obj: [] for obj in grasps['sample_qpos'].keys()}
    collisions_dict2 = {obj: [] for obj in grasps['sample_qpos'].keys()}
    collisions_dict1 = []
    for object_name in grasps['sample_qpos'].keys():
        qpos = grasps['sample_qpos'][object_name]
        obj_pcd_nor = obj_pcds_nors_dict[object_name][:_NPOINTS, :]
        for i in range(qpos.shape[0] // _BATCHSIZE):
            i_qpos = qpos[i * _BATCHSIZE: (i + 1) * _BATCHSIZE, :]
            hand_model.update_kinematics(q=torch.tensor(i_qpos, device=args.device))
            hand_surface_points = hand_model.get_surface_points()
            #TODO: needed to be checked
            depth_collision = compute_collision(torch.tensor(obj_pcd_nor, device=args.device), hand_surface_points)
            collisions_dict[object_name].append(np.array(depth_collision.cpu()[stability_results[object_name]['case_list'][i * _BATCHSIZE : (i + 1) * _BATCHSIZE]]))
            collisions_dict2[object_name].append(np.array(depth_collision.cpu()))
            #TODO: needed to be checked
            pen_loss_sdf_value = hand_model.pen_loss_sdf(torch.tensor(obj_pcd_nor[:, :3], device=args.device), q= torch.tensor(i_qpos, device=args.device), test = True)
        collisions_dict[object_name] = np.concatenate(collisions_dict[object_name], axis=0)
        collisions_dict2[object_name] = np.concatenate(collisions_dict2[object_name], axis=0)
        collisions_dict1.append(pen_loss_sdf_value)
    collisions_means = [tensor.cpu().numpy() for tensor in collisions_dict1]
    collision_values = np.concatenate([collisions_dict[object_name] for object_name in grasps['sample_qpos'].keys()], axis=0)
    collision_values2 = np.concatenate([collisions_dict2[object_name] for object_name in grasps['sample_qpos'].keys()], axis=0)
    logger.info(f'**Collision** (depth: mm.) across succ grasps: {collision_values.mean() * 1e3}')
    logger.info(f'**Collision** (depth: mm.) across all grasps: {collision_values2.mean() * 1e3}')
    logger.info(f'**Collision unidex** (depth: cm.) across all grasps: {np.mean(collisions_means)}')





def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    args.device = 'cpu' if args.cpu else 'cuda'

    logger.add(args.eval_dir + '/evaluation.log')
    logger.info(f'Evaluation directory: {args.eval_dir}')

    logger.info('Start evaluating..')

    stability_results = stability_tester(args)
    diversity_tester(args, stability_results)
    collision_tester(args, stability_results)
    
    logger.info('End evaluating..')


if __name__ == '__main__':
    main()
