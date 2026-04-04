"""
This script will bake poses from the MOYO dataset into source/target 
mesh pairs for the reposing experiment. A subset of poses from MOYO
are sampled using a farthest point algorithm to filter extremely
similar poses. The source/target pairs are created using randomly
sampled body shape parameters for geometric diversity. Note that
we delete geometry associated with the eyes and eyeball classes
to avoid disconnected geometry.

The output PyTorch dict contains:
{
    'src_verts': (N, V, 3)  -- source vertices
    'src_poses': (N, d)     -- source poses parameters
    'src_betas': (N, 10)    -- source body shape parameters
    'tar_verts': (N, V, 3)  -- target vertices
    'tar_poses': (N, d)     -- target poses parameters
    'tar_betas': (N, 10)    -- target body shape parameters
    'faces': (F, 3)         -- SMPLX face indices (shared across all shapes)
    'genders': (N,)         -- gender of each shape [0, 1, 2] -> [neutral, male, female]
    'body_shape_std': float -- standard deviation used for sampling body shape parameters
}

Please see /smplx_data/README.md for setup instructions.
"""

import os
import tqdm
import json
import smplx
import torch
import random
import trimesh
import numpy as np

from utils import smplx_breakdown, farthest_point_subset, remove_parts_from_mesh

import argparse
parser = argparse.ArgumentParser()
# To replicate original paper, use these defaults:
parser.add_argument('--train_samples', type=int, default=32000, help='Number of samples to generate for training set')
parser.add_argument('--test_samples', type=int, default=2000, help='Number of samples to generate for test set')
parser.add_argument('--num_dupes', type=int, default=1, help='Duplicates each source/target pair, each with different body shape parameters')
parser.add_argument('--body_shape_std', type=float, default=5.0, help='Standard deviation of sampled body shape parameters: N(0, body_shape_std^2)')
parser.add_argument('--moyo_dir', type=str, default='./smplx_data/moyo/data', help='Location of MOYO dataset')
parser.add_argument('--smplx_dir', type=str, default='./smplx_data/models', help='Location of SMPLX models (npz/pkl files)')
args = parser.parse_args()

# If needed, changes these paths:
MOYO_DIR = args.moyo_dir # location of MOYO dataset
SMPLX_DIR = args.smplx_dir   # location of SMPLX models (npz/pkl files)

USE_HANDS = True    # if false hand/finger poses will be zero-out (making them neutral)
device = 'cuda'     # device used to run SMPLX model

DATA_DIR = os.path.join(MOYO_DIR, 'mosh')
SMPLX_MODEL_PATH = SMPLX_DIR
MOYO_V_TEMPLATE = os.path.join(MOYO_DIR, 'v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply')

# Two stage process:

# First, load the dataset and filter it through farthest_point_subset (removing redundant samples)
# this gives a list of (diverse) samples that we'll use for the baked dataset.

# Then, re-create the dataset with randomly sampled body shape parameters (betas)
# for geometric diversity. Sample betas from truncated normal N^d(0,1)
def bake_dataset(train, num_samples=None, num_dupes=1, body_shape_std=1.0):
    dir_suffix = 'train' if train else 'val'
    print(f'--- Baking {dir_suffix} dataset ---')
    data_dir = os.path.join(MOYO_DIR, 'mosh', dir_suffix)
    all_verts, faces, all_poses, all_lhand, all_rhand = load_moyo_smplx(data_dir, sub_sample_rate=10, middle=True)
    all_verts, faces, all_poses, all_lhand, all_rhand = all_verts.cpu(), faces.cpu(), all_poses.cpu(), all_lhand.cpu(), all_rhand.cpu()
    if not USE_HANDS:
        all_lhand = all_lhand * 0.0
        all_rhand = all_rhand * 0.0
    
    print('Loaded {} poses'.format(len(all_verts)))
    keep_idx = farthest_point_subset(all_verts, max_samples=None, threshold=0.5)
    print('Filtered to {} poses'.format(len(keep_idx)))
    all_poses = all_poses[keep_idx]
    all_lhand = all_lhand[keep_idx]
    all_rhand = all_rhand[keep_idx]
    print(all_verts.shape, all_poses.shape, all_lhand.shape, all_rhand.shape)

    # random pairings of (source, target) poses:
    N = len(all_poses)
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j] # all ordered pairs (i, j) for i != j.
    if num_samples is not None:
        # shuffle then select `num_samples` pairs
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:num_samples]
    # repeat each pair `num_dupes` times to increase the dataset size -- each will have different body shapes
    pairs = [(i, j) for i, j in all_pairs for _ in range(num_dupes)]
    pairs = torch.tensor(pairs, dtype=torch.long) # (N, 2) -- N pairs of (source, target) indices
    pose_pairs = all_poses[pairs] # (N, 2, d) -- N pairs of (source, target) poses
    lhand_pairs = all_lhand[pairs]
    rhand_pairs = all_rhand[pairs]
    
    # generate actual geometry:
    with open('./smplx_data/smplx_vertex_segmentation.json', 'r') as f:
        smplx_vertex_segmentation = json.load(f) # dict of (body_part, list of vertices)
        smplx_vertex_segmentation = {k: torch.tensor(v, dtype=torch.long).tolist() for k, v in smplx_vertex_segmentation.items()}

    template_mesh = trimesh.load(MOYO_V_TEMPLATE, process=False)
    v_temp, f_temp = template_mesh.vertices, template_mesh.faces
    v_temp = torch.from_numpy(v_temp).float()
    f_temp = torch.from_numpy(f_temp).long()

    # Remove eyeballs from template mesh, and reuse `keep_mask` later:
    v_temp_noeyes, f_temp_noeyes, keep_mask = remove_parts_from_mesh(v_temp, f_temp, smplx_vertex_segmentation)
    f_temp_noeyes = f_temp_noeyes.to('cpu')
    genders = ['neutral', 'male', 'female']

    baked_src_verts = []
    baked_src_feats = []
    baked_src_betas = []
    baked_tar_verts = []
    baked_tar_feats = []
    baked_tar_betas = []
    genders_list = []
    print(f'Baking {len(pose_pairs)} pairs...')
    for (pose_pair, lhand_pair, rhand_pair) in tqdm.tqdm(zip(pose_pairs, lhand_pairs, rhand_pairs), total=len(pose_pairs)):
        gender = random.choice(genders)
        body_model_params = dict(model_path=SMPLX_MODEL_PATH,
                                    model_type='smplx',
                                    gender=gender,
                                    v_template=v_temp,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    num_betas=10,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    use_pca=False,
                                    flat_hand_mean=True,
                                    dtype=torch.float32)

        body_model_params['batch_size'] = 2
        body_model = smplx.create(**body_model_params).to(device)

        # ensure canonical orientation and translation:
        global_orient = None
        trans = None

        betas = torch.randn(1, 10).to(device) * body_shape_std
        betas = torch.clamp(betas, -3, 3)
        betas = betas.repeat(2, 1) # same betas for source and target
        
        body_model_output = body_model(
                            betas=betas.to(device),
                            transl=trans,
                            global_orient=global_orient,
                            body_pose=pose_pair.to(device),
                            left_hand_pose=lhand_pair.to(device),
                            right_hand_pose=rhand_pair.to(device))

        src_tar_verts = body_model_output.vertices.detach().cpu()
        src_tar_verts = src_tar_verts[:, keep_mask]

        baked_src_verts += [src_tar_verts[0].cpu()]
        baked_src_feats += [torch.cat([pose_pair[0], lhand_pair[0], rhand_pair[0]], dim=-1).cpu()] # features corrs. to source shape
        baked_src_betas += [betas[0].cpu()] # betas corrs. to source shape
        baked_tar_verts += [src_tar_verts[1].cpu()]
        baked_tar_feats += [torch.cat([pose_pair[1], lhand_pair[1], rhand_pair[1]], dim=-1).cpu()] # features corrs. to target shape
        baked_tar_betas += [betas[1].cpu()] # betas corrs. to target shape
        genders_list += [{'neutral': 0, 'male': 1, 'female': 2}[gender]]

    baked_src_verts = torch.stack(baked_src_verts, dim=0)
    baked_src_feats = torch.stack(baked_src_feats, dim=0)
    baked_src_betas = torch.stack(baked_src_betas, dim=0)
    baked_tar_verts = torch.stack(baked_tar_verts, dim=0)
    baked_tar_feats = torch.stack(baked_tar_feats, dim=0)
    baked_tar_betas = torch.stack(baked_tar_betas, dim=0)
    genders_list = torch.tensor(genders_list, dtype=torch.int64) # FIXED.

    # Export to torch files:
    hand_suffix = 'hands' if USE_HANDS else 'noHands'
    # out_file = os.path.join(MOYO_DIR, 'mosh_baked', f"{dir_suffix}_baked_{hand_suffix}_{len(baked_src_verts)}_{body_shape_std}.pt")
    out_file = os.path.join(f'./smplx_data/{dir_suffix}_baked_{hand_suffix}_{len(baked_src_verts)}_{body_shape_std}.pt')
    torch.save({
        'src_verts': baked_src_verts,
        'src_poses': baked_src_feats,
        'src_betas': baked_src_betas,
        'tar_verts': baked_tar_verts,
        'tar_poses': baked_tar_feats,
        'tar_betas': baked_tar_betas,
        'faces': f_temp_noeyes,
        'body_shape_std': body_shape_std,
        'genders': genders_list
        }, out_file)
    print('Saved baked dataset to', out_file)
    print('verts:', baked_src_verts.shape, baked_tar_verts.shape, 'poses:', baked_src_feats.shape, baked_tar_feats.shape, 'betas:', baked_src_betas.shape, baked_tar_betas.shape)

def load_moyo_smplx(data_dir, sub_sample_rate=1, middle=False):
    with open('./smplx_data/smplx_vertex_segmentation.json', 'r') as f:
        smplx_vertex_segmentation = json.load(f) # dict of (body part -> list of vertices)
        smplx_vertex_segmentation = {k: torch.tensor(v, dtype=torch.long).tolist() for k, v in smplx_vertex_segmentation.items()}

    template_mesh = trimesh.load(MOYO_V_TEMPLATE, process=False)
    v_temp, f_temp = template_mesh.vertices, template_mesh.faces
    v_temp = torch.from_numpy(v_temp).float()
    f_temp = torch.from_numpy(f_temp).long()

    v_temp_noeyes, f_temp_noeyes, keep_mask = remove_parts_from_mesh(v_temp, f_temp, smplx_vertex_segmentation)
    v_temp_noeyes = v_temp_noeyes.to(device)
    f_temp_noeyes = f_temp_noeyes.to(device)

    body_model_params = dict(model_path=SMPLX_MODEL_PATH,
                                model_type='smplx',
                                gender='neutral',
                                v_template=v_temp,
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                num_betas=10,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=True,
                                use_pca=False,
                                flat_hand_mean=True,
                                dtype=torch.float32)

    list_dir = os.listdir(data_dir)
    criteria = lambda x: x.endswith('.pkl') and (os.path.basename(x).startswith('220923') or os.path.basename(x).startswith('220926'))
    list_dir = sorted(list(filter(criteria, list_dir)))
    print(f'Found {len(list_dir)} sequence files in {data_dir}')

    loaded_verts = []
    body_poses = []
    left_hand_poses = []
    right_hand_poses = []
    nloaded = 0
    for i, file in enumerate(list_dir):
        if (i + 1) % 10 == 0 or i == 0:
            print(f'[{i+1}/{len(list_dir)}] Loading {file}...')
        try:
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            batch_size = len(data['trans'])
            body_model_params['batch_size'] = batch_size
            body_model = smplx.create(**body_model_params).to(device)
            smplx_params = smplx_breakdown(data, v_temp, device, canonicalize=True)

            # zero out global orientation and translation (not relevant for our experiment):
            global_orient = smplx_params['global_orient'] * 0.0
            trans = torch.from_numpy(data['trans']).float().to(device) * 0.0

            body_pose = smplx_params['body_pose']
            left_hand_pose = smplx_params['left_hand_pose']
            right_hand_pose = smplx_params['right_hand_pose']

            betas = None
            
            body_model_output = body_model(
                                betas=betas,
                                transl=trans,
                                global_orient=global_orient,
                                body_pose=body_pose,
                                left_hand_pose=left_hand_pose,
                                right_hand_pose=right_hand_pose)
            
            all_verts = body_model_output.vertices.detach().cpu()
            all_verts = all_verts[:, keep_mask]
            
            if middle:
                # keep only the middle third of the sequence to filter trivially redundant samples in MOYO captures
                all_verts = all_verts[len(all_verts)//3:2*len(all_verts)//3]
                body_pose = body_pose[len(body_pose)//3:2*len(body_pose)//3]
                left_hand_pose = left_hand_pose[len(left_hand_pose)//3:2*len(left_hand_pose)//3]
                right_hand_pose = right_hand_pose[len(right_hand_pose)//3:2*len(right_hand_pose)//3]

            loaded_verts += [all_verts[::sub_sample_rate].cpu()]
            body_poses += [body_pose[::sub_sample_rate].cpu()]
            left_hand_poses += [left_hand_pose[::sub_sample_rate].cpu()]
            right_hand_poses += [right_hand_pose[::sub_sample_rate].cpu()]

            nloaded += len(loaded_verts[-1])
        except Exception as e:
            print('Error processing', file, e, 'skipping...')
    
    loaded_verts = torch.cat(loaded_verts, dim=0)
    body_poses = torch.cat(body_poses, dim=0)
    left_hand_poses = torch.cat(left_hand_poses, dim=0)
    right_hand_poses = torch.cat(right_hand_poses, dim=0)
    return loaded_verts, f_temp_noeyes, body_poses, left_hand_poses, right_hand_poses

bake_dataset(train=True, num_samples=args.train_samples, num_dupes=args.num_dupes, body_shape_std=args.body_shape_std)
bake_dataset(train=False, num_samples=args.test_samples, num_dupes=args.num_dupes, body_shape_std=args.body_shape_std)