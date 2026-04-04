import torch
import numpy as np

def farthest_point_subset(all_shapes, max_samples=None, threshold=0.1):
    """
    Performs a farthest-point-like sampling of the meshes in all_shapes until
    the maximum distance among the remaining candidates falls below `threshold`.
    all_shapes: torch.Tensor of shape (N, V, 3)
    threshold: float distance cutoff for termination
    Returns a 1D torch.Tensor of selected (keep) indices.
    """
    N, V, d = all_shapes.shape
    all_shapes_flat = all_shapes.view(N, -1).cuda()
    
    keep_idx = [0]
    
    # Compute the Euclidean distances from all meshes to the initial mesh.
    dists = torch.norm(all_shapes_flat - all_shapes_flat[0], dim=1) # shape (N,)
    dists[0] = 0  # Already selected mesh
    
    # Iteratively add the mesh which is farthest from the current set.
    while True:
        if max_samples is not None and len(keep_idx) >= max_samples:
            break

        candidate_val, candidate_idx = torch.max(dists, dim=0)
        if candidate_val < threshold:
            break
        
        keep_idx.append(candidate_idx.item())
        candidate = all_shapes_flat[candidate_idx].unsqueeze(0)  # Shape (1, V*d)
        new_dists = torch.norm(all_shapes_flat - candidate, dim=1)
        dists = torch.minimum(dists, new_dists)
    
    return torch.tensor(keep_idx)


def smplx_breakdown(bdata, verts_template, device, canonicalize=True):
    num_frames = len(bdata['trans'])

    bdata['poses'] = bdata['fullpose']
    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][:, 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][:, 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][:, 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][:, 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][:, 120:]).float().to(device)

    if canonicalize:
        global_orient = global_orient * 0.0
    
    if isinstance(verts_template, np.ndarray):
        verts_template = torch.from_numpy(verts_template)
    verts_template = verts_template.to(device)

    body_params = {'global_orient': global_orient, 'body_pose': body_pose,
                   'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                   'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   'v_template': verts_template, }
    return body_params


def remove_parts_from_mesh(vertices: torch.Tensor,
                             faces: torch.Tensor,
                             seg_dict: dict):
    """
    Removes vertices (and any face that uses them) corresponding to the parts
    given in seg_dict. This functions specifically removes the elements
    corresponding to "leftEye", "rightEye", and "eyeballs" in SMPLX.
    
    Args:
        vertices: (V, 3) vertex coordinates.
        faces: (F, 3) triangle indices into vertices.
        seg_dict: Dictionary with keys "leftEye", "rightEye", "eyeballs" each
                  mapping to a list of vertex indices to remove.

    Returns:
        new_vertices: (V_new, 3) with removed vertices omitted.
        new_faces: (F_new, 3) with face indices updated and
                   faces referencing any removed vertex dropped.
    """
    remove_set = set(seg_dict.get("leftEye", []))
    remove_set.update(seg_dict.get("rightEye", []))
    remove_set.update(seg_dict.get("eyeballs", []))
    
    if len(remove_set) == 0:
        return vertices, faces

    V = vertices.shape[0]

    # Create a boolean mask for vertices to keep (True == keep)
    keep_mask = torch.ones(V, dtype=torch.bool, device=vertices.device)
    remove_indices = torch.tensor(sorted(list(remove_set)), dtype=torch.long, device=vertices.device)
    keep_mask[remove_indices] = False

    new_vertices = vertices[keep_mask]
    
    # mapping from old vertex indices to new indices:
    # for kept vertices, new_index = rank in keep_mask; for removed, set to -1.
    new_index = -torch.ones(V, dtype=torch.long, device=vertices.device)
    new_index[keep_mask] = torch.arange(keep_mask.sum(), device=vertices.device)

    new_faces = new_index[faces]  # shape (F, 3)
    
    valid_face_mask = (new_faces != -1).all(dim=1)
    new_faces = new_faces[valid_face_mask]
    return new_vertices, new_faces, keep_mask