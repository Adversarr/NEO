```to render teaser figure
python renders/batch_render.py --root tmp_teaser/input_samples --feat 1 2 4 8 --k 2 8 16 20 --err_cmap Reds -- --denoise --world_strength=0.5 --radius=0.007 --ico_subdiv=3 --view_transform=Filmic --emission_strength 0.3 --cam_p 1 4 0.5 --auto_lights --lights_target_center --cam_target 0 0 0 --resolution 1000 1000 --fov 30 --point_light_energy=0 --samples=128 --area_light_pos 0 8 4 --area_light_energy=300 --obj_rot_deg 90 0 0
blender -b -P renders/render_once.py -- --input tmp/test_row/bunny_row.ply --output tmp_teaser/render_bunny_row_3.png --denoise --world_strength=0.5 --radius=0.007 --ico_subdiv=3 --view_transform=Filmic --emission_strength 0.3 --cam_pos 4 4 2 --auto_lights --lights_target_center --cam_target 2 2 0.3 --resolution 2000 4000 --fov 30 --point_light_energy=0 --samples=128 --area_light_pos 0 8 4 --area_light_energy=200 --obj_rot_deg 90 0 0 --samples=8 --orthographic --ortho_scale=6

 python renders/make_a_row.py --case /home/adversarr/Repo/g2pt/tmp_teaser/input_samples/Bunny --output tmp/test_row/bunny_row.ply --features 7 2 16 4 10 --direction 1 0 -1 --spacing 1.2 --colormap PuOr --no-symmetric
python renders/make_a_row.py --case /home/adversarr/Repo/g2pt/tmp_teaser/input_samples/Bunny --output tmp/test_row/bunny_row_eigs.ply --features pc:4 pc:8 pc:16 --direction 1 0 -1 --spacing 2.4 --colormap coolwarm

blender -b -P renders/render_once.py -- --input tmp/test_row/bunny_row_eigs.ply --output tmp_teaser/render_bunny_row_4.png --denoise --world_strength=0.5 --radius=0.007 --ico_subdiv=3 --view_transform=Filmic --emission_strength 0.3 --cam_pos 4 4 2 --auto_lights --lights_target_center --cam_target 2 2 0.3 --resolution 2000 4000 --fov 30 --point_light_energy=0 --samples=128 --area_light_pos 0 8 4 --area_light_energy=200 --obj_rot_deg 90 0 0 --samples=8 --orthographic --ortho_scale=6
```

```to sample a non-uniform grid
python scripts/sample_points.py --mesh_path ldata/featuring_models/igea-Rot180.obj --n_points 131072 --out tmp_teaser/input_samples/Igea-NonUniform/sample_points.npy --normalize --non_uniform --non_uniform_sigma 0.5 --seed=3407
```