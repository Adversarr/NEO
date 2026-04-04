For camel
```
blender -b -P renders/render_mesh_once.py -- --input res_inf_zo.ply --output res_inf_zo.png --view_transform Filmic --denoise --world_strength=0.5 --cam_pos 0.6 0.3 1.2 --cam_target 0 0 0 --area_light_pos 2 1 8 --lights_target_center --resolution 1280 1280 --area_light_energy=300 --world_up 0 1 0.5
```