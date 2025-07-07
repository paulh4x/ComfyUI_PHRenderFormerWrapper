import os
import sys
import torch
import numpy as np
import trimesh
from pathlib import Path
import json
import io
import h5py
from dacite import from_dict, Config
import tempfile
import shutil

import comfy.model_management as mm
import folder_paths
import imageio
script_directory = Path(__file__).parent
renderformer_repo_path = script_directory / "example_repos" / "renderformer"
scene_processor_path = renderformer_repo_path / "scene_processor"
# Add both the repo root and the scene_processor directory to the path
# This allows imports like `from renderformer...` and also allows the
# scripts inside scene_processor to import each other directly (e.g., `from scene_config...`)
if str(renderformer_repo_path) not in sys.path:
    sys.path.append(str(renderformer_repo_path))
if str(scene_processor_path) not in sys.path:
    sys.path.append(str(scene_processor_path))

try:
    from renderformer.pipelines.rendering_pipeline import RenderFormerRenderingPipeline
    from scene_config import SceneConfig
    from scene_mesh import generate_scene_mesh
    from to_h5 import save_to_h5 as original_save_to_h5
except ImportError as e:
    print(f"Could not import RenderFormer components. Error: {e}. Please ensure the repository exists in 'example_repos/renderformer' and you have installed the requirements.")


def look_at_to_c2w(camera_position, target_position, up_dir=np.array([0.0, 0.0, 1.0])):
    """
    Calculates the camera-to-world (c2w) matrix.
    Assumes a Z-up, right-handed coordinate system to match RenderFormer's examples.
    """
    camera_position = np.array(camera_position)
    target_position = np.array(target_position)
    
    z_axis = camera_position - target_position
    z_axis /= np.linalg.norm(z_axis)
    
    x_axis = np.cross(up_dir, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_position
    
    return c2w


class RenderFormerModelLoader:
    """
    Loads the RenderFormer model pipeline.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "microsoft/renderformer-v1.1-swin-large",
                    "tooltip": "The model ID from Hugging Face or a local path."
                }),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PHRenderFormer"

    def load_model(self, model_id, precision):
        device = mm.get_torch_device()
        
        torch_dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32
        }[precision]

        # from_pretrained handles model downloading and caching
        pipeline = RenderFormerRenderingPipeline.from_pretrained(model_id)
        pipeline.to(device)
        
        model = {
            "pipeline": pipeline,
            "torch_dtype": torch_dtype
        }
        
        return (model,)

class LoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        # Reverting input order for debugging purposes.
        return {
            "required": {
                "mesh": (folder_paths.get_filename_list("3d"), ),
            },
            "optional": {
                "material": ("RENDERFORMER_MATERIAL",),
                "mesh_path": ("STRING", ),
                
                # Transform properties
                "normalize_mesh": ("BOOLEAN", {"default": True}),
                "trans_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "rot_x": ("FLOAT", {"default": -90.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),

                # Material properties
                "diffuse_rgb": ("STRING", {"default": "204, 204, 204", "multiline": False, "tooltip": "Diffuse color (R, G, B) from 0-255"}),
                "specular_rgb": ("STRING", {"default": "25, 25, 25", "multiline": False, "tooltip": "Specular color (R, G, B) from 0-255"}),
                "roughness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emissive_rgb": ("STRING", {"default": "0, 0, 0", "multiline": False, "tooltip": "Emissive strength (R, G, B). Values can exceed 255."}),
            }
        }

    RETURN_TYPES = ("PH_MESH",)
    FUNCTION = "load_mesh"
    CATEGORY = "PHRenderFormer"

    def _parse_color_string(self, color_str, scale=1.0):
        if not isinstance(color_str, str):
            return None
        try:
            parts = [float(p.strip()) for p in color_str.split(',')]
            if len(parts) == 3:
                return [p / scale for p in parts]
        except (ValueError, AttributeError):
            pass
        return None

    def load_mesh(self, mesh, material=None, mesh_path=None,
                  normalize_mesh=True, trans_x=0.0, trans_y=0.0, trans_z=0.0,
                  rot_x=-90.0, rot_y=0.0, rot_z=0.0, scale=1.0,
                  diffuse_rgb="204, 204, 204",
                  specular_rgb="25, 25, 25", roughness=0.7, emissive_rgb="0, 0, 0"):
        
        final_material = None

        # If a material is explicitly connected, it overrides the internal settings.
        if material:
            final_material = material
        else:
            # Otherwise, build the material from the node's own inputs.
            diffuse_color = self._parse_color_string(diffuse_rgb, 255.0) or [0.8, 0.8, 0.8]
            specular_color = self._parse_color_string(specular_rgb, 255.0) or [0.1, 0.1, 0.1]
            emissive_color = self._parse_color_string(emissive_rgb, 1.0) or [0.0, 0.0, 0.0]

            final_material = {
                "diffuse": diffuse_color,
                "specular": specular_color,
                "roughness": roughness,
                "emissive": emissive_color
            }

        # Load mesh from path
        path_to_load = None
        if mesh_path and mesh_path.strip():
            path_to_load = folder_paths.get_annotated_filepath(mesh_path)
        else:
            path_to_load = folder_paths.get_full_path("3d", mesh)

        if not path_to_load or not os.path.exists(path_to_load):
            raise Exception(f"RenderFormer Mesh Loader: Mesh file not found.")

        loaded_mesh = trimesh.load(path_to_load, force='mesh')
        
        # Define transform
        transform = {
            "translation": [trans_x, trans_y, trans_z],
            "rotation": [rot_x, rot_y, rot_z],
            "scale": [scale, scale, scale],
            "normalize": normalize_mesh
        }
        
        # Package the final mesh, material, and transform
        ph_mesh = {
            "meshes": [loaded_mesh],
            "materials": [final_material],
            "transforms": [transform]
        }
        
        return (ph_mesh,)

class RemeshMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("PH_MESH",),
                "target_face_count": ("INT", {"default": 4096, "min": 100, "max": 100000, "step": 100}),
            }
        }

    RETURN_TYPES = ("PH_MESH",)
    FUNCTION = "remesh"
    CATEGORY = "PHRenderFormer"

    def remesh(self, mesh, target_face_count):
        import pymeshlab
        
        # We only remesh the first mesh in the list for now.
        # A more complex implementation could handle multiple meshes.
        if not mesh["meshes"]:
            return (mesh,)

        first_mesh = mesh["meshes"][0]
        
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=first_mesh.vertices, face_matrix=first_mesh.faces))
        
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(0.5),
            featuredeg=30,
            adaptive=False
        )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face_count,
            qualitythr=1.0
        )
        
        processed_mesh_lab = ms.current_mesh()
        vertices = processed_mesh_lab.vertex_matrix()
        faces = processed_mesh_lab.face_matrix()
        
        processed_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create a new PH_MESH object with the processed mesh and original material
        new_ph_mesh = {
            "meshes": [processed_trimesh] + mesh["meshes"][1:],
            "materials": mesh["materials"],
            "transforms": mesh["transforms"]
        }
        
        return (new_ph_mesh,)

class RandomizeColors:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("PH_MESH",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("PH_MESH",)
    FUNCTION = "randomize"
    CATEGORY = "PHRenderFormer"

    def randomize(self, mesh, seed):
        # We only randomize colors on the first mesh in the list.
        if not mesh["meshes"]:
            return (mesh,)
            
        mesh_copy = mesh["meshes"][0].copy()
        
        # Generate a deterministic gradient of colors for each face to prove functionality.
        # The seed will still slightly alter the gradient pattern.
        num_faces = len(mesh_copy.faces)
        np.random.seed(seed)
        
        # Create a base progression
        progression = np.linspace(0, 1, num_faces)
        
        # Create varied color channels based on the progression
        r = (np.sin(progression * 2 * np.pi + np.random.uniform(0, np.pi)) * 0.5 + 0.5) * 255
        g = (np.cos(progression * 2 * np.pi + np.random.uniform(0, np.pi)) * 0.5 + 0.5) * 255
        b = (progression * 255) % 255
        
        # Combine into an RGB array
        colors = np.vstack([r, g, b]).T
        
        # Apply the colors to the mesh faces
        mesh_copy.visual.face_colors = colors.astype(np.uint8)
        
        new_ph_mesh = {
            "meshes": [mesh_copy] + mesh["meshes"][1:],
            "materials": mesh["materials"],
            "transforms": mesh["transforms"]
        }
        
        return (new_ph_mesh,)

class RenderFormerCamera:
    """
    Defines the camera position, orientation, and field of view.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_pos_y": ("FLOAT", {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_pos_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_look_at_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_look_at_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_look_at_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "fov": ("FLOAT", {"default": 37.5, "min": 1.0, "max": 179.0, "step": 1.0, "tooltip": "Field of View in degrees"}),
            }
        }

    RETURN_TYPES = ("RENDERFORMER_CAMERA",)
    FUNCTION = "get_camera"
    CATEGORY = "PHRenderFormer"

    def get_camera(self, camera_pos_x, camera_pos_y, camera_pos_z,
                   camera_look_at_x, camera_look_at_y, camera_look_at_z, fov):
        camera_settings = {
            "position": [camera_pos_x, camera_pos_y, camera_pos_z],
            "look_at": [camera_look_at_x, camera_look_at_y, camera_look_at_z],
            "fov": fov
        }
        return (camera_settings,)

class RenderFormerLighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emissive_rgb": ("STRING", {"default": "255, 255, 255", "multiline": False, "tooltip": "Emissive color (R, G, B) from 0-255"}),
                "emissive_strength": ("FLOAT", {"default": 5000.0, "min": 0.0, "max": 100000.0, "step": 100.0}),
                "trans_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_z": ("FLOAT", {"default": 2.1, "min": -10.0, "max": 10.0, "step": 0.001}),
                "rot_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "scale": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("RENDERFORMER_LIGHTING",)
    FUNCTION = "get_lighting"
    CATEGORY = "PHRenderFormer"

    def _parse_color_string(self, color_str, scale=1.0):
        if not isinstance(color_str, str):
            return None
        try:
            parts = [float(p.strip()) for p in color_str.split(',')]
            if len(parts) == 3:
                return [p / scale for p in parts]
        except (ValueError, AttributeError):
            pass
        return None

    def get_lighting(self, emissive_rgb, emissive_strength, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, scale):
        transform = {
            "translation": [trans_x, trans_y, trans_z],
            "rotation": [rot_x, rot_y, rot_z],
            "scale": [scale, scale, scale],
            "normalize": False
        }
        
        # Parse color and apply strength
        base_color = self._parse_color_string(emissive_rgb, 255.0) or [1.0, 1.0, 1.0]
        final_emissive = [c * emissive_strength for c in base_color]

        material = {
            "diffuse": [1.0, 1.0, 1.0],
            "specular": [0.0, 0.0, 0.0],
            "roughness": 1.0,
            "emissive": final_emissive,
            "smooth_shading": False,
            "rand_tri_diffuse_seed": None
        }
        
        # Each lighting node now produces a list containing a single light definition
        light_definition = [{
            "transform": transform,
            "material": material
        }]
        
        return (light_definition,)

class RenderFormerLightingCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lighting_1": ("RENDERFORMER_LIGHTING",),
            },
            "optional": {
                "lighting_2": ("RENDERFORMER_LIGHTING",),
                "lighting_3": ("RENDERFORMER_LIGHTING",),
                "lighting_4": ("RENDERFORMER_LIGHTING",),
                "lighting_5": ("RENDERFORMER_LIGHTING",),
                "lighting_6": ("RENDERFORMER_LIGHTING",),
                "lighting_7": ("RENDERFORMER_LIGHTING",),
                "lighting_8": ("RENDERFORMER_LIGHTING",),
            }
        }

    RETURN_TYPES = ("RENDERFORMER_LIGHTING",)
    FUNCTION = "combine_lighting"
    CATEGORY = "PHRenderFormer"

    def combine_lighting(self, lighting_1, **kwargs):
        combined_lights = list(lighting_1)
        
        for key in sorted(kwargs.keys()):
            lighting_list = kwargs.get(key)
            if lighting_list and isinstance(lighting_list, list):
                combined_lights.extend(lighting_list)
        
        return (combined_lights,)

class RenderFormerSceneBuilder:
    """
    Builds a scene for RenderFormer from a mesh and camera/material properties.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("PH_MESH",),
                "camera": ("RENDERFORMER_CAMERA",),
                "lighting": ("RENDERFORMER_LIGHTING",),
            },
            "optional": {
                 "add_default_background": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RENDERFORMER_SCENE",)
    FUNCTION = "build_scene"
    CATEGORY = "PHRenderFormer"

    def build_scene(self, mesh, camera, lighting, add_default_background=False):
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            config_data = {
                "scene_name": "built_scene",
                "version": "1.0",
                "objects": {},
                "cameras": [
                    {
                        "position": camera["position"],
                        "look_at": camera["look_at"],
                        "up": [0.0, 0.0, 1.0],
                        "fov": camera["fov"]
                    }
                ]
            }

            # Process each mesh from the PH_MESH input
            for i, (mesh_obj, material, transform) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
                obj_key = f"main_object_{i}"
                temp_mesh_path = tmpdir_path / f"{obj_key}.obj"
                mesh_obj.export(temp_mesh_path)

                # Define a default material if none is provided
                if material is None:
                    material = {
                        "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1],
                        "roughness": 0.7, "emissive": [0.0, 0.0, 0.0]
                    }

                config_data["objects"][obj_key] = {
                    "mesh_path": temp_mesh_path.name,
                    "transform": transform,
                    "material": {
                        "diffuse": material["diffuse"],
                        "specular": material["specular"],
                        "roughness": material["roughness"],
                        "emissive": material["emissive"],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                }

            # Process each light from the RENDERFORMER_LIGHTING input list
            if lighting and isinstance(lighting, list):
                for i, light_def in enumerate(lighting):
                    obj_key = f"comfy_light_{i}"
                    config_data["objects"][obj_key] = {
                        "mesh_path": "templates/lighting/tri.obj",
                        "transform": light_def["transform"],
                        "material": light_def["material"]
                    }
            
            if add_default_background:
                background_files = ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]
                for i, bg_file in enumerate(background_files):
                    obj_key = f"comfy_default_background_{i}"
                    config_data["objects"][obj_key] = {
                        "mesh_path": f"templates/backgrounds/{bg_file}",
                        "transform": {
                            "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                            "scale": [0.5, 0.5, 0.5], "normalize": False
                        },
                        "material": {
                            "diffuse": [0.4, 0.4, 0.4], "specular": [0.0, 0.0, 0.0],
                            "random_diffuse_max": 0.4, "roughness": 0.99,
                            "emissive": [0.0, 0.0, 0.0], "smooth_shading": True,
                            "rand_tri_diffuse_seed": None
                        }
                    }

            # 4. Use the same processing pipeline as RenderFormerFromJSON
            # This is a simplified version of the logic in RenderFormerFromJSON
            scene_config_path = tmpdir_path / "scene.json"
            with open(scene_config_path, "w") as f:
                json.dump(config_data, f)

            # Copy template files if needed
            template_mesh_root = Path(__file__).parent / "example_repos" / "renderformer" / "examples"
            # Copy light mesh if any lights are present
            if lighting and isinstance(lighting, list) and len(lighting) > 0:
                light_mesh_path = template_mesh_root / "templates" / "lighting" / "tri.obj"
                dest_light_path = tmpdir_path / "templates" / "lighting"
                dest_light_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(light_mesh_path, dest_light_path / "tri.obj")
            
            if add_default_background:
                background_template_path = template_mesh_root / "templates" / "backgrounds"
                dest_background_path = tmpdir_path / "templates" / "backgrounds"
                dest_background_path.mkdir(parents=True, exist_ok=True)
                for bg_file in ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]:
                    shutil.copy(background_template_path / bg_file, dest_background_path / bg_file)


            # --- Start of conversion logic ---
            scene_config_dir = tmpdir_path
            original_export = trimesh.exchange.export.export_mesh
            original_load = trimesh.load
            virtual_filesystem = {}

            def patched_export(mesh, file_obj, file_type, **kwargs):
                with io.BytesIO() as buffer:
                    actual_file_type = Path(file_obj).suffix[1:]
                    mesh.export(buffer, file_type=actual_file_type, **kwargs)
                    buffer.seek(0)
                    virtual_filesystem[str(file_obj)] = buffer.read()

            def patched_load(file_obj, **kwargs):
                if str(file_obj) in virtual_filesystem:
                    with io.BytesIO(virtual_filesystem[str(file_obj)]) as buffer:
                        return original_load(buffer, file_type=Path(file_obj).suffix[1:], **kwargs)
                return original_load(file_obj, **kwargs)

            trimesh.exchange.export.export_mesh = patched_export
            trimesh.load = patched_load

            try:
                scene_config = from_dict(data_class=SceneConfig, data=config_data, config=Config(check_types=True, strict=True))

                with io.BytesIO() as h5_buffer:
                    temp_mesh_dir_name = "temp_mesh_dir"
                    
                    generate_scene_mesh(scene_config, f"{temp_mesh_dir_name}/scene.obj", str(scene_config_dir))

                    def patched_save_to_h5(scene_config, mesh_path, output_h5_buffer):
                        all_triangles, all_vn, all_texture = [], [], []
                        size = 32
                        mask = np.zeros((size, size), dtype=bool)
                        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                        mask[x + y <= size] = 1
                        
                        split_mesh_path_prefix = os.path.dirname(mesh_path) + '/split'
                        for obj_key, obj_config in scene_config.objects.items():
                            mesh = trimesh.load(f'{split_mesh_path_prefix}/{obj_key}.obj', process=False, force='mesh')
                            triangles = mesh.triangles
                            vn = mesh.vertex_normals[mesh.faces]
                            material_config = obj_config.material
                            
                            diffuse = mesh.visual.face_colors[..., :3] / 255.

                            specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
                            roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
                            normal = np.array([0.5, 0.5, 1.0])[None].repeat(triangles.shape[0], axis=0)
                            irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
                            texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
                            texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
                            texture[:, :, ~mask] = 0.0
                            all_triangles.append(triangles)
                            all_vn.append(vn)
                            all_texture.append(texture)
                        
                        all_triangles = np.concatenate(all_triangles, axis=0)
                        all_vn = np.concatenate(all_vn, axis=0)
                        all_texture = np.concatenate(all_texture, axis=0)
                        
                        all_c2w, all_fov = [], []
                        for camera_config in scene_config.cameras:
                            c2w = look_at_to_c2w(camera_config.position, camera_config.look_at)
                            all_c2w.append(c2w)
                            all_fov.append(camera_config.fov)
                        
                        all_c2w = np.stack(all_c2w)
                        all_fov = np.array(all_fov)

                        with h5py.File(output_h5_buffer, "w") as f:
                            f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip", compression_opts=9)
                            f.create_dataset("c2w", data=all_c2w.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("fov", data=all_fov.astype(np.float32), compression="gzip", compression_opts=9)

                    patched_save_to_h5(scene_config, f"{temp_mesh_dir_name}/scene.obj", h5_buffer)

                    h5_buffer.seek(0)
                    with h5py.File(h5_buffer, 'r') as f:
                        c2w_data = f['c2w'][:]
                        fov_data = f['fov'][:]
                        scene = {
                            'triangles': torch.from_numpy(f['triangles'][:]).unsqueeze(0),
                            'texture': torch.from_numpy(f['texture'][:]).unsqueeze(0),
                            'mask': torch.ones(f['triangles'][:].shape[0], dtype=torch.bool).unsqueeze(0),
                            'vn': torch.from_numpy(f['vn'][:]).unsqueeze(0),
                            'c2w': torch.from_numpy(c2w_data).unsqueeze(0),
                            'fov': torch.from_numpy(fov_data).unsqueeze(0).unsqueeze(-1),
                        }
            finally:
                trimesh.load = original_load
                trimesh.exchange.export.export_mesh = original_export
            
            return (scene,)

class RenderFormerMeshCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_1": ("PH_MESH",),
            },
            "optional": {
                "mesh_2": ("PH_MESH",),
                "mesh_3": ("PH_MESH",),
                "mesh_4": ("PH_MESH",),
                "mesh_5": ("PH_MESH",),
                "mesh_6": ("PH_MESH",),
                "mesh_7": ("PH_MESH",),
                "mesh_8": ("PH_MESH",),
            }
        }

    RETURN_TYPES = ("PH_MESH",)
    FUNCTION = "combine_meshes"
    CATEGORY = "PHRenderFormer"

    def combine_meshes(self, mesh_1, **kwargs):
        all_meshes = []
        all_materials = []
        all_transforms = []

        # Add the first mesh and material
        all_meshes.extend(mesh_1["meshes"])
        all_materials.extend(mesh_1["materials"])
        all_transforms.extend(mesh_1["transforms"])
        
        # Add subsequent meshes and materials from optional inputs
        for key in sorted(kwargs.keys()):
            ph_mesh = kwargs.get(key)
            if ph_mesh and isinstance(ph_mesh, dict) and "meshes" in ph_mesh and "materials" in ph_mesh and "transforms" in ph_mesh:
                all_meshes.extend(ph_mesh["meshes"])
                all_materials.extend(ph_mesh["materials"])
                all_transforms.extend(ph_mesh["transforms"])
        
        combined_ph_mesh = {"meshes": all_meshes, "materials": all_materials, "transforms": all_transforms}

        return (combined_ph_mesh,)

class RenderFormerGenerator:
    """
    Renders an image using the RenderFormer pipeline.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scene": ("RENDERFORMER_SCENE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tone_mapper": (["none", "agx", "filmic", "pbr_neutral"], {"default": "agx"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "PHRenderFormer"

    def generate(self, model, scene, resolution, tone_mapper):
        device = mm.get_torch_device()

        pipeline = model["pipeline"]
        torch_dtype = model["torch_dtype"]

        # Create a shallow copy of the scene to avoid modifying the original cache
        scene_on_device = scene.copy()
        
        # Move all scene tensors to the correct device
        for key, tensor in scene_on_device.items():
            scene_on_device[key] = tensor.to(device)

        rendered_imgs = pipeline(
            triangles=scene_on_device['triangles'],
            texture=scene_on_device['texture'],
            mask=scene_on_device['mask'],
            vn=scene_on_device['vn'],
            c2w=scene_on_device['c2w'],
            fov=scene_on_device['fov'],
            resolution=resolution,
            torch_dtype=torch_dtype,
        )

        # Process the output image
        # Manually apply log decoding to expand the value range
        rendered_imgs = torch.pow(10., rendered_imgs) - 1.
        
        hdr_img_tensor = rendered_imgs[0, 0].cpu()
        
        hdr_img = hdr_img_tensor.numpy().astype(np.float32)

        if tone_mapper != 'none':
            from simple_ocio import ToneMapper
            if tone_mapper == 'pbr_neutral':
                tone_mapper = 'Khronos PBR Neutral'
            tm = ToneMapper(tone_mapper)
            ldr_img = tm.hdr_to_ldr(hdr_img)
        else:
            ldr_img = np.clip(hdr_img, 0, 1)
        
        # Convert to torch tensor for ComfyUI
        ldr_img_tensor = torch.from_numpy(ldr_img).unsqueeze(0)

        return (ldr_img_tensor,)

class LoadRenderFormerExampleScene:
    """
    Loads an example scene from the RenderFormer examples directory.
    This node programmatically executes the equivalent of 'convert_scene.py'
    to generate the scene data in memory, replicating the original workflow.
    """
    
    EXAMPLE_DIR = Path(__file__).parent / "example_repos" / "renderformer" / "examples"
    EXAMPLE_FILES = [f.name for f in EXAMPLE_DIR.glob("*.json")]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_file": (cls.EXAMPLE_FILES, )
            }
        }

    RETURN_TYPES = ("RENDERFORMER_SCENE",)
    FUNCTION = "load_example_scene"
    CATEGORY = "PHRenderFormer/Test"

    def load_example_scene(self, scene_file):
        # Define paths
        base_path = Path(__file__).parent / "example_repos" / "renderformer"
        scene_config_path = base_path / "examples" / scene_file
        scene_config_dir = scene_config_path.parent

        # 1. Load and parse the scene configuration JSON
        with open(scene_config_path, 'r') as f:
            config_data = json.load(f)
        scene_config = from_dict(data_class=SceneConfig, data=config_data, config=Config(check_types=True, strict=True))

        # 2. Create an in-memory HDF5 file to store the processed scene
        with io.BytesIO() as h5_buffer:
            temp_mesh_dir = "temp_mesh_dir"
            virtual_filesystem = {}

            # --- Patching mechanism for trimesh ---
            original_export = trimesh.exchange.export.export_mesh
            original_load = trimesh.load

            def patched_export(mesh, file_obj, file_type, **kwargs):
                with io.BytesIO() as buffer:
                    # Use the file extension from the provided file_obj string
                    actual_file_type = file_obj.split('.')[-1]
                    mesh.export(buffer, file_type=actual_file_type, **kwargs)
                    buffer.seek(0)
                    virtual_filesystem[file_obj] = buffer.read()

            def patched_load(file_obj, **kwargs):
                if file_obj in virtual_filesystem:
                    with io.BytesIO(virtual_filesystem[file_obj]) as buffer:
                        # The file type is obj for our intermediate files
                        return original_load(buffer, file_type='obj', **kwargs)
                # If not in our virtual system, load from disk as usual
                return original_load(file_obj, **kwargs)

            try:
                # 3. Generate scene mesh, exporting intermediate files to memory
                trimesh.exchange.export.export_mesh = patched_export
                generate_scene_mesh(scene_config, f"{temp_mesh_dir}/scene.obj", str(scene_config_dir))

                # 4. Save to HDF5, loading intermediate files from memory
                trimesh.load = patched_load
                
                # The original function expects a file path for the h5, but we have a buffer.
                # We'll create a small, patched version of the save function that can handle a buffer.
                def patched_save_to_h5(scene_config, mesh_path, output_h5_buffer):
                    # This is the core logic from the original to_h5.py, but adapted for our in-memory objects
                    all_triangles = []
                    all_vn = []
                    all_texture = []
                    size = 32
                    mask = np.zeros((size, size), dtype=bool)
                    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                    mask[x + y <= size] = 1
                    
                    split_mesh_path_prefix = os.path.dirname(mesh_path) + '/split'
                    for obj_key, obj_config in scene_config.objects.items():
                        # Use the patched trimesh.load to get the mesh from our virtual filesystem
                        mesh = trimesh.load(f'{split_mesh_path_prefix}/{obj_key}.obj', process=False, force='mesh')
                        triangles = mesh.triangles
                        vn = mesh.vertex_normals[mesh.faces]
                        material_config = obj_config.material
                        diffuse = mesh.visual.face_colors[..., :3] / 255.
                        specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
                        roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
                        normal = np.array([0.5, 0.5, 1.0])[None].repeat(triangles.shape[0], axis=0)
                        irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
                        texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
                        texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
                        texture[:, :, ~mask] = 0.0
                        all_triangles.append(triangles)
                        all_vn.append(vn)
                        all_texture.append(texture)
                    
                    all_triangles = np.concatenate(all_triangles, axis=0)
                    all_vn = np.concatenate(all_vn, axis=0)
                    all_texture = np.concatenate(all_texture, axis=0)
                    
                    all_c2w = []
                    all_fov = []
                    for camera_config in scene_config.cameras:
                        c2w = look_at_to_c2w(camera_config.position, camera_config.look_at)
                        all_c2w.append(c2w)
                        all_fov.append(camera_config.fov)
                    
                    all_c2w = np.stack(all_c2w)
                    all_fov = np.array(all_fov)

                    # This is the key change: write directly to the buffer
                    with h5py.File(output_h5_buffer, "w") as f:
                        f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip", compression_opts=9)
                        f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip", compression_opts=9)
                        f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip", compression_opts=9)
                        f.create_dataset("c2w", data=all_c2w.astype(np.float32), compression="gzip", compression_opts=9)
                        f.create_dataset("fov", data=all_fov.astype(np.float32), compression="gzip", compression_opts=9)

                patched_save_to_h5(scene_config, f"{temp_mesh_dir}/scene.obj", h5_buffer)

            finally:
                # Ensure original functions are always restored
                trimesh.exchange.export.export_mesh = original_export
                trimesh.load = original_load

            # 5. Read the data back from the HDF5 buffer and create the scene dictionary
            h5_buffer.seek(0)
            with h5py.File(h5_buffer, 'r') as f:
                # Data from H5 is shaped for multiple views (V, ...), add a batch dim (B, V, ...)
                # B=1 (batch size), V=1 (views for cbox)
                c2w_data = f['c2w'][:] # Shape: (V, 4, 4) -> (1, 4, 4)
                fov_data = f['fov'][:]   # Shape: (V,) -> (1,)

                scene = {
                    'triangles': torch.from_numpy(f['triangles'][:]).unsqueeze(0),
                    'texture': torch.from_numpy(f['texture'][:]).unsqueeze(0),
                    'mask': torch.ones(f['triangles'][:].shape[0], dtype=torch.bool).unsqueeze(0),
                    'vn': torch.from_numpy(f['vn'][:]).unsqueeze(0),
                    'c2w': torch.from_numpy(c2w_data).unsqueeze(0), # (1, 4, 4) -> (1, 1, 4, 4)
                    'fov': torch.from_numpy(fov_data).unsqueeze(0).unsqueeze(-1), # (1,) -> (1, 1) -> (1, 1, 1)
                }
        
        return (scene,)


INIT_TEMPLATE_JSON = """{
    "scene_name": "initial template",
    "version": "1.0",
    "objects": {
        "background_0": {
            "mesh_path": "templates/backgrounds/plane.obj",
            "transform": {
                "translation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "rotation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "scale": [
                    0.5,
                    0.5,
                    0.5
                ],
                "normalize": false
            },
            "material": {
                "diffuse": [
                    0.4,
                    0.4,
                    0.4
                ],
                "specular": [
                    0.0,
                    0.0,
                    0.0
                ],
                "random_diffuse_max": 0.4,
                "roughness": 0.99,
                "emissive": [
                    0.0,
                    0.0,
                    0.0
                ],
                "smooth_shading": true,
                "rand_tri_diffuse_seed": null
            }
        },
        "background_1": {
            "mesh_path": "templates/backgrounds/wall0.obj",
            "transform": {
                "translation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "rotation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "scale": [
                    0.5,
                    0.5,
                    0.5
                ],
                "normalize": false
            },
            "material": {
                "diffuse": [
                    0.4,
                    0.4,
                    0.4
                ],
                "specular": [
                    0.0,
                    0.0,
                    0.0
                ],
                "random_diffuse_max": 0.4,
                "roughness": 0.99,
                "emissive": [
                    0.0,
                    0.0,
                    0.0
                ],
                "smooth_shading": true,
                "rand_tri_diffuse_seed": null
            }
        },
        "background_2": {
            "mesh_path": "templates/backgrounds/wall1.obj",
            "transform": {
                "translation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "rotation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "scale": [
                    0.5,
                    0.5,
                    0.5
                ],
                "normalize": false
            },
            "material": {
                "diffuse": [
                    0.4,
                    0.4,
                    0.4
                ],
                "specular": [
                    0.0,
                    0.0,
                    0.0
                ],
                "random_diffuse_max": 0.5,
                "roughness": 0.99,
                "emissive": [
                    0.0,
                    0.0,
                    0.0
                ],
                "smooth_shading": true,
                "rand_tri_diffuse_seed": null
            }
        },
        "background_3": {
            "mesh_path": "templates/backgrounds/wall2.obj",
            "transform": {
                "translation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "rotation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "scale": [
                    0.5,
                    0.5,
                    0.5
                ],
                "normalize": false
            },
            "material": {
                "diffuse": [
                    0.4,
                    0.4,
                    0.4
                ],
                "specular": [
                    0.0,
                    0.0,
                    0.0
                ],
                "random_diffuse_max": 0.5,
                "roughness": 0.99,
                "emissive": [
                    0.0,
                    0.0,
                    0.0
                ],
                "smooth_shading": false,
                "rand_tri_diffuse_seed": null
            }
        },
        "light_0": {
            "mesh_path": "templates/lighting/tri.obj",
            "transform": {
                "translation": [
                    0.0,
                    0.0,
                    2.1
                ],
                "rotation": [
                    0.0,
                    0.0,
                    0.0
                ],
                "scale": [
                    2.5,
                    2.5,
                    2.5
                ],
                "normalize": false
            },
            "material": {
                "diffuse": [
                    1.0,
                    1.0,
                    1.0
                ],
                "specular": [
                    0.0,
                    0.0,
                    0.0
                ],
                "random_diffuse_max": 0.0,
                "roughness": 1.0,
                "emissive": [
                    5000.0,
                    5000.0,
                    5000.0
                ],
                "smooth_shading": false,
                "rand_tri_diffuse_seed": null
            }
        }
    },
    "cameras": [
        {
            "position": [
                0.0,
                -2.0,
                0.0
            ],
            "look_at": [
                0.0,
                0.0,
                0.0
            ],
            "up": [
                0.0,
                0.0,
                1.0
            ],
            "fov": 37.5
        }
    ]
}"""

class RenderFormerFromJSON:
    """
    Loads a RenderFormer scene from a user-provided JSON definition.
    This node allows for creating custom scenes by specifying the JSON
    and optionally adding an external .obj file.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Create a dropdown for .obj files in the 'input/3d' directory
        input_3d_dir = os.path.join(folder_paths.get_input_directory(), "3d")
        if not os.path.exists(input_3d_dir):
            os.makedirs(input_3d_dir)
        obj_files = [f for f in os.listdir(input_3d_dir) if f.lower().endswith(".obj")]
        
        return {
            "required": {
                "scene_json": ("STRING", {"multiline": True, "default": INIT_TEMPLATE_JSON}),
                "add_default_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "additional_mesh_file": (["None"] + sorted(obj_files),),
                "additional_mesh_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Absolute path to an .obj file"}),
                "mesh_root_dir": ("STRING", {"default": "", "tooltip": "Optional root directory for meshes in the JSON"}),
            }
        }

    RETURN_TYPES = ("RENDERFORMER_SCENE",)
    FUNCTION = "load_from_json"
    CATEGORY = "PHRenderFormer"

    def load_from_json(self, scene_json, add_default_background, mesh_root_dir=None, additional_mesh_file=None, additional_mesh_path=None):
        config_data = json.loads(scene_json)

        # Add the user-provided .obj file to the scene configuration
        obj_path_to_load = None
        if additional_mesh_path and isinstance(additional_mesh_path, str) and additional_mesh_path.strip().lower().endswith('.obj'):
            obj_path_to_load = folder_paths.get_annotated_filepath(additional_mesh_path)
        elif additional_mesh_file and isinstance(additional_mesh_file, str) and additional_mesh_file != "None":
            obj_path_to_load = os.path.join(folder_paths.get_input_directory(), "3d", additional_mesh_file)

        if obj_path_to_load:
            if "objects" not in config_data:
                config_data["objects"] = {}
            
            obj_key = f"comfy_added_obj_{Path(obj_path_to_load).stem}"
            config_data["objects"][obj_key] = {
                "mesh_path": obj_path_to_load,  # Use the absolute path
                "transform": {
                    "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0], "normalize": True
                },
                "material": {
                    "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1],
                    "roughness": 0.7, "emissive": [0.0, 0.0, 0.0],
                    "smooth_shading": True, "rand_tri_diffuse_seed": None
                }
            }

        # Add default background geometry if the toggle is enabled
        if add_default_background:
            if "objects" not in config_data:
                config_data["objects"] = {}
            
            background_files = ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]
            for i, bg_file in enumerate(background_files):
                obj_key = f"comfy_default_background_{i}"
                config_data["objects"][obj_key] = {
                    "mesh_path": f"templates/backgrounds/{bg_file}",
                    "transform": {
                        "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.5, 0.5, 0.5], "normalize": False
                    },
                    "material": {
                        "diffuse": [0.4, 0.4, 0.4], "specular": [0.0, 0.0, 0.0],
                        "random_diffuse_max": 0.4, "roughness": 0.99,
                        "emissive": [0.0, 0.0, 0.0], "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            scene_config_path = tmpdir_path / "scene.json"

            with open(scene_config_path, "w") as f:
                json.dump(config_data, f)

            user_mesh_root = Path(mesh_root_dir) if mesh_root_dir and os.path.isdir(mesh_root_dir) else None
            template_mesh_root = Path(__file__).parent / "example_repos" / "renderformer" / "examples"

            if "objects" in config_data:
                for obj_key, obj_data in config_data["objects"].items():
                    if "mesh_path" not in obj_data:
                        continue
                    
                    mesh_path_str = obj_data["mesh_path"]
                    mesh_path = Path(mesh_path_str)

                    if mesh_path.is_absolute():
                        if not mesh_path.exists():
                            raise FileNotFoundError(f"Absolute mesh file path not found: '{mesh_path_str}'")
                        
                        dest_path = tmpdir_path / mesh_path.name
                        shutil.copy(mesh_path, dest_path)
                        config_data["objects"][obj_key]["mesh_path"] = mesh_path.name
                    else:
                        source_path = None
                        if user_mesh_root:
                            candidate_path = user_mesh_root / mesh_path
                            if candidate_path.exists():
                                source_path = candidate_path
                        
                        if not source_path:
                            candidate_path = template_mesh_root / mesh_path
                            if candidate_path.exists():
                                source_path = candidate_path
                        
                        if not source_path:
                            raise FileNotFoundError(f"Relative mesh file not found: '{mesh_path_str}'. Searched in user root ('{user_mesh_root}') and templates ('{template_mesh_root}').")

                        dest_path = tmpdir_path / mesh_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(source_path, dest_path)
            
            # Re-write the potentially modified config to the temp file
            with open(scene_config_path, "w") as f:
                json.dump(config_data, f)

            scene_config_dir = tmpdir_path
            # --- Start of conversion logic (remains the same) ---
            original_export = trimesh.exchange.export.export_mesh
            original_load = trimesh.load
            virtual_filesystem = {}

            def patched_export(mesh, file_obj, file_type, **kwargs):
                with io.BytesIO() as buffer:
                    actual_file_type = Path(file_obj).suffix[1:]
                    mesh.export(buffer, file_type=actual_file_type, **kwargs)
                    buffer.seek(0)
                    virtual_filesystem[str(file_obj)] = buffer.read()

            def patched_load(file_obj, **kwargs):
                if str(file_obj) in virtual_filesystem:
                    with io.BytesIO(virtual_filesystem[str(file_obj)]) as buffer:
                        return original_load(buffer, file_type=Path(file_obj).suffix[1:], **kwargs)
                return original_load(file_obj, **kwargs)

            trimesh.exchange.export.export_mesh = patched_export
            trimesh.load = patched_load

            try:
                with open(scene_config_path, 'r') as f:
                    scene_config_json = json.load(f)
                scene_config = from_dict(data_class=SceneConfig, data=scene_config_json, config=Config(check_types=True, strict=True))

                with io.BytesIO() as h5_buffer:
                    temp_mesh_dir = "temp_mesh_dir"
                    
                    generate_scene_mesh(scene_config, f"{temp_mesh_dir}/scene.obj", str(scene_config_dir))

                    def patched_save_to_h5(scene_config, mesh_path, output_h5_buffer):
                        all_triangles, all_vn, all_texture = [], [], []
                        size = 32
                        mask = np.zeros((size, size), dtype=bool)
                        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                        mask[x + y <= size] = 1
                        
                        split_mesh_path_prefix = os.path.dirname(mesh_path) + '/split'
                        for obj_key, obj_config in scene_config.objects.items():
                            mesh = trimesh.load(f'{split_mesh_path_prefix}/{obj_key}.obj', process=False, force='mesh')
                            triangles = mesh.triangles
                            vn = mesh.vertex_normals[mesh.faces]
                            material_config = obj_config.material
                            diffuse = mesh.visual.face_colors[..., :3] / 255.
                            specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
                            roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
                            normal = np.array([0.5, 0.5, 1.0])[None].repeat(triangles.shape[0], axis=0)
                            irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
                            texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
                            texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
                            texture[:, :, ~mask] = 0.0
                            all_triangles.append(triangles)
                            all_vn.append(vn)
                            all_texture.append(texture)
                        
                        all_triangles = np.concatenate(all_triangles, axis=0)
                        all_vn = np.concatenate(all_vn, axis=0)
                        all_texture = np.concatenate(all_texture, axis=0)
                        
                        all_c2w, all_fov = [], []
                        for camera_config in scene_config.cameras:
                            c2w = look_at_to_c2w(camera_config.position, camera_config.look_at)
                            all_c2w.append(c2w)
                            all_fov.append(camera_config.fov)
                        
                        all_c2w = np.stack(all_c2w)
                        all_fov = np.array(all_fov)

                        with h5py.File(output_h5_buffer, "w") as f:
                            f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip", compression_opts=9)
                            f.create_dataset("c2w", data=all_c2w.astype(np.float32), compression="gzip", compression_opts=9)
                            f.create_dataset("fov", data=all_fov.astype(np.float32), compression="gzip", compression_opts=9)

                    patched_save_to_h5(scene_config, f"{temp_mesh_dir}/scene.obj", h5_buffer)

                    h5_buffer.seek(0)
                    with h5py.File(h5_buffer, 'r') as f:
                        c2w_data = f['c2w'][:]
                        fov_data = f['fov'][:]
                        scene = {
                            'triangles': torch.from_numpy(f['triangles'][:]).unsqueeze(0),
                            'texture': torch.from_numpy(f['texture'][:]).unsqueeze(0),
                            'mask': torch.ones(f['triangles'][:].shape[0], dtype=torch.bool).unsqueeze(0),
                            'vn': torch.from_numpy(f['vn'][:]).unsqueeze(0),
                            'c2w': torch.from_numpy(c2w_data).unsqueeze(0),
                            'fov': torch.from_numpy(fov_data).unsqueeze(0).unsqueeze(-1),
                        }
            finally:
                trimesh.load = original_load
                trimesh.exchange.export.export_mesh = original_export
            # --- End of conversion logic ---

            return (scene,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RenderFormerModelLoader": RenderFormerModelLoader,
    "RenderFormerCamera": RenderFormerCamera,
    "RenderFormerLighting": RenderFormerLighting,
    "RenderFormerSceneBuilder": RenderFormerSceneBuilder,
    "RenderFormerGenerator": RenderFormerGenerator,
    "LoadMesh": LoadMesh,
    "RemeshMesh": RemeshMesh,
    "RandomizeColors": RandomizeColors,
    "LoadRenderFormerExampleScene": LoadRenderFormerExampleScene,
    "RenderFormerFromJSON": RenderFormerFromJSON,
    "RenderFormerMeshCombine": RenderFormerMeshCombine,
    "RenderFormerLightingCombine": RenderFormerLightingCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenderFormerModelLoader": "PHRenderFormer Model Loader",
    "RenderFormerCamera": "PHRenderFormer Camera",
    "RenderFormerLighting": "PHRenderFormer Lighting",
    "RenderFormerSceneBuilder": "PHRenderFormer Scene Builder",
    "RenderFormerGenerator": "PHRenderFormer Sampler",
    "LoadMesh": "PHRenderFormer Mesh Loader",
    "RemeshMesh": "PHRenderFormer Remesh",
    "RandomizeColors": "PHRenderFormer Random Colors",
    "LoadRenderFormerExampleScene": "PHRenderFormer Example Scene",
    "RenderFormerFromJSON": "PHRenderFormer From JSON",
    "RenderFormerMeshCombine": "PHRenderFormer Mesh Combine",
    "RenderFormerLightingCombine": "PHRenderFormer Lighting Combine",
}