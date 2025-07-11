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
import comfy.utils
script_directory = Path(__file__).parent
renderformer_repo_path = script_directory / "renderformer"
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
    print(f"Could not import RenderFormer components. Error: {e}. Please ensure the repository exists in 'renderformer' and you have installed the requirements.")


def look_at_to_c2w(camera_position, target_position, up_dir=np.array([0.0, 0.0, 1.0])):
    """
    Calculates the camera-to-world (c2w) matrix.
    Assumes a Z-up, right-handed coordinate system to match RenderFormer's examples.
    """
    camera_position = np.array(camera_position, dtype=np.float32)
    target_position = np.array(target_position, dtype=np.float32)
    up_dir = np.array(up_dir, dtype=np.float32)
    
    z_axis = camera_position - target_position
    z_axis /= np.linalg.norm(z_axis)
    
    x_axis = np.cross(up_dir, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4, dtype=np.float32)
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

class RenderFormerLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        # Reverting input order for debugging purposes.
        return {
            "required": {
                "mesh": (folder_paths.get_filename_list("3d"), ),
            },
            "optional": {
                "material": ("MATERIAL",),
                "mesh_path": ("STRING", ),
                
                # Transform properties
                "normalize_mesh": ("BOOLEAN", {"default": True}),
                "trans_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "rot_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),

                # Material properties
                "diffuse_rgb": ("STRING", {"default": "204, 204, 204", "multiline": False, "tooltip": "Diffuse color (R, G, B) from 0-255"}),
                "specular_rgb": ("STRING", {"default": "25, 25, 25", "multiline": False, "tooltip": "Specular color (R, G, B) from 0-255"}),
                "roughness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH", "MATERIAL",)
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
                  rot_x=0.0, rot_y=0.0, rot_z=0.0, scale=1.0,
                  diffuse_rgb="204, 204, 204",
                  specular_rgb="25, 25, 25", roughness=0.7):
        
        # Default transformations for specific background files from cbox-bunny.json
        background_files = ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]
        if mesh in background_files:
            normalize_mesh = False
            trans_x, trans_y, trans_z = 0.0, 0.0, 0.0
            rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
            scale = 0.5
        
        
        final_material = None

        # If a material is explicitly connected, it overrides the internal settings.
        if material:
            final_material = material
        else:
            # Otherwise, build the material from the node's own inputs.
            diffuse_color = self._parse_color_string(diffuse_rgb, 255.0) or [0.8, 0.8, 0.8]
            specular_color = self._parse_color_string(specular_rgb, 255.0) or [0.1, 0.1, 0.1]

            final_material = {
                "diffuse": diffuse_color,
                "specular": specular_color,
                "roughness": roughness,
                "emissive": [0.0, 0.0, 0.0]
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
        
        return (ph_mesh, final_material)

class RenderFormerRemeshMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_face_count": ("INT", {"default": 4096, "min": 100, "max": 100000, "step": 100}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "remesh"
    CATEGORY = "PHRenderFormer"

    def remesh(self, mesh, target_face_count):
        import pymeshlab
        
        new_meshes = []
        pbar = comfy.utils.ProgressBar(len(mesh["meshes"]))
        for m in mesh["meshes"]:
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=m.vertices, face_matrix=m.faces))
            
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
            new_meshes.append(processed_trimesh)
            pbar.update(1)

        new_mesh_data = mesh.copy()
        new_mesh_data["meshes"] = new_meshes
        
        return (new_mesh_data,)

class RenderFormerRandomizeColors:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "mode": (["per-triangle", "per-object"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "randomize"
    CATEGORY = "PHRenderFormer"

    def randomize(self, mesh, mode, seed, max_brightness):
        if not mesh["meshes"]:
            return (mesh,)

        # Create a deep copy of the materials to avoid modifying the original input
        new_materials = [mat.copy() for mat in mesh["materials"]]
        rng = np.random.default_rng(seed)

        for material in new_materials:
            if mode == "per-object":
                # Generate a single random color and apply it to the diffuse property
                color = rng.random(size=3) * max_brightness
                material['diffuse'] = color.tolist()
                # Ensure the per-triangle seed is disabled
                material['rand_tri_diffuse_seed'] = None
            
            elif mode == "per-triangle":
                # Use the built-in per-triangle randomization by setting the seed
                material['rand_tri_diffuse_seed'] = seed
                material['random_diffuse_max'] = max_brightness

        # Return a new mesh dictionary with the updated materials
        new_mesh_data = {
            "meshes": mesh["meshes"],
            "materials": new_materials,
            "transforms": mesh["transforms"]
        }
        
        return (new_mesh_data,)

class RenderFormerCamera:
    """
    Defines the camera position, orientation, and field of view.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "camera_pos_y": ("FLOAT", {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "camera_pos_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "camera_look_at_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "camera_look_at_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "camera_look_at_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "fov": ("FLOAT", {"default": 37.5, "min": 1.0, "max": 179.0, "step": 1.0, "tooltip": "Field of View in degrees"}),
            }
        }

    RETURN_TYPES = ("CAMERA",)
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

class RenderFormerCameraTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_camera": ("CAMERA",),
                "end_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_pos_y": ("FLOAT", {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_pos_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_look_at_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_look_at_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_look_at_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_fov": ("FLOAT", {"default": 37.5, "min": 1.0, "max": 179.0, "step": 1.0}),
                "num_frames": ("INT", {"default": 24, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("CAMERA_SEQUENCE",)
    FUNCTION = "get_camera_sequence"
    CATEGORY = "PHRenderFormer"

    def get_camera_sequence(self, start_camera, end_pos_x, end_pos_y, end_pos_z,
                            end_look_at_x, end_look_at_y, end_look_at_z, end_fov, num_frames):
        
        end_camera = {
            "position": [end_pos_x, end_pos_y, end_pos_z],
            "look_at": [end_look_at_x, end_look_at_y, end_look_at_z],
            "fov": end_fov
        }

        # The sequence now only contains the start and end keyframes
        key_frames = [start_camera, end_camera]
        
        camera_output = {
            "sequence": key_frames,
            "num_frames": num_frames
        }
        
        return (camera_output,)

class RenderFormerLighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emissive_rgb": ("STRING", {"default": "255, 255, 255", "multiline": False, "tooltip": "Emissive color (R, G, B) from 0-255"}),
                "emissive_strength": ("FLOAT", {"default": 5000.0, "min": 0.0, "max": 100000.0, "step": 10.0}),
                "trans_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "trans_z": ("FLOAT", {"default": 2.1, "min": -10.0, "max": 10.0, "step": 0.001}),
                "rot_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LIGHTING",)
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
                "lighting_1": ("LIGHTING",),
            },
            "optional": {
                "lighting_2": ("LIGHTING",),
                "lighting_3": ("LIGHTING",),
                "lighting_4": ("LIGHTING",),
                "lighting_5": ("LIGHTING",),
                "lighting_6": ("LIGHTING",),
                "lighting_7": ("LIGHTING",),
                "lighting_8": ("LIGHTING",),
            }
        }

    RETURN_TYPES = ("LIGHTING",)
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
                "mesh": ("MESH",),
                "lighting": ("LIGHTING",),
            },
            "optional": {
                "camera": ("CAMERA",),
                "camera_sequence": ("CAMERA_SEQUENCE",),
                "add_default_background": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SCENE", "SCENE_SEQUENCE",)
    FUNCTION = "build_scene"
    CATEGORY = "PHRenderFormer"

    def build_scene(self, mesh, lighting, camera=None, camera_sequence=None, add_default_background=False):
        output_scene = None
        output_sequence = None

        if camera:
            output_scene = self._build_single_scene(mesh, camera, lighting, add_default_background)

        if camera_sequence:
            key_frames = camera_sequence["sequence"]
            interpolation_frames = camera_sequence["num_frames"]

            if interpolation_frames > 0 and len(key_frames) > 1:
                print(f"PHRenderFormer: Building video scene with {interpolation_frames} interpolated frames from {len(key_frames)} keyframes.")
                
                base_camera = key_frames[0]
                base_scene_data = self._build_single_scene(mesh, base_camera, lighting, add_default_background)
                
                if not base_scene_data:
                    raise Exception("Failed to build the base scene for interpolation.")

                base_triangles = base_scene_data['triangles']
                base_texture = base_scene_data['texture']
                base_mask = base_scene_data['mask']
                base_vn = base_scene_data['vn']

                interpolated_camera_sequence = []
                num_keyframes = len(key_frames)
                
                for i in range(interpolation_frames):
                    t = i / (interpolation_frames - 1) if interpolation_frames > 1 else 0
                    segment_float = t * (num_keyframes - 1)
                    segment_index = min(int(segment_float), num_keyframes - 2)
                    local_t = segment_float - segment_index
                    start_cam = key_frames[segment_index]
                    end_cam = key_frames[segment_index + 1]

                    interp_pos = np.array(start_cam["position"], dtype=np.float32) + local_t * (np.array(end_cam["position"], dtype=np.float32) - np.array(start_cam["position"], dtype=np.float32))
                    interp_look_at = np.array(start_cam["look_at"], dtype=np.float32) + local_t * (np.array(end_cam["look_at"], dtype=np.float32) - np.array(start_cam["look_at"], dtype=np.float32))
                    interp_fov = np.float32(start_cam["fov"]) + local_t * (np.float32(end_cam["fov"]) - np.float32(start_cam["fov"]))
                    
                    interpolated_camera_sequence.append({
                        "position": interp_pos.tolist(),
                        "look_at": interp_look_at.tolist(),
                        "fov": interp_fov
                    })

                final_scenes = []
                pbar = comfy.utils.ProgressBar(len(interpolated_camera_sequence))
                for cam in interpolated_camera_sequence:
                    c2w = look_at_to_c2w(cam["position"], cam["look_at"])
                    fov = cam["fov"]
                    
                    scene = {
                        'triangles': base_triangles.clone(), 'texture': base_texture.clone(),
                        'mask': base_mask.clone(), 'vn': base_vn.clone(),
                        'c2w': torch.from_numpy(c2w).unsqueeze(0).unsqueeze(0),
                        'fov': torch.tensor([[[fov]]], dtype=torch.float32),
                    }
                    final_scenes.append(scene)
                    pbar.update(1)
                output_sequence = final_scenes
            else:
                total_steps = len(key_frames)
                pbar = comfy.utils.ProgressBar(total_steps)
                scene_sequence_out = []
                for cam in key_frames:
                    scene = self._build_single_scene(mesh, cam, lighting, add_default_background)
                    if scene:
                        scene_sequence_out.append(scene)
                    pbar.update(1)
                output_sequence = scene_sequence_out

        return (output_scene, output_sequence)

    def _build_single_scene(self, mesh, camera, lighting, add_default_background):
        """Helper function to build a single scene. Factored out to be reusable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # --- Pre-copy all mesh files ---
            for i, (mesh_obj, _, _) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
                temp_mesh_path = tmpdir_path / f"main_object_{i}.obj"
                mesh_obj.export(temp_mesh_path)
            
            template_mesh_root = Path(__file__).parent / "renderformer" / "examples"
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

            # --- Process camera frame ---
            config_data = {
                "scene_name": "built_scene_frame", "version": "1.0", "objects": {},
                "cameras": [{"position": camera["position"], "look_at": camera["look_at"], "up": [0.0, 0.0, 1.0], "fov": camera["fov"]}]
            }

            for i, (mesh_obj, material, transform) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
                obj_key = f"main_object_{i}"
                final_material = {
                    "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1], "roughness": 0.7,
                    "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None
                }
                if material:
                    for key in ["diffuse", "specular", "roughness", "emissive"]:
                        if key in material: final_material[key] = material[key]
                config_data["objects"][obj_key] = {"mesh_path": f"{obj_key}.obj", "transform": transform, "material": final_material}

            if lighting and isinstance(lighting, list):
                for i, light_def in enumerate(lighting):
                    config_data["objects"][f"comfy_light_{i}"] = {"mesh_path": "templates/lighting/tri.obj", "transform": light_def["transform"], "material": light_def["material"]}
            
            if add_default_background:
                background_files = ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]
                for i, bg_file in enumerate(background_files):
                    config_data["objects"][f"comfy_default_background_{i}"] = {
                        "mesh_path": f"templates/backgrounds/{bg_file}",
                        "transform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.5, 0.5, 0.5], "normalize": False},
                        "material": {"diffuse": [0.4, 0.4, 0.4], "specular": [0.0, 0.0, 0.0], "random_diffuse_max": 0.4, "roughness": 0.99, "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None}
                    }

            # --- Conversion logic for a single frame ---
            scene_config_dir = tmpdir_path
            original_export, original_load = trimesh.exchange.export.export_mesh, trimesh.load
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

            trimesh.exchange.export.export_mesh, trimesh.load = patched_export, patched_load

            try:
                scene_config = from_dict(data_class=SceneConfig, data=config_data, config=Config(check_types=True, strict=True))
                with io.BytesIO() as h5_buffer:
                    temp_mesh_dir_name = "temp_mesh_dir"
                    generate_scene_mesh(scene_config, f"{temp_mesh_dir_name}/scene.obj", str(scene_config_dir))
                    
                    def patched_save_to_h5(scene_config, mesh_path, output_h5_buffer):
                        all_triangles, all_vn, all_texture = [], [], []
                        size = 32
                        mask_np = np.zeros((size, size), dtype=bool)
                        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                        mask_np[x + y <= size] = 1
                        
                        split_mesh_path_prefix = os.path.dirname(mesh_path) + '/split'
                        for obj_key, obj_config in scene_config.objects.items():
                            mesh = trimesh.load(f'{split_mesh_path_prefix}/{obj_key}.obj', process=False, force='mesh')
                            triangles, vn = mesh.triangles, mesh.vertex_normals[mesh.faces]
                            material_config = obj_config.material
                            diffuse = mesh.visual.face_colors[..., :3] / 255.
                            specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
                            roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
                            normal = np.array([0.5, 0.5, 1.0])[None].repeat(triangles.shape[0], axis=0)
                            irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
                            texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
                            texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
                            texture[:, :, ~mask_np] = 0.0
                            all_triangles.append(triangles)
                            all_vn.append(vn)
                            all_texture.append(texture)
                        
                        all_triangles, all_vn, all_texture = np.concatenate(all_triangles, axis=0), np.concatenate(all_vn, axis=0), np.concatenate(all_texture, axis=0)
                        all_c2w, all_fov = [], []
                        for camera_config in scene_config.cameras:
                            all_c2w.append(look_at_to_c2w(camera_config.position, camera_config.look_at))
                            all_fov.append(camera_config.fov)
                        
                        with h5py.File(output_h5_buffer, "w") as f:
                            f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip")
                            f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip")
                            f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip")
                            f.create_dataset("c2w", data=np.stack(all_c2w).astype(np.float32), compression="gzip")
                            f.create_dataset("fov", data=np.array(all_fov).astype(np.float32), compression="gzip")

                    patched_save_to_h5(scene_config, f"{temp_mesh_dir_name}/scene.obj", h5_buffer)

                    h5_buffer.seek(0)
                    with h5py.File(h5_buffer, 'r') as f:
                        c2w_data, fov_data = f['c2w'][:], f['fov'][:]
                        scene = {
                            'triangles': torch.from_numpy(f['triangles'][:]).unsqueeze(0),
                            'texture': torch.from_numpy(f['texture'][:]).unsqueeze(0),
                            'mask': torch.ones(f['triangles'][:].shape[0], dtype=torch.bool).unsqueeze(0),
                            'vn': torch.from_numpy(f['vn'][:]).unsqueeze(0),
                            'c2w': torch.from_numpy(c2w_data).unsqueeze(0),
                            'fov': torch.from_numpy(fov_data).unsqueeze(0).unsqueeze(-1),
                        }
                        return scene
            except Exception as e:
                print(f"Error processing frame: {e}")
                return None
            finally:
                trimesh.load, trimesh.exchange.export.export_mesh = original_load, original_export

class RenderFormerMeshCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_1": ("MESH",),
            },
            "optional": {
                "mesh_2": ("MESH",),
                "mesh_3": ("MESH",),
                "mesh_4": ("MESH",),
                "mesh_5": ("MESH",),
                "mesh_6": ("MESH",),
                "mesh_7": ("MESH",),
                "mesh_8": ("MESH",),
            }
        }

    RETURN_TYPES = ("MESH",)
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
            mesh = kwargs.get(key)
            if mesh and isinstance(mesh, dict) and "meshes" in mesh and "materials" in mesh and "transforms" in mesh:
                all_meshes.extend(mesh["meshes"])
                all_materials.extend(mesh["materials"])
                all_transforms.extend(mesh["transforms"])
        
        combined_ph_mesh = {"meshes": all_meshes, "materials": all_materials, "transforms": all_transforms}

        return (combined_ph_mesh,)

    def _build_single_scene(self, mesh, camera, lighting, add_default_background):
        """Helper function to build a single scene. Factored out to be reusable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # --- Pre-copy all mesh files ---
            for i, (mesh_obj, _, _) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
                temp_mesh_path = tmpdir_path / f"main_object_{i}.obj"
                mesh_obj.export(temp_mesh_path)
            
            template_mesh_root = Path(__file__).parent / "renderformer" / "examples"
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

            # --- Process camera frame ---
            config_data = {
                "scene_name": "built_scene_frame", "version": "1.0", "objects": {},
                "cameras": [{"position": camera["position"], "look_at": camera["look_at"], "up": [0.0, 0.0, 1.0], "fov": camera["fov"]}]
            }

            for i, (mesh_obj, material, transform) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
                obj_key = f"main_object_{i}"
                final_material = {
                    "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1], "roughness": 0.7,
                    "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None
                }
                if material:
                    for key in ["diffuse", "specular", "roughness", "emissive"]:
                        if key in material: final_material[key] = material[key]
                config_data["objects"][obj_key] = {"mesh_path": f"{obj_key}.obj", "transform": transform, "material": final_material}

            if lighting and isinstance(lighting, list):
                for i, light_def in enumerate(lighting):
                    config_data["objects"][f"comfy_light_{i}"] = {"mesh_path": "templates/lighting/tri.obj", "transform": light_def["transform"], "material": light_def["material"]}
            
            if add_default_background:
                background_files = ["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]
                for i, bg_file in enumerate(background_files):
                    config_data["objects"][f"comfy_default_background_{i}"] = {
                        "mesh_path": f"templates/backgrounds/{bg_file}",
                        "transform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.5, 0.5, 0.5], "normalize": False},
                        "material": {"diffuse": [0.4, 0.4, 0.4], "specular": [0.0, 0.0, 0.0], "random_diffuse_max": 0.4, "roughness": 0.99, "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None}
                    }

            # --- Conversion logic for a single frame ---
            scene_config_dir = tmpdir_path
            original_export, original_load = trimesh.exchange.export.export_mesh, trimesh.load
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

            trimesh.exchange.export.export_mesh, trimesh.load = patched_export, patched_load

            try:
                scene_config = from_dict(data_class=SceneConfig, data=config_data, config=Config(check_types=True, strict=True))
                with io.BytesIO() as h5_buffer:
                    temp_mesh_dir_name = "temp_mesh_dir"
                    generate_scene_mesh(scene_config, f"{temp_mesh_dir_name}/scene.obj", str(scene_config_dir))
                    
                    def patched_save_to_h5(scene_config, mesh_path, output_h5_buffer):
                        all_triangles, all_vn, all_texture = [], [], []
                        size = 32
                        mask_np = np.zeros((size, size), dtype=bool)
                        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                        mask_np[x + y <= size] = 1
                        
                        split_mesh_path_prefix = os.path.dirname(mesh_path) + '/split'
                        for obj_key, obj_config in scene_config.objects.items():
                            mesh = trimesh.load(f'{split_mesh_path_prefix}/{obj_key}.obj', process=False, force='mesh')
                            triangles, vn = mesh.triangles, mesh.vertex_normals[mesh.faces]
                            material_config = obj_config.material
                            diffuse = mesh.visual.face_colors[..., :3] / 255.
                            specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
                            roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
                            normal = np.array([0.5, 0.5, 1.0])[None].repeat(triangles.shape[0], axis=0)
                            irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
                            texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
                            texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
                            texture[:, :, ~mask_np] = 0.0
                            all_triangles.append(triangles)
                            all_vn.append(vn)
                            all_texture.append(texture)
                        
                        all_triangles, all_vn, all_texture = np.concatenate(all_triangles, axis=0), np.concatenate(all_vn, axis=0), np.concatenate(all_texture, axis=0)
                        all_c2w, all_fov = [], []
                        for camera_config in scene_config.cameras:
                            all_c2w.append(look_at_to_c2w(camera_config.position, camera_config.look_at))
                            all_fov.append(camera_config.fov)
                        
                        with h5py.File(output_h5_buffer, "w") as f:
                            f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip")
                            f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip")
                            f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip")
                            f.create_dataset("c2w", data=np.stack(all_c2w).astype(np.float32), compression="gzip")
                            f.create_dataset("fov", data=np.array(all_fov).astype(np.float32), compression="gzip")

                    patched_save_to_h5(scene_config, f"{temp_mesh_dir_name}/scene.obj", h5_buffer)

                    h5_buffer.seek(0)
                    with h5py.File(h5_buffer, 'r') as f:
                        c2w_data, fov_data = f['c2w'][:], f['fov'][:]
                        scene = {
                            'triangles': torch.from_numpy(f['triangles'][:]).unsqueeze(0),
                            'texture': torch.from_numpy(f['texture'][:]).unsqueeze(0),
                            'mask': torch.ones(f['triangles'][:].shape[0], dtype=torch.bool).unsqueeze(0),
                            'vn': torch.from_numpy(f['vn'][:]).unsqueeze(0),
                            'c2w': torch.from_numpy(c2w_data).unsqueeze(0),
                            'fov': torch.from_numpy(fov_data).unsqueeze(0).unsqueeze(-1),
                        }
                        return scene
            except Exception as e:
                print(f"Error processing frame: {e}")
                return None
            finally:
                trimesh.load, trimesh.exchange.export.export_mesh = original_load, original_export

class RenderFormerGenerator:
    """
    Renders an image using the RenderFormer pipeline.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tone_mapper": (["none", "agx", "filmic", "pbr_neutral"], {"default": "agx"}),
            },
            "optional": {
                "scene": ("SCENE",),
                "scene_sequence": ("SCENE_SEQUENCE",),
                "resolution_vid": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("IMAGE", "IMAGES",)
    FUNCTION = "generate"
    CATEGORY = "PHRenderFormer"

    def generate(self, model, resolution, tone_mapper, scene=None, scene_sequence=None, resolution_vid=512):
        device = mm.get_torch_device()
        pipeline = model["pipeline"]
        torch_dtype = model["torch_dtype"]
        
        output_image = None
        output_video = None

        # --- Single Image Generation ---
        if scene is not None:
            pbar = comfy.utils.ProgressBar(4)
            scene_on_device = {k: v.to(device) for k, v in scene.items()}
            pbar.update(1)

            rendered_imgs = pipeline(
                triangles=scene_on_device['triangles'], texture=scene_on_device['texture'],
                mask=scene_on_device['mask'], vn=scene_on_device['vn'],
                c2w=scene_on_device['c2w'], fov=scene_on_device['fov'],
                resolution=resolution, torch_dtype=torch_dtype,
            )
            pbar.update(1)

            rendered_imgs = torch.pow(10., rendered_imgs) - 1.
            hdr_img_tensor = rendered_imgs[0, 0].cpu()
            hdr_img = hdr_img_tensor.numpy().astype(np.float32)

            tm = None
            if tone_mapper != 'none':
                from simple_ocio import ToneMapper
                tm_name = 'Khronos PBR Neutral' if tone_mapper == 'pbr_neutral' else tone_mapper
                tm = ToneMapper(tm_name)
                ldr_img = tm.hdr_to_ldr(hdr_img)
            else:
                ldr_img = np.clip(hdr_img, 0, 1)
            
            output_image = torch.from_numpy(ldr_img).unsqueeze(0)
            pbar.update(1)

        # --- Video (Batched) Generation ---
        if scene_sequence is not None:
            batch_size = 1
            all_frames = []
            num_scenes = len(scene_sequence)
            pbar = comfy.utils.ProgressBar(num_scenes)

            for i in range(0, num_scenes, batch_size):
                batch_chunk = scene_sequence[i:i + batch_size]
                
                batch_triangles = torch.cat([s['triangles'] for s in batch_chunk], dim=0).to(device)
                batch_texture = torch.cat([s['texture'] for s in batch_chunk], dim=0).to(device)
                batch_mask = torch.cat([s['mask'] for s in batch_chunk], dim=0).to(device)
                batch_vn = torch.cat([s['vn'] for s in batch_chunk], dim=0).to(device)
                batch_c2w = torch.cat([s['c2w'] for s in batch_chunk], dim=0).to(device)
                batch_fov = torch.cat([s['fov'] for s in batch_chunk], dim=0).to(device)

                rendered_imgs = pipeline(
                    triangles=batch_triangles, texture=batch_texture, mask=batch_mask,
                    vn=batch_vn, c2w=batch_c2w, fov=batch_fov,
                    resolution=resolution_vid, torch_dtype=torch_dtype,
                )

                rendered_imgs = torch.pow(10., rendered_imgs) - 1.
                
                tm = None
                if tone_mapper != 'none':
                    from simple_ocio import ToneMapper
                    tm_name = 'Khronos PBR Neutral' if tone_mapper == 'pbr_neutral' else tone_mapper
                    tm = ToneMapper(tm_name)

                for j in range(rendered_imgs.shape[0]):
                    hdr_img_tensor = rendered_imgs[j, 0].cpu()
                    hdr_img = hdr_img_tensor.numpy().astype(np.float32)
                    ldr_img = tm.hdr_to_ldr(hdr_img) if tm else np.clip(hdr_img, 0, 1)
                    all_frames.append(ldr_img)
                    pbar.update(1)

            if all_frames:
                output_video = torch.from_numpy(np.array(all_frames).astype(np.float32))

        # Return based on what was processed
        if output_image is not None and output_video is not None:
            return (output_image, output_video)
        elif output_image is not None:
            return (output_image, torch.empty(0))
        elif output_video is not None:
            return (torch.empty(0), output_video)
        else:
            return (torch.empty(0), torch.empty(0))

class RenderFormerExampleScene:
    """
    Loads an example scene from the RenderFormer examples directory.
    This node programmatically executes the equivalent of 'convert_scene.py'
    to generate the scene data in memory, replicating the original workflow.
    """
    
    EXAMPLE_DIR = Path(__file__).parent / "renderformer" / "examples"
    EXAMPLE_FILES = [f.name for f in EXAMPLE_DIR.glob("*.json")]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_file": (cls.EXAMPLE_FILES, )
            }
        }

    RETURN_TYPES = ("SCENE",)
    FUNCTION = "load_example_scene"
    CATEGORY = "PHRenderFormer/Test"

    def load_example_scene(self, scene_file):
        # Define paths
        base_path = Path(__file__).parent / "renderformer"
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

    RETURN_TYPES = ("SCENE",)
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
            template_mesh_root = Path(__file__).parent / "renderformer" / "examples"

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
    "RenderFormerCameraTarget": RenderFormerCameraTarget,
    "RenderFormerLighting": RenderFormerLighting,
    "RenderFormerSceneBuilder": RenderFormerSceneBuilder,
    "RenderFormerGenerator": RenderFormerGenerator,
    "RenderFormerLoadMesh": RenderFormerLoadMesh,
    "RenderFormerRemeshMesh": RenderFormerRemeshMesh,
    "RenderFormerRandomizeColors": RenderFormerRandomizeColors,
    "RenderFormerExampleScene": RenderFormerExampleScene,
    "RenderFormerFromJSON": RenderFormerFromJSON,
    "RenderFormerMeshCombine": RenderFormerMeshCombine,
    "RenderFormerLightingCombine": RenderFormerLightingCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenderFormerModelLoader": "RenderFormer Model Loader",
    "RenderFormerCamera": "RenderFormer Camera",
    "RenderFormerCameraTarget": "RenderFormer Camera Target",
    "RenderFormerLighting": "RenderFormer Lighting",
    "RenderFormerSceneBuilder": "RenderFormer Scene Builder",
    "RenderFormerGenerator": "RenderFormer Sampler",
    "RenderFormerLoadMesh": "RenderFormer Mesh Loader",
    "RenderFormerRemeshMesh": "RenderFormer Remesh",
    "RenderFormerRandomizeColors": "RenderFormer Random Colors",
    "RenderFormerExampleScene": "RenderFormer Example Scene",
    "RenderFormerFromJSON": "RenderFormer From JSON",
    "RenderFormerMeshCombine": "RenderFormer Mesh Combine",
    "RenderFormerLightingCombine": "RenderFormer Lighting Combine",
}