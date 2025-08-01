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
import copy
import uuid
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
                "mode": (["per-object", "per-shading", "per-triangle"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999}),
                "max_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "randomize"
    CATEGORY = "PHRenderFormer"

    def randomize(self, mesh, mode, seed, max_brightness):
        if not mesh["meshes"]:
            return (mesh,)

        # Create a deep copy to avoid modifying the original inputs
        new_materials = [mat.copy() for mat in mesh["materials"]]
        new_meshes = [m.copy() for m in mesh["meshes"]]
        
        rng = np.random.default_rng(seed)

        if mode == "per-triangle":
            # This mode now correctly uses the material properties.
            for material in new_materials:
                material['rand_tri_diffuse_seed'] = seed
                material['random_diffuse_max'] = max_brightness
                material['random_diffuse_type'] = 'per-triangle'
                # Ensure direct diffuse is not overriding this
                if 'diffuse' in material:
                    del material['diffuse']
        
        else:
            # Handle per-object and per-shading, which modify the material properties
            for material in new_materials:
                if mode == "per-object":
                    color = rng.random(size=3) * max_brightness
                    material['diffuse'] = color.tolist()
                    material['rand_tri_diffuse_seed'] = None
                
                elif mode == "per-shading":
                    material['rand_tri_diffuse_seed'] = seed
                    material['random_diffuse_max'] = max_brightness
                    material['random_diffuse_type'] = 'per-shading-group'
                    # Ensure direct diffuse is not overriding this
                    if 'diffuse' in material:
                        del material['diffuse']

        # Return a new mesh dictionary with the updated components
        new_mesh_data = {
            "meshes": new_meshes,
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
            }
        }

    RETURN_TYPES = ("CAMERA_SEQUENCE",)
    FUNCTION = "get_camera_sequence"
    CATEGORY = "PHRenderFormer"

    def get_camera_sequence(self, start_camera, end_pos_x, end_pos_y, end_pos_z,
                            end_look_at_x, end_look_at_y, end_look_at_z, end_fov):
        
        end_camera = {
            "position": [end_pos_x, end_pos_y, end_pos_z],
            "look_at": [end_look_at_x, end_look_at_y, end_look_at_z],
            "fov": end_fov
        }

        # The sequence now only contains the start and end keyframes
        key_frames = [start_camera, end_camera]
        
        camera_output = {
            "sequence": key_frames
        }
        
        return (camera_output,)

class RenderFormerLighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emissive_rgb": ("STRING", {"default": "255, 255, 255", "multiline": False, "tooltip": "Emissive color (R, G, B) from 0-255"}),
                "emissive_strength": ("FLOAT", {"default": 5000.0, "min": 0.0, "max": 100000.0, "step": 10.0}),
                "light_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "light_pos_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "light_pos_z": ("FLOAT", {"default": 2.1, "min": -10.0, "max": 10.0, "step": 0.001}),
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

    def get_lighting(self, emissive_rgb, emissive_strength, light_pos_x, light_pos_y, light_pos_z, rot_x, rot_y, rot_z, scale):
        transform = {
            "translation": [light_pos_x, light_pos_y, light_pos_z],
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
            "material": material,
            "ph_uuid": str(uuid.uuid4()) # Add a unique ID for reliable animation targeting
        }]
        
        return (light_definition,)

class RenderFormerLightingTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_lighting": ("LIGHTING",),
                "end_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_pos_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_pos_z": ("FLOAT", {"default": 2.1, "min": -10.0, "max": 10.0, "step": 0.01}),
                "end_rot_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_scale": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 10.0, "step": 0.01}),
                "end_emissive_strength": ("FLOAT", {"default": 5000.0, "min": 0.0, "max": 100000.0, "step": 10.0}),
            }
        }
    
    RETURN_TYPES = ("LIGHTING",)
    FUNCTION = "get_lighting_sequence"
    CATEGORY = "PHRenderFormer"

    def get_lighting_sequence(self, start_lighting, end_pos_x, end_pos_y, end_pos_z, end_rot_x, end_rot_y, end_rot_z, end_scale, end_emissive_strength):
        if not start_lighting or not isinstance(start_lighting, list) or not start_lighting[0]:
            raise Exception("RenderFormerLightingTarget: Invalid start_lighting input.")

        start_light_def = start_lighting[0]
        
        end_light_def = copy.deepcopy(start_light_def)
        
        end_light_def["transform"]["translation"] = [end_pos_x, end_pos_y, end_pos_z]
        end_light_def["transform"]["rotation"] = [end_rot_x, end_rot_y, end_rot_z]
        end_light_def["transform"]["scale"] = [end_scale, end_scale, end_scale]
        
        start_emissive = start_light_def["material"]["emissive"]
        base_color = [c / max(start_emissive) if max(start_emissive) > 0 else 0 for c in start_emissive]
        end_light_def["material"]["emissive"] = [c * end_emissive_strength for c in base_color]

        # The output is a dictionary containing both start and end states.
        # It is returned under the generic 'LIGHTING' type to allow it to be connected
        # to the Combine node.
        lighting_output = {
            "start_lights": [start_light_def],
            "end_lights": [end_light_def]
        }
        
        return (lighting_output,)


class RenderFormerLightingCombine:
    @classmethod
    def INPUT_TYPES(cls):
        # All inputs are now of the generic 'LIGHTING' type.
        # The node's logic will distinguish between static lists and animated dicts.
        return {
            "optional": {
                "lighting_1": ("LIGHTING",), "lighting_2": ("LIGHTING",),
                "lighting_3": ("LIGHTING",), "lighting_4": ("LIGHTING",),
                "lighting_5": ("LIGHTING",), "lighting_6": ("LIGHTING",),
                "lighting_7": ("LIGHTING",), "lighting_8": ("LIGHTING",),
            }
        }

    RETURN_TYPES = ("LIGHTING", "LIGHTING",)
    RETURN_NAMES = ("LIGHTING", "LIGHTING_SEQUENCE",)
    FUNCTION = "combine_lighting"
    CATEGORY = "PHRenderFormer"

    def combine_lighting(self, **kwargs):
        all_start_lights = []
        animated_end_map = {}
        
        # First pass: Collect all start lights and map animated end states by UUID
        for i in range(1, 9):
            light_input = kwargs.get(f"lighting_{i}")
            if not light_input:
                continue

            # It's an animation sequence (a dictionary)
            if isinstance(light_input, dict):
                start_defs = light_input.get("start_lights", [])
                end_defs = light_input.get("end_lights", [])
                all_start_lights.extend(start_defs)
                # Map the end state of each light in the sequence by its UUID
                for j, start_def in enumerate(start_defs):
                    if "ph_uuid" in start_def and j < len(end_defs):
                        animated_end_map[start_def["ph_uuid"]] = end_defs[j]
            
            # It's a static light (a list of light definitions)
            elif isinstance(light_input, list):
                all_start_lights.extend(light_input)

        # Second pass: Build the end_frame_lights list to ensure 1-to-1 correspondence
        all_end_lights = []
        for start_light in all_start_lights:
            uuid = start_light.get("ph_uuid")
            # If this light has a defined end state in our map, use it.
            if uuid and uuid in animated_end_map:
                all_end_lights.append(animated_end_map[uuid])
            # Otherwise, it's a static light, so duplicate its start state for the end frame.
            else:
                all_end_lights.append(copy.deepcopy(start_light))

        # The combined output is a dictionary, but it's returned as type 'LIGHTING'.
        lighting_sequence = {
            "start_lights": all_start_lights,
            "end_lights": all_end_lights
        }

        # Return the start frame lights first (for the top output) and then the full sequence.
        return (all_start_lights, lighting_sequence)

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
                "camera": ("CAMERA",),
            },
            "optional": {
                "camera_sequence": ("CAMERA_SEQUENCE",),
                "camera_sequence": ("CAMERA_SEQUENCE",),
                "lighting_sequence": ("LIGHTING",),
                "mesh_sequence": ("MESH",),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "add_default_background": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SCENE", "SCENE_SEQUENCE",)
    FUNCTION = "build_scene"
    CATEGORY = "PHRenderFormer"

    def build_scene(self, mesh, lighting, camera, camera_sequence=None, lighting_sequence=None, mesh_sequence=None, num_frames=1, add_default_background=False):
        output_scene = None
        output_sequence = None

        is_animation = num_frames > 1 and (camera_sequence is not None or lighting_sequence is not None or mesh_sequence is not None)

        if not is_animation:
            # Build a single static scene
            with tempfile.TemporaryDirectory() as tmpdir:
                output_scene = self._build_single_scene(tmpdir, mesh, camera, lighting, add_default_background)
            return (output_scene, None)

        # --- Animation Sequence Building ---
        print(f"PHRenderFormer: Building video scene with {num_frames} frames.")
        pbar = comfy.utils.ProgressBar(num_frames)

        # --- Determine Start and End Configurations ---
        # The 'camera' and 'lighting' inputs are ALWAYS the start frame.
        # For animations, the sequence inputs are the source of truth for BOTH start and end frames.
        # If a sequence isn't provided, we fall back to the static input for that element.
        start_cam_config = camera_sequence["sequence"][0] if camera_sequence and "sequence" in camera_sequence and camera_sequence["sequence"] else camera
        end_cam_config = camera_sequence["sequence"][-1] if camera_sequence and "sequence" in camera_sequence and camera_sequence["sequence"] else camera

        start_lighting_config = lighting_sequence["start_lights"] if lighting_sequence and isinstance(lighting_sequence, dict) and "start_lights" in lighting_sequence else lighting
        end_lighting_config = lighting_sequence["end_lights"] if lighting_sequence and isinstance(lighting_sequence, dict) and "end_lights" in lighting_sequence else lighting

        # --- Mesh Configuration: Smartly combine static and animated meshes ---
        if not (mesh_sequence and isinstance(mesh_sequence, dict)):
            # If there's no animation sequence, we just use the static meshes.
            start_mesh_config = mesh
            end_mesh_config = copy.deepcopy(mesh) if mesh else None
        else:
            # An animation sequence exists. We need to filter out the animated mesh's
            # static counterpart from the main 'mesh' input to avoid duplication.
            static_meshes = mesh.get("meshes", []) if mesh else []
            static_materials = mesh.get("materials", []) if mesh else []
            static_transforms = mesh.get("transforms", []) if mesh else []

            animated_start_data = mesh_sequence.get("start_meshes", {})
            animated_start_meshes = animated_start_data.get("meshes", [])

            # These lists will hold the static meshes that are NOT being animated.
            clean_static_meshes, clean_static_materials, clean_static_transforms = [], [], []

            # Iterate through the static meshes and keep only the ones that are not
            # also present in the animated list (by object identity).
            for i, static_mesh_obj in enumerate(static_meshes):
                if not any(static_mesh_obj is anim_mesh_obj for anim_mesh_obj in animated_start_meshes):
                    clean_static_meshes.append(static_mesh_obj)
                    clean_static_materials.append(static_materials[i])
                    clean_static_transforms.append(static_transforms[i])

            # The start frame is the clean static list plus the start state of the animated meshes.
            start_mesh_config = {
                "meshes": clean_static_meshes + animated_start_data.get("meshes", []),
                "materials": clean_static_materials + animated_start_data.get("materials", []),
                "transforms": clean_static_transforms + animated_start_data.get("transforms", [])
            }

            # The end frame is a copy of the clean static list plus the end state of the animated meshes.
            animated_end_data = mesh_sequence.get("end_meshes", {})
            end_mesh_config = {
                "meshes": copy.deepcopy(clean_static_meshes) + animated_end_data.get("meshes", []),
                "materials": copy.deepcopy(clean_static_materials) + animated_end_data.get("materials", []),
                "transforms": copy.deepcopy(clean_static_transforms) + animated_end_data.get("transforms", [])
            }


        # --- Build Keyframe Scenes in completely isolated directories ---
        start_scene = None
        end_scene = None

        with tempfile.TemporaryDirectory() as start_tmpdir, tempfile.TemporaryDirectory() as end_tmpdir:
            # Build scene for the START frame
            print("PHRenderFormer: Building start frame...")
            start_scene = self._build_single_scene(start_tmpdir, start_mesh_config, start_cam_config, start_lighting_config, add_default_background)
            if not start_scene:
                raise Exception("Failed to build the start frame for interpolation.")
            pbar.update(1)

            # Build scene for the END frame
            print("PHRenderFormer: Building end frame...")
            end_scene = self._build_single_scene(end_tmpdir, end_mesh_config, end_cam_config, end_lighting_config, add_default_background)
            if not end_scene:
                raise Exception("Failed to build the end frame for interpolation.")
            pbar.update(1)

        # --- Interpolate Tensors ---
        final_scenes = []
        
        # This tensor is constant for all frames
        base_mask = start_scene['mask']

        # Ensure tensors are the same size before interpolating
        if start_scene['triangles'].shape != end_scene['triangles'].shape:
            raise RuntimeError(f"Triangle tensors must have the same shape for interpolation, but got {start_scene['triangles'].shape} and {end_scene['triangles'].shape}")
        if start_scene['vn'].shape != end_scene['vn'].shape:
            raise RuntimeError(f"Vertex normal tensors must have the same shape for interpolation, but got {start_scene['vn'].shape} and {end_scene['vn'].shape}")
        if start_scene['texture'].shape != end_scene['texture'].shape:
            raise RuntimeError(f"Texture tensors must have the same shape for interpolation, but got {start_scene['texture'].shape} and {end_scene['texture'].shape}")

        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0.0
            
            # Interpolate all tensors that can change between frames
            interp_triangles = torch.lerp(start_scene['triangles'], end_scene['triangles'], t)
            interp_vn = torch.lerp(start_scene['vn'], end_scene['vn'], t)
            interp_c2w = torch.lerp(start_scene['c2w'], end_scene['c2w'], t)
            interp_fov = torch.lerp(start_scene['fov'], end_scene['fov'], t)
            interp_texture = torch.lerp(start_scene['texture'], end_scene['texture'], t)

            # Assemble the new scene for the current frame
            frame_scene = {
                'triangles': interp_triangles,
                'vn': interp_vn,
                'mask': base_mask, # The mask of which triangles are visible is constant
                'texture': interp_texture,
                'c2w': interp_c2w,
                'fov': interp_fov,
            }
            final_scenes.append(frame_scene)
            if i > 1: # pbar already updated for start and end frames
                pbar.update(1)

        output_sequence = final_scenes
        return (None, output_sequence)

    def _build_single_scene(self, tmpdir, mesh, camera, lighting, add_default_background):
        """Helper function to build a single scene within a provided temporary directory."""
        tmpdir_path = Path(tmpdir)
        template_mesh_root = Path(__file__).parent / "renderformer" / "examples"

        # --- Configuration setup ---
        config_data = {
            "scene_name": "built_scene_frame", "version": "1.0", "objects": {},
            "cameras": [{"position": camera["position"], "look_at": camera["look_at"], "up": [0.0, 0.0, 1.0], "fov": camera["fov"]}]
        }

        # --- Write main meshes to temp dir and update config ---
        for i, (mesh_obj, material, transform) in enumerate(zip(mesh["meshes"], mesh["materials"], mesh["transforms"])):
            obj_key = f"main_object_{i}"
            temp_mesh_path = tmpdir_path / f"{obj_key}.obj"
            mesh_obj.export(temp_mesh_path)
            
            final_material = {
                "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1], "roughness": 0.7,
                "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None
            }
            if material:
                final_material.update(material)
            
            config_data["objects"][obj_key] = {
                "mesh_path": f"{obj_key}.obj", # Use relative path
                "transform": transform,
                "material": final_material
            }

        # --- Copy template meshes to temp dir and update config ---
        if lighting and isinstance(lighting, list):
            dest_light_path = tmpdir_path / "templates" / "lighting"
            dest_light_path.mkdir(parents=True, exist_ok=True)
            
            # Create a uniquely named copy of the light mesh for each light instance.
            # This is a robust way to prevent potential caching issues in the underlying
            # scene processing library, which might be caching results based on mesh_path.
            for i, light_def in enumerate(lighting):
                unique_light_mesh_name = f"tri_light_{i}.obj"
                shutil.copy(template_mesh_root / "templates" / "lighting" / "tri.obj", dest_light_path / unique_light_mesh_name)
                config_data["objects"][f"comfy_light_{i}"] = {
                    "mesh_path": f"templates/lighting/{unique_light_mesh_name}", # Use unique path
                    "transform": light_def["transform"],
                    "material": light_def["material"]
                }
        
        if add_default_background:
            dest_background_path = tmpdir_path / "templates" / "backgrounds"
            dest_background_path.mkdir(parents=True, exist_ok=True)
            for i, bg_file in enumerate(["plane.obj", "wall0.obj", "wall1.obj", "wall2.obj"]):
                shutil.copy(template_mesh_root / "templates" / "backgrounds" / bg_file, dest_background_path / bg_file)
                config_data["objects"][f"comfy_default_background_{i}"] = {
                    "mesh_path": f"templates/backgrounds/{bg_file}", # Use relative path
                    "transform": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.5, 0.5, 0.5], "normalize": False},
                    "material": {"diffuse": [0.4, 0.4, 0.4], "specular": [0.0, 0.0, 0.0], "random_diffuse_max": 0.4, "roughness": 0.99, "emissive": [0.0, 0.0, 0.0], "smooth_shading": True, "rand_tri_diffuse_seed": None}
                }

        # --- Conversion logic using the temporary directory ---
        try:
            scene_config = from_dict(data_class=SceneConfig, data=config_data, config=Config(check_types=True, strict=True))
            with io.BytesIO() as h5_buffer:
                temp_mesh_dir_name = "temp_mesh_dir"
                # The scene_config_dir is now the root of our temp directory
                generate_scene_mesh(scene_config, f"{temp_mesh_dir_name}/scene.obj", str(tmpdir_path))
                
                split_mesh_path_prefix = tmpdir_path / temp_mesh_dir_name / 'split'
                
                all_triangles, all_vn, all_texture = [], [], []
                size = 32
                mask_np = np.zeros((size, size), dtype=bool)
                x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
                mask_np[x + y <= size] = 1

                for obj_key, obj_config in scene_config.objects.items():
                    # The obj_config.mesh_path is now an absolute path, so generate_scene_mesh will have used that.
                    # The split files are named after the object key, not the original filename.
                    mesh_file_path = split_mesh_path_prefix / f'{obj_key}.obj'
                    mesh = trimesh.load(str(mesh_file_path), process=False, force='mesh')
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
                
                all_triangles = np.concatenate(all_triangles, axis=0)
                all_vn = np.concatenate(all_vn, axis=0)
                all_texture = np.concatenate(all_texture, axis=0)
                
                all_c2w, all_fov = [], []
                for camera_config in scene_config.cameras:
                    all_c2w.append(look_at_to_c2w(camera_config.position, camera_config.look_at))
                    all_fov.append(camera_config.fov)
                
                with h5py.File(h5_buffer, "w") as f:
                    f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip")
                    f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip")
                    f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip")
                    f.create_dataset("c2w", data=np.stack(all_c2w).astype(np.float32), compression="gzip")
                    f.create_dataset("fov", data=np.array(all_fov).astype(np.float32), compression="gzip")

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
            import traceback
            print(f"PHRenderFormer ERROR in _build_single_scene: {e}")
            traceback.print_exc()
        return None

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

    RETURN_TYPES = ("MESH", "MESH",)
    RETURN_NAMES = ("MESH", "MESH_ANIMATED",)
    FUNCTION = "combine_meshes"
    CATEGORY = "PHRenderFormer"

    def combine_meshes(self, **kwargs):
        # Lists to hold the components for the start and end frames
        all_start_meshes, all_start_materials, all_start_transforms = [], [], []
        all_end_meshes, all_end_materials, all_end_transforms = [], [], []

        # Iterate through all possible mesh inputs
        for i in range(1, 9):
            mesh_input = kwargs.get(f"mesh_{i}")
            if not mesh_input:
                continue

            # Case 1: Input is an animated MESH object (previously MESH_SEQUENCE)
            if isinstance(mesh_input, dict) and "start_meshes" in mesh_input and "end_meshes" in mesh_input:
                start_data = mesh_input["start_meshes"]
                end_data = mesh_input["end_meshes"]
                
                all_start_meshes.extend(start_data["meshes"])
                all_start_materials.extend(start_data["materials"])
                all_start_transforms.extend(start_data["transforms"])
                
                all_end_meshes.extend(end_data["meshes"])
                all_end_materials.extend(end_data["materials"])
                all_end_transforms.extend(end_data["transforms"])

            # Case 2: Input is a static MESH object
            elif isinstance(mesh_input, dict) and "meshes" in mesh_input:
                # For a static mesh, its properties are the same for both start and end frames.
                # We do a deepcopy for the end frame to prevent any potential downstream mutation issues.
                all_start_meshes.extend(mesh_input["meshes"])
                all_start_materials.extend(mesh_input["materials"])
                all_start_transforms.extend(mesh_input["transforms"])
                
                all_end_meshes.extend(copy.deepcopy(mesh_input["meshes"]))
                all_end_materials.extend(copy.deepcopy(mesh_input["materials"]))
                all_end_transforms.extend(copy.deepcopy(mesh_input["transforms"]))

        # Package the final outputs
        start_frame_combined = {
            "meshes": all_start_meshes,
            "materials": all_start_materials,
            "transforms": all_start_transforms
        }
        
        end_frame_combined = {
            "meshes": all_end_meshes,
            "materials": all_end_materials,
            "transforms": all_end_transforms
        }

        # This is the final sequence object sent to the Scene Builder
        mesh_sequence = {
            "start_meshes": start_frame_combined,
            "end_meshes": end_frame_combined
        }

        return (start_frame_combined, mesh_sequence)


class RenderFormerMeshTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_mesh": ("MESH",),
                "end_mesh_pos_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "end_mesh_pos_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "end_mesh_pos_z": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "end_mesh_rot_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_mesh_rot_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_mesh_rot_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "end_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "end_diffuse_rgb": ("STRING", {"default": "204, 204, 204", "multiline": False}),
                "end_specular_rgb": ("STRING", {"default": "25, 25, 25", "multiline": False}),
                "end_roughness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "get_mesh_sequence"
    CATEGORY = "PHRenderFormer"

    def _parse_color_string(self, color_str, scale=1.0):
        if not isinstance(color_str, str): return None
        try:
            parts = [float(p.strip()) for p in color_str.split(',')]
            if len(parts) == 3: return [p / scale for p in parts]
        except (ValueError, AttributeError): pass
        return None

    def get_mesh_sequence(self, start_mesh, end_mesh_pos_x, end_mesh_pos_y, end_mesh_pos_z,
                          end_mesh_rot_x, end_mesh_rot_y, end_mesh_rot_z, end_scale,
                          end_diffuse_rgb, end_specular_rgb, end_roughness):
        
        if not start_mesh or "meshes" not in start_mesh or not start_mesh["meshes"]:
            raise Exception("RenderFormerMeshTarget: Invalid start_mesh input.")

        # --- Create Start and End Data Structures ---
        start_meshes_data = []
        end_meshes_data = []

        # The input `start_mesh` can contain multiple sub-meshes. We need to create a
        # start/end pair for each one.
        for i in range(len(start_mesh["meshes"])):
            # --- Start State ---
            # We create a new dictionary for each mesh to ensure they are independent.
            start_data = {
                "mesh": start_mesh["meshes"][i],
                "material": start_mesh["materials"][i],
                "transform": start_mesh["transforms"][i],
                "ph_uuid": str(uuid.uuid4()) # Add a unique ID for tracking in the combine node
            }
            start_meshes_data.append(start_data)

            # --- End State ---
            # Create a deep copy of the start state to modify for the end frame.
            end_data = copy.deepcopy(start_data)
            
            # Update transform for the end state
            end_data["transform"]["translation"] = [end_mesh_pos_x, end_mesh_pos_y, end_mesh_pos_z]
            end_data["transform"]["rotation"] = [end_mesh_rot_x, end_mesh_rot_y, end_mesh_rot_z]
            end_data["transform"]["scale"] = [end_scale, end_scale, end_scale]
            
            # Update material for the end state
            end_diffuse = self._parse_color_string(end_diffuse_rgb, 255.0)
            if end_diffuse:
                end_data["material"]["diffuse"] = end_diffuse

            end_specular = self._parse_color_string(end_specular_rgb, 255.0)
            if end_specular:
                end_data["material"]["specular"] = end_specular
            
            end_data["material"]["roughness"] = end_roughness
            
            end_meshes_data.append(end_data)

        # Convert the lists of dicts into the final MESH dictionary format
        start_frame_combined = {
            "meshes": [d["mesh"] for d in start_meshes_data],
            "materials": [d["material"] for d in start_meshes_data],
            "transforms": [d["transform"] for d in start_meshes_data]
        }
        end_frame_combined = {
            "meshes": [d["mesh"] for d in end_meshes_data],
            "materials": [d["material"] for d in end_meshes_data],
            "transforms": [d["transform"] for d in end_meshes_data]
        }

        # The output is a dictionary containing both start and end states,
        # with each state being a MESH-formatted dictionary.
        mesh_output = {
            "start_meshes": start_frame_combined,
            "end_meshes": end_frame_combined
        }
        
        return (mesh_output,)


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
        # Return based on what was processed
        if output_video is not None and output_video.shape[0] > 0:
            # If a video was generated, use its first frame as the single image preview
            if output_image is None:
                output_image = output_video[0].unsqueeze(0)
            return (output_image, output_video)
        elif output_image is not None:
            # If only a single image was generated, return it and a correctly shaped empty tensor for the video
            _, height, width, channels = output_image.shape
            return (output_image, torch.empty(0, height, width, channels))
        else:
            # If nothing was generated, return empty tensors for both
            return (torch.zeros(1, 1, 1, 3), torch.empty(0, 1, 1, 3))

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
    "RenderFormerLightingTarget": RenderFormerLightingTarget,
    "RenderFormerSceneBuilder": RenderFormerSceneBuilder,
    "RenderFormerGenerator": RenderFormerGenerator,
    "RenderFormerLoadMesh": RenderFormerLoadMesh,
    "RenderFormerRemeshMesh": RenderFormerRemeshMesh,
    "RenderFormerRandomizeColors": RenderFormerRandomizeColors,
    "RenderFormerExampleScene": RenderFormerExampleScene,
    "RenderFormerFromJSON": RenderFormerFromJSON,
    "RenderFormerMeshCombine": RenderFormerMeshCombine,
    "RenderFormerLightingCombine": RenderFormerLightingCombine,
    "RenderFormerMeshTarget": RenderFormerMeshTarget,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenderFormerModelLoader": "RenderFormer Model Loader",
    "RenderFormerCamera": "RenderFormer Camera",
    "RenderFormerCameraTarget": "RenderFormer Camera Target",
    "RenderFormerLighting": "RenderFormer Lighting",
    "RenderFormerLightingTarget": "RenderFormer Lighting Target",
    "RenderFormerSceneBuilder": "RenderFormer Scene Builder",
    "RenderFormerGenerator": "RenderFormer Sampler",
    "RenderFormerLoadMesh": "RenderFormer Mesh Loader",
    "RenderFormerRemeshMesh": "RenderFormer Remesh",
    "RenderFormerRandomizeColors": "RenderFormer Random Colors",
    "RenderFormerExampleScene": "RenderFormer Example Scene",
    "RenderFormerFromJSON": "RenderFormer From JSON",
    "RenderFormerMeshCombine": "RenderFormer Mesh Combine",
    "RenderFormerLightingCombine": "RenderFormer Lighting Combine",
    "RenderFormerMeshTarget": "RenderFormer Mesh Target",
}