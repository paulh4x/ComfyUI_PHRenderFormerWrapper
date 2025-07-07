import os
import folder_paths
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register the 3D models directory
input_3d_dir = os.path.join(folder_paths.get_input_directory(), "3d")
if not os.path.exists(input_3d_dir):
    os.makedirs(input_3d_dir)

folder_paths.add_model_folder_path("3d", input_3d_dir)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]