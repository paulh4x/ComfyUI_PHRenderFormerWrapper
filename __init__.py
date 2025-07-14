import os
import folder_paths

# ANSI escape codes for colors
dark_yellow = "\033[33m"
dark_grey = "\033[90m"
green = "\033[92m"
red = "\033[91m"
reset = "\033[0m"

# ASCII Art Logo
logo = fr"""
{dark_yellow}[ -->> Initializing PH's ComfyUI Custom Nodes <<-- ]{reset}

{dark_yellow}   _ \ |  |    _ \                  |            __|{reset}
{dark_yellow}   __/ __ |      /   -_)    \    _` |   -_)   _| _|  _ \   _| ` \    -_)   _|{reset}
{dark_yellow}  _|  _| _|   _|_\ \___| _| _| \__,_| \___| _|  _| \___/ _| _|_|_| \___| _|{reset}

{dark_grey}      -->> PH's RenderFormer Wrapper for ComfyUI Version: 0.3.15 <<--{reset}
"""

print(logo)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"{dark_yellow}[ -->> PH RenderFormer Nodes loaded {green}successfully{dark_yellow}! <<-- ]{reset}")
except ImportError as e:
    print(f"{dark_yellow}[ -->> PH RenderFormer Nodes {red}loading ERROR{dark_yellow}, please make sure requirements are installed <<-- ]{reset}")
    print(f"Error: {e}")


# Register the 3D models directory
input_3d_dir = os.path.join(folder_paths.get_input_directory(), "3d")
if not os.path.exists(input_3d_dir):
    os.makedirs(input_3d_dir)

folder_paths.add_model_folder_path("3d", input_3d_dir)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]