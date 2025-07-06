# ComfyUI-PHRenderFormerWrapper

**Author:** paulh4x

> [!WARNING]
> **WORK IN PROGRESS:** This project is currently under active development and should be considered experimental. It will be officially released as soon as it is stable.

This repository contains a set of custom nodes for ComfyUI that provide a wrapper for Microsoft's **RenderFormer** model. It allows you to render complex 3D scenes with physically-based materials and global illumination directly within the ComfyUI interface.

## Features

- **End-to-End Rendering:** Load 3D models, define materials, set up cameras, and render—all within ComfyUI.
- **Modular Node-Based Workflow:** Each step of the rendering pipeline is a separate node, allowing for flexible and complex setups.
- **Advanced Mesh Processing:** Includes nodes for loading, combining, remeshing, and applying simple color randomization to your 3D assets.
- **Lighting and Material Control:** Easily add default light sources and control PBR material properties like diffuse, specular, roughness, and emission.
- **Full Transformation Control:** Apply translation, rotation, and scaling to objects within the scene.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone <repository_url> ComfyUI-PHRenderFormerWrapper
    ```
3.  Install the required dependencies:
    ```bash
    cd ComfyUI-PHRenderFormerWrapper
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

## Nodes

-   **PHRenderFormer Model Loader**: Loads a specified RenderFormer model from Hugging Face or a local path.
-   **PHRenderFormer Mesh Loader**: Loads a 3D mesh file (e.g., `.obj`, `.glb`) from your `ComfyUI/input/3d` directory.
-   **PHRenderFormer Mesh Combine**: Combines multiple meshes into a single scene object.
-   **PHRenderFormer Remesh**: Simplifies the geometry of a mesh to a target face count using `pymeshlab`.
-   **PHRenderFormer Random Colors**: Applies random vertex colors to a mesh for creative effects.
-   **PHRenderFormer Camera**: Defines the camera's position, look-at target, and field of view (FOV).
-   **PHRenderFormer Material**: Defines the PBR material properties for a mesh.
-   **PHRenderFormer Lighting**: Controls the scene's lighting, allowing you to add and configure a default emissive light source.
-   **PHRenderFormer Scene Builder**: Assembles the final scene by combining meshes, a camera, materials, lighting, and transformations.
-   **PHRenderFormer Sampler**: Executes the RenderFormer pipeline on the assembled scene to produce the final rendered image.
-   **PHRenderFormer From JSON**: Loads a scene from a JSON definition, allowing for more complex and customized setups.

## Acknowledgements

This project would not be possible without the foundational work of others.

-   **RenderFormer**
    This project is a wrapper for the incredible **RenderFormer** model. All credit for the underlying rendering technology goes to the original authors: **Chong Zeng, Yue Dong, Pieter Peers, Hongzhi Wu, and Xin Tong**.
    -   [Official RenderFormer Project Page](https://microsoft.github.io/renderformer/)
    -   [Official RenderFormer GitHub](https://github.com/microsoft/renderformer)

-   **comfyui-hunyuan3dwrapper**
    Special thanks to **kijai** for their work on the `comfyui-hunyan3dwrapper`, which served as an invaluable reference and starting point for this project during its development in `vibecoding`.