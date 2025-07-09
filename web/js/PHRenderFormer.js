import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// --- Helper Functions ---

async function uploadFile(file) {
    try {
        const body = new FormData();
        body.append("image", file);
        body.append("subfolder", "3d");
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });
        return resp.status === 200 ? resp : alert(resp.status + " - " + resp.statusText);
    } catch (error) {
        alert(error);
    }
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) } : null;
}

function rgbToHsv(r, g, b) {
    r /= 255, g /= 255, b /= 255;
    let max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, v = max;
    let d = max - min;
    s = max == 0 ? 0 : d / max;
    if (max == min) {
        h = 0;
    } else {
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return { h, s, v };
}

function rgb_to_hex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
}

function rgbStringToRgb(rgbStr) {
    if (!rgbStr || typeof rgbStr !== 'string') return { r: 204, g: 204, b: 204 };
    const parts = rgbStr.split(',').map(s => parseInt(s.trim(), 10));
    if (parts.length !== 3 || parts.some(isNaN)) return { r: 204, g: 204, b: 204 };
    return { r: parts[0], g: parts[1], b: parts[2] };
}

app.registerExtension({
    name: "Comfy.RenderFormer.Nodes",
    async nodeCreated(node) {
        const node_names = [
            "RenderFormerModelLoader",
            "RenderFormerCamera",
            "PHRenderFormerCameraTarget",
            "RenderFormerLighting",
            "RenderFormerSceneBuilder",
            "RenderFormerVideoSceneBuilder",
            "RenderFormerGenerator",
            "PHRenderFormerVideoSampler",
            "LoadMesh",
            "RemeshMesh",
            "RandomizeColors",
            "LoadRenderFormerExampleScene",
            "RenderFormerFromJSON",
            "RenderFormerMeshCombine",
            "RenderFormerLightingCombine",
        ];

        if (node_names.includes(node.comfyClass)) {
            const HEADER_COLOR = "#FDC501"; // yellow
            const BG_COLOR = "#111417"; // anthrazit
            node.color = HEADER_COLOR;
            node.bgcolor = BG_COLOR;
        }
        
        // -- PHRenderFormer Mesh Loader Node --
        if (node.comfyClass === "LoadMesh") {
            // 1. Add File Upload Widget
            const pathWidget = node.widgets.find((w) => w.name === "mesh");
            const fileInput = document.createElement("input");
            
            node.onRemoved = () => {
                fileInput?.remove();
            };
        	
            Object.assign(fileInput, {
                type: "file",
                accept: ".obj,.glb,.gltf,.stl,.3mf,.ply",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        let resp = await uploadFile(fileInput.files[0]);
                        if (resp?.status !== 200) return;
                        
                        const filename = (await resp.json()).name;
                        if (!pathWidget.options.values.includes(filename)) {
                            pathWidget.options.values.push(filename);
                        }
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename);
                        }
                    }
                },
            });
            
            document.body.append(fileInput);
            let uploadWidget = node.addWidget("button", "choose mesh to upload", "image", () => {
                fileInput.click();
            });
            uploadWidget.options.serialize = false;

            // 2. Add Advanced Color Picker
            const PADDING = 10;
            const WIDGET_HEIGHT = 20;
            const WIDGET_MARGIN = 5;
            const PICKER_AREA_HEIGHT = 150;

            let gradientWidth, gradientHeight = PICKER_AREA_HEIGHT;
            let pickerPos = { x: 0, y: 0 };
            let isPickerActive = false;

            const colorWidget = node.widgets.find(w => w.name === "diffuse_rgb");
            if (!colorWidget) {
                console.error("PHRenderFormer Mesh Loader: Could not find 'diffuse_rgb' widget!");
                return;
            }

            let selectedColor = "#CCCCCC";

            function calculatePickerPosFromHex(hex) {
                const rgb = hexToRgb(hex);
                if (!rgb) return { x: 0, y: 0 };
                const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
                const x = hsv.h * (gradientWidth || 200);
                const y = (1 - hsv.v) * gradientHeight;
                return {
                    x: Math.max(0, Math.min(x, gradientWidth || 200)),
                    y: Math.max(0, Math.min(y, gradientHeight))
                };
            }

            function updateColorOutput() {
                const rgb = hexToRgb(selectedColor);
                if (rgb) {
                    colorWidget.value = `${rgb.r}, ${rgb.g}, ${rgb.b}`;
                }
                node.setDirtyCanvas(true, true);
            }

            function updateSelectedColor(x, y) {
                x = Math.max(0, Math.min(x, gradientWidth || 0));
                y = Math.max(0, Math.min(y, gradientHeight || 0));
                pickerPos = { x, y };

                if (gradientWidth > 0 && gradientHeight > 0) {
                    const offscreenCanvas = document.createElement('canvas');
                    offscreenCanvas.width = gradientWidth;
                    offscreenCanvas.height = gradientHeight;
                    const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

                    const gradient = offscreenCtx.createLinearGradient(0, 0, gradientWidth, 0);
                    gradient.addColorStop(0, "rgb(255, 0, 0)");
                    gradient.addColorStop(0.17, "rgb(255, 255, 0)");
                    gradient.addColorStop(0.33, "rgb(0, 255, 0)");
                    gradient.addColorStop(0.5, "rgb(0, 255, 255)");
                    gradient.addColorStop(0.67, "rgb(0, 0, 255)");
                    gradient.addColorStop(0.83, "rgb(255, 0, 255)");
                    gradient.addColorStop(1, "rgb(255, 0, 0)");
                    offscreenCtx.fillStyle = gradient;
                    offscreenCtx.fillRect(0, 0, gradientWidth, gradientHeight);

                    const blackGradient = offscreenCtx.createLinearGradient(0, 0, 0, gradientHeight);
                    blackGradient.addColorStop(0, "rgba(0, 0, 0, 0)");
                    blackGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
                    offscreenCtx.fillStyle = blackGradient;
                    offscreenCtx.fillRect(0, 0, gradientWidth, gradientHeight);

                    const pixelData = offscreenCtx.getImageData(Math.round(x), Math.round(y), 1, 1).data;
                    selectedColor = rgb_to_hex(pixelData[0], pixelData[1], pixelData[2]);
                }
                updateColorOutput();
            }

            const pickerWidget = node.addCustomWidget({
                name: "COLOR_PICKER_DISPLAY",
                type: "CANVAS_WIDGET",
                y: colorWidget.y,
                draw: function (ctx, node, widgetWidth, widgetY, height) {
                    gradientWidth = widgetWidth - PADDING * 2;
                    const drawY = widgetY + PADDING;

                    ctx.save();
                    ctx.translate(PADDING, drawY);

                    if (gradientWidth > 0 && gradientHeight > 0) {
                        const gradient = ctx.createLinearGradient(0, 0, gradientWidth, 0);
                        gradient.addColorStop(0, "rgb(255, 0, 0)");
                        gradient.addColorStop(0.17, "rgb(255, 255, 0)");
                        gradient.addColorStop(0.33, "rgb(0, 255, 0)");
                        gradient.addColorStop(0.5, "rgb(0, 255, 255)");
                        gradient.addColorStop(0.67, "rgb(0, 0, 255)");
                        gradient.addColorStop(0.83, "rgb(255, 0, 255)");
                        gradient.addColorStop(1, "rgb(255, 0, 0)");
                        ctx.fillStyle = gradient;
                        ctx.fillRect(0, 0, gradientWidth, gradientHeight);

                        const blackGradient = ctx.createLinearGradient(0, 0, 0, gradientHeight);
                        blackGradient.addColorStop(0, "rgba(0, 0, 0, 0)");
                        blackGradient.addColorStop(1, "rgba(0, 0, 0, 1)");
                        ctx.fillStyle = blackGradient;
                        ctx.fillRect(0, 0, gradientWidth, gradientHeight);

                        const pickerDrawX = Math.max(0, Math.min(pickerPos.x, gradientWidth));
                        const pickerDrawY = Math.max(0, Math.min(pickerPos.y, gradientHeight));
                        ctx.beginPath();
                        ctx.arc(pickerDrawX, pickerDrawY, 5, 0, Math.PI * 2);
                        ctx.strokeStyle = isPickerActive ? "yellow" : "white";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    }

                    ctx.fillStyle = selectedColor;
                    ctx.fillRect(0, gradientHeight + WIDGET_MARGIN, gradientWidth, WIDGET_HEIGHT);

                    ctx.fillStyle = "white";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "left";
                    ctx.fillText(selectedColor, 5, gradientHeight + WIDGET_MARGIN + WIDGET_HEIGHT - 5);

                    ctx.restore();
                },
                mouse: function (event, pos, node) {
                    const widgetRect = this.computeArea();
                    const gradientAreaX = PADDING;
                    const gradientAreaY = widgetRect.y + PADDING;

                    const clickX = pos[0] - gradientAreaX;
                    const clickY = pos[1] - gradientAreaY;

                    if (event.type === "pointerdown") {
                        if (clickX >= 0 && clickX <= gradientWidth && clickY >= 0 && clickY <= gradientHeight) {
                            isPickerActive = true;
                            updateSelectedColor(clickX, clickY);
                            event.stopPropagation();
                            return true;
                        }
                    } else if (event.type === "pointermove" && isPickerActive) {
                        updateSelectedColor(clickX, clickY);
                        return true;
                    } else if (event.type === "pointerup" && isPickerActive) {
                        isPickerActive = false;
                        return true;
                    }
                    return false;
                },
                computeSize: function (width) {
                    const totalHeight = PADDING * 2 + gradientHeight + WIDGET_MARGIN + WIDGET_HEIGHT;
                    return [width, totalHeight];
                },
                computeArea: function() {
                    let y = this.y || 0;
                    let totalHeight = this.computeSize(node.size[0])[1];
                    return { x: 0, y: y, w: node.size[0], h: totalHeight };
                }
            });

            colorWidget.type = "HIDDEN";

            function initializePicker() {
                const initialRgb = rgbStringToRgb(colorWidget.value);
                selectedColor = rgb_to_hex(initialRgb.r, initialRgb.g, initialRgb.b);
                
                node.setSize(node.computeSize());
                gradientWidth = node.size[0] - PADDING * 2;
                pickerPos = calculatePickerPosFromHex(selectedColor);
                updateColorOutput();
            }
            
            setTimeout(() => initializePicker(), 0);

            const originalSetValue = colorWidget.setValue;
            colorWidget.setValue = function(value) {
                if (originalSetValue) {
                    originalSetValue.call(this, value);
                }
                initializePicker();
            };
        }

        // -- PHRenderFormer Mesh Combine Node --
        // No custom JS is needed. The default ComfyUI behavior for optional
        // inputs handles this node correctly, just like the lighting combine node.
    },
});