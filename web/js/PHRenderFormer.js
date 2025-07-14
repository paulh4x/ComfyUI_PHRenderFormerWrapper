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
    return { h: h * 360, s: s, v: v };
}

function hsvToRgb(h, s, v) {
    let r, g, b;
    let i = Math.floor(h / 60);
    let f = h / 60 - i;
    let p = v * (1 - s);
    let q = v * (1 - f * s);
    let t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
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
            "RenderFormerCameraTarget",
            "RenderFormerLighting",
            "RenderFormerSceneBuilder",
            "RenderFormerVideoSceneBuilder",
            "RenderFormerGenerator",
            "RenderFormerVideoSampler",
            "RenderFormerVideoSamplerBatched",
            "RenderFormerLoadMesh",
            "RenderFormerRemeshMesh",
            "RenderFormerRandomizeColors",
            "RenderFormerExampleScene",
            "RenderFormerFromJSON",
            "RenderFormerMeshCombine",
            "RenderFormerLightingCombine",
            "RenderFormerLightingTarget",
            "RenderFormerMeshTarget",
        ];

        if (node_names.includes(node.comfyClass)) {
            const HEADER_COLOR = "#FDC501"; // yellow
            const BG_COLOR = "#111417"; // anthrazit
            node.color = HEADER_COLOR;
            node.bgcolor = BG_COLOR;
        }
        
        // -- PHRenderFormer Mesh Loader Node --
        if (node.comfyClass === "RenderFormerLoadMesh") {
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
            const HUE_SLIDER_HEIGHT = 20;
            const PREVIEW_BOX_WIDTH = 50;

            let mainPickerWidth, mainPickerHeight = PICKER_AREA_HEIGHT;
            let hueSliderWidth, hueSliderHeight = HUE_SLIDER_HEIGHT;
            
            let svPickerPos = { x: 0, y: 0 };
            let huePickerPos = 0;
            
            let isSVPickerActive = false;
            let isHuePickerActive = false;

            const colorWidget = node.widgets.find(w => w.name === "diffuse_rgb");
            if (!colorWidget) {
                console.error("RenderFormer Mesh Loader: Could not find 'diffuse_rgb' widget!");
                return;
            }

            let hsv = { h: 0, s: 1, v: 1 };

            function updateColorFromHSV() {
                const rgb = hsvToRgb(hsv.h, hsv.s, hsv.v);
                colorWidget.value = `${rgb.r}, ${rgb.g}, ${rgb.b}`;
                node.setDirtyCanvas(true, true);
            }

            function updateSV(x, y) {
                x = Math.max(0, Math.min(x, mainPickerWidth));
                y = Math.max(0, Math.min(y, mainPickerHeight));
                svPickerPos = { x, y };
                hsv.s = x / mainPickerWidth;
                hsv.v = 1 - (y / mainPickerHeight);
                updateColorFromHSV();
            }

            function updateHue(x) {
                x = Math.max(0, Math.min(x, hueSliderWidth));
                huePickerPos = x;
                hsv.h = (x / hueSliderWidth) * 360;
                updateColorFromHSV();
            }

            const pickerWidget = node.addCustomWidget({
                name: "COLOR_PICKER_DISPLAY",
                type: "CANVAS_WIDGET",
                y: colorWidget.y,
                draw: function (ctx, node, widgetWidth, widgetY, height) {
                    const totalWidth = widgetWidth - PADDING * 2;
                    mainPickerWidth = totalWidth - PREVIEW_BOX_WIDTH - WIDGET_MARGIN;
                    hueSliderWidth = mainPickerWidth;
                    const drawY = widgetY + PADDING;

                    ctx.save();
                    ctx.translate(PADDING, drawY);

                    // --- Draw Saturation/Value Box ---
                    const mainHueRgb = hsvToRgb(hsv.h, 1, 1);
                    ctx.fillStyle = `rgb(${mainHueRgb.r}, ${mainHueRgb.g}, ${mainHueRgb.b})`;
                    ctx.fillRect(PREVIEW_BOX_WIDTH + WIDGET_MARGIN, 0, mainPickerWidth, mainPickerHeight);

                    const svBoxX = PREVIEW_BOX_WIDTH + WIDGET_MARGIN;
                    const whiteGradient = ctx.createLinearGradient(svBoxX, 0, svBoxX + mainPickerWidth, 0);
                    whiteGradient.addColorStop(0, "rgba(255,255,255,1)");
                    whiteGradient.addColorStop(1, "rgba(255,255,255,0)");
                    ctx.fillStyle = whiteGradient;
                    ctx.fillRect(svBoxX, 0, mainPickerWidth, mainPickerHeight);

                    const blackGradient = ctx.createLinearGradient(svBoxX, 0, svBoxX, mainPickerHeight);
                    blackGradient.addColorStop(0, "rgba(0,0,0,0)");
                    blackGradient.addColorStop(1, "rgba(0,0,0,1)");
                    ctx.fillStyle = blackGradient;
                    ctx.fillRect(svBoxX, 0, mainPickerWidth, mainPickerHeight);

                    // --- Draw Hue Slider ---
                    const hueSliderY = mainPickerHeight + WIDGET_MARGIN;
                    const hueGradient = ctx.createLinearGradient(svBoxX, 0, svBoxX + hueSliderWidth, 0);
                    hueGradient.addColorStop(0, "rgb(255, 0, 0)");
                    hueGradient.addColorStop(0.17, "rgb(255, 255, 0)");
                    hueGradient.addColorStop(0.33, "rgb(0, 255, 0)");
                    hueGradient.addColorStop(0.5, "rgb(0, 255, 255)");
                    hueGradient.addColorStop(0.67, "rgb(0, 0, 255)");
                    hueGradient.addColorStop(0.83, "rgb(255, 0, 255)");
                    hueGradient.addColorStop(1, "rgb(255, 0, 0)");
                    ctx.fillStyle = hueGradient;
                    ctx.fillRect(svBoxX, hueSliderY, hueSliderWidth, hueSliderHeight);

                    // --- Draw Pickers ---
                    const svX = svPickerPos.x + PREVIEW_BOX_WIDTH + WIDGET_MARGIN;
                    const svY = svPickerPos.y;
                    ctx.beginPath();
                    ctx.arc(svX, svY, 5, 0, 2 * Math.PI);
                    ctx.strokeStyle = "white";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.arc(svX, svY, 7, 0, 2 * Math.PI);
                    ctx.strokeStyle = "black";
                    ctx.stroke();

                    const hueX = huePickerPos + PREVIEW_BOX_WIDTH + WIDGET_MARGIN;
                    ctx.fillStyle = "white";
                    ctx.fillRect(hueX - 2, hueSliderY, 4, hueSliderHeight);
                    ctx.strokeStyle = "black";
                    ctx.strokeRect(hueX - 2, hueSliderY, 4, hueSliderHeight);

                    // --- Draw Preview Box ---
                    const finalRgb = hsvToRgb(hsv.h, hsv.s, hsv.v);
                    ctx.fillStyle = `rgb(${finalRgb.r}, ${finalRgb.g}, ${finalRgb.b})`;
                    ctx.fillRect(0, 0, PREVIEW_BOX_WIDTH, mainPickerHeight + WIDGET_MARGIN + hueSliderHeight);
                    
                    ctx.restore();
                },
                mouse: function (event, pos, node) {
                    const widgetRect = this.computeArea();
                    const clickX = pos[0] - PADDING;
                    const clickY = pos[1] - (widgetRect.y + PADDING);

                    const svBox = { x: PREVIEW_BOX_WIDTH + WIDGET_MARGIN, y: 0, w: mainPickerWidth, h: mainPickerHeight };
                    const hueBox = { x: PREVIEW_BOX_WIDTH + WIDGET_MARGIN, y: mainPickerHeight + WIDGET_MARGIN, w: hueSliderWidth, h: hueSliderHeight };

                    if (event.type === "pointerdown") {
                        if (clickX > svBox.x && clickX < svBox.x + svBox.w && clickY > svBox.y && clickY < svBox.y + svBox.h) {
                            isSVPickerActive = true;
                            updateSV(clickX - svBox.x, clickY - svBox.y);
                            return true;
                        }
                        if (clickX > hueBox.x && clickX < hueBox.x + hueBox.w && clickY > hueBox.y && clickY < hueBox.y + hueBox.h) {
                            isHuePickerActive = true;
                            updateHue(clickX - hueBox.x);
                            return true;
                        }
                    } else if (event.type === "pointermove") {
                        if (isSVPickerActive) {
                            updateSV(clickX - svBox.x, clickY - svBox.y);
                            return true;
                        }
                        if (isHuePickerActive) {
                            updateHue(clickX - hueBox.x);
                            return true;
                        }
                    } else if (event.type === "pointerup") {
                        isSVPickerActive = false;
                        isHuePickerActive = false;
                    }
                    return false;
                },
                computeSize: function (width) {
                    const totalHeight = PADDING * 2 + mainPickerHeight + WIDGET_MARGIN + hueSliderHeight;
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
                hsv = rgbToHsv(initialRgb.r, initialRgb.g, initialRgb.b);
                
                node.setSize(node.computeSize());
                const totalWidth = node.size[0] - PADDING * 2;
                mainPickerWidth = totalWidth - PREVIEW_BOX_WIDTH - WIDGET_MARGIN;
                hueSliderWidth = mainPickerWidth;

                svPickerPos.x = hsv.s * mainPickerWidth;
                svPickerPos.y = (1 - hsv.v) * mainPickerHeight;
                huePickerPos = (hsv.h / 360) * hueSliderWidth;
                
                updateColorFromHSV();
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