import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

import MaterialService from './materialService.js';
import { Voxelizer } from './voxelizer.js';

class CastFlowApp {
    constructor() {
        this.initDOM();
        this.initThree();
        this.initEngine(); // Async: tries GPU first, falls back to CPU Worker
        this.loadMaterials();
        
        this.simulationData = null;
        this.viewMode = 'fill';
        this.moldMesh = null;
        this.fluidMesh = null;
        this.materialsMap = {};
        this.currentMatLine = null;
        this.fluidCount = 0;
        this.autoDt = true; // Auto dt mode
        this.useGPU = false;
        this.gpuEngine = null;
        this.gpuRunning = false;
        
        // Surface Injection System
        this.surfaceModeActive = false;
        this.injectionSurfaces = []; 
        this.selectedFaces = new Set();
        this.isPainting = false;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.highlightMesh = null;
        this.surfaceArrows = new THREE.Group();
        this.scene.add(this.surfaceArrows);
        this.faceAdjacency = null; 
        
        this.dummy = new THREE.Object3D();
        this.colorBuffer = new THREE.Color();

        this.initTheme();
        this.bindEvents();
        this.animate();
        this.hideLoader();
    }

    initDOM() {
        this.loader = document.getElementById('loader-overlay');
        this.container = document.getElementById('viewport');
        
        this.btnSimulate = document.getElementById('btn-simulate');
        this.btnReset = document.getElementById('btn-reset');
        this.btnPlayPause = document.getElementById('btn-play-pause');
        this.btnStep = document.getElementById('btn-step');
        this.btnToggleMold = document.getElementById('btn-toggle-mold');
        this.btnToggleVoxels = document.getElementById('btn-toggle-voxels');
        this.btnRecenter = document.getElementById('btn-recenter');
        this.btnToggleGrid = document.getElementById('btn-toggle-grid');
        this.btnTheme = document.getElementById('btn-theme');
        
        this.inputGeometry = document.getElementById('geometry-upload');
        this.selectMat = document.getElementById('material-select');
        this.inputVxSize = document.getElementById('voxel-size');
        this.inputPadding = document.getElementById('domain-padding');
        this.inputDt = document.getElementById('time-step');
        this.dtBadge = document.getElementById('dt-badge');
        
        this.visButtons = document.querySelectorAll('.mode-btn');
        this.playbackUI = document.getElementById('playback-controls');
        this.timeDisplay = document.getElementById('time-display');
        this.timeSlider = document.getElementById('time-slider');
        
        // Debug
        this.btnWireframe = document.getElementById('btn-wireframe');
        this.sliceSlider = document.getElementById('slice-slider');
        
        // Units
        this.unitScaleSelect = document.getElementById('geo-units');
        this.geoBoundsDisplay = document.getElementById('geo-bounds');
        this.unitScale = 1.0;

        // Legend
        this.legendOverlay = document.getElementById('legend-overlay');
        this.legendCanvas = document.getElementById('legend-canvas');
        this.legendTitle = document.getElementById('legend-title');
        this.legendMax = document.getElementById('legend-max');
        this.legendMin = document.getElementById('legend-min');

        // Frames scrubbing
        this.frames = [];
        this.playing = false;

        // Surface Editor UI
        this.btnSurfaceMode = document.getElementById('btn-surface-mode');
        this.surfaceTools = document.getElementById('surface-tools');
        this.selectToolMode = document.getElementById('select-mode');
        this.brushSizeRow = document.getElementById('brush-size-row');
        this.angleLimitRow = document.getElementById('angle-limit-row');
        this.btnSurfaceClear = document.getElementById('btn-surface-clear');
        this.btnSurfaceAdd = document.getElementById('btn-surface-add');
        this.surfaceList = document.getElementById('surface-list');
    }

    // ─── Theme System ──────────────────────────────────────────
    initTheme() {
        const saved = localStorage.getItem('castflow_theme');
        if (saved === 'light') {
            document.body.dataset.theme = 'light';
            this.scene.background = new THREE.Color(0xe2e8f0);
        }
        this.updateThemeIcon();
    }

    toggleTheme() {
        const isLight = document.body.dataset.theme === 'light';
        if (isLight) {
            delete document.body.dataset.theme;
            this.scene.background = new THREE.Color(0x0f172a);
            localStorage.setItem('castflow_theme', 'dark');
        } else {
            document.body.dataset.theme = 'light';
            this.scene.background = new THREE.Color(0xe2e8f0);
            localStorage.setItem('castflow_theme', 'light');
        }
        this.updateThemeIcon();
        // Update grid colors
        if (this.gridHelper) {
            const isNowLight = document.body.dataset.theme === 'light';
            this.gridHelper.material[0].color.set(isNowLight ? 0x2563eb : 0x3b82f6);
            this.gridHelper.material[1].color.set(isNowLight ? 0x94a3b8 : 0x475569);
        }
    }

    updateThemeIcon() {
        const isLight = document.body.dataset.theme === 'light';
        this.btnTheme.innerHTML = `<i data-feather="${isLight ? 'sun' : 'moon'}" style="width: 16px; height: 16px;"></i>`;
        feather.replace();
    }

    // ─── Three.js ──────────────────────────────────────────────
    initThree() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a);
        
        this.camera = new THREE.PerspectiveCamera(45, this.container.clientWidth / this.container.clientHeight, 0.01, 500);
        this.camera.position.set(0, 5, 10);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Cap for perf
        this.container.appendChild(this.renderer.domElement);
        
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 0.001;
        this.controls.maxDistance = 500;
        
        const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
        dirLight.position.set(10, 20, 10);
        this.scene.add(dirLight);

        this.gridHelper = new THREE.GridHelper(10, 20, 0x3b82f6, 0x475569);
        this.gridHelper.position.y = -0.001;
        this.scene.add(this.gridHelper);
        
        // Setup renderer clipping
        this.renderer.localClippingEnabled = true;
        this.clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 9999);

        window.addEventListener('resize', this.onWindowResize.bind(this), false);

        // Raycasting events setup on canvas
        this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown.bind(this));
        this.renderer.domElement.addEventListener('pointermove', this.onPointerMove.bind(this));
        window.addEventListener('pointerup', this.onPointerUp.bind(this)); // Catch up outside canvas
    }

    async initEngine() {
        try {
            const { GPUSimEngine } = await import('./gpuEngine.js');
            if (await GPUSimEngine.isSupported()) {
                this.gpuEngine = await GPUSimEngine.create();
                this.useGPU = true;
                console.log('CastFlow: Using WebGPU compute engine');
                this.updateEngineBadge();
                return;
            }
        } catch(e) {
            console.warn('WebGPU not available, using CPU worker:', e);
        }
        this.initWorker();
        this.updateEngineBadge();
    }

    updateEngineBadge() {
        let badge = document.getElementById('engine-badge');
        if (!badge) {
            badge = document.createElement('span');
            badge.id = 'engine-badge';
            badge.className = 'engine-badge';
            document.querySelector('.sidebar-header h1').appendChild(badge);
        }
        badge.className = `engine-badge ${this.useGPU ? 'gpu' : 'cpu'}`;
        badge.innerText = this.useGPU ? 'GPU' : 'CPU';
    }

    initWorker() {
        this.worker = new Worker('./worker.js');
        this.worker.onmessage = (e) => {
            const data = e.data;
            if (data.type === 'INIT_DONE') {
                this.btnSimulate.disabled = false;
                this.btnReset.disabled = false;
                this.btnPlayPause.innerHTML = '<i data-feather="pause"></i>';
                feather.replace();
                this.playbackUI.style.display = 'flex';
                this.legendOverlay.style.display = 'block';
                this.drawLegend();
                this.frames = [];
                this.timeSlider.value = 0;
                this.timeSlider.max = 0;
                this.playing = true;
                this.worker.postMessage({ type: 'START' });
            } else if (data.type === 'RENDER_DATA') {
                this.onSimData(data.buffer, data.time);
            }
        };
    }

    async loadMaterials() {
        const mats = await MaterialService.loadMaterials();
        this.selectMat.innerHTML = '';
        mats.forEach(m => {
            this.materialsMap[m.name] = m;
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.innerText = m.name;
            this.selectMat.appendChild(opt);
        });
        
        this.updateMaterialUI();
    }

    updateMaterialUI() {
        const matName = this.selectMat.value;
        const m = this.materialsMap[matName];
        if(!m) return;
        this.currentMatLine = m;
        document.getElementById('inject-temp').value = m.liquidusTemp + 100;
        document.getElementById('mold-temp').value = m.solidusTemp > 500 ? 250 : 100;
        // Recalculate auto dt
        this.computeAutoDt();
    }

    // ─── Auto dt Computation ──────────────────────────────────
    computeAutoDt() {
        if (!this.autoDt) return;
        if (!this.simulationData || !this.currentMatLine) return;

        const mat = this.currentMatLine;
        const dx = this.simulationData.voxelSize;
        const P = parseFloat(document.getElementById('inject-pressure').value) || 100000;

        // CFL condition: dt ≤ CFL * dx / v_char
        // Characteristic velocity from Bernoulli: v ≈ sqrt(2P / ρ)
        const vChar = Math.sqrt(2.0 * P / mat.density);
        const cfl = 0.2; // Conservative CFL number for explicit schemes
        const dt_cfl = cfl * dx / Math.max(vChar, 0.01);

        // Diffusion stability: dt ≤ dx² / (6 * α)  (3D explicit)
        const alpha = mat.thermalConductivity / (mat.density * mat.specificHeat);
        const dt_diff = (dx * dx) / (6.0 * Math.max(alpha, 1e-10));

        // Gravity stability: dt ≤ sqrt(dx / g)
        const dt_grav = Math.sqrt(dx / 9.81);

        // Take the most restrictive
        let dt = Math.min(dt_cfl, dt_diff, dt_grav);

        // Clamp to reasonable range
        dt = Math.max(dt, 1e-6); // Minimum practical dt
        dt = Math.min(dt, 0.1);  // Maximum practical dt

        // Round to nice value
        const magnitude = Math.pow(10, Math.floor(Math.log10(dt)));
        dt = Math.round(dt / magnitude) * magnitude;
        if (dt < 1e-6) dt = 1e-6;

        this.inputDt.value = dt.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
    }

    // ─── Events ──────────────────────────────────────────────
    bindEvents() {
        this.selectMat.addEventListener('change', () => this.updateMaterialUI());
        
        this.inputGeometry.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.loadGeometry(e.target.files[0]);
            }
        });

        const doRevoxelize = async () => {
            if (this.moldMesh) {
                this.showLoader(`Applying BBox / Resolution...`);
                await new Promise(r => setTimeout(r, 50));
                
                const vSizeUI = parseFloat(this.inputVxSize.value);
                const paddingV = parseFloat(this.inputPadding.value) || 0;
                const physicsVoxelSize = vSizeUI * this.unitScale;
                
                this.simulationData = await Voxelizer.voxelize(this.moldMesh, physicsVoxelSize, paddingV, (prog) => {
                    const pct = Math.floor(prog * 100);
                    document.getElementById('loader-text').innerText = `Voxelizing BBox ${pct}%`;
                });

                this.fluidCount = this.simulationData.totalCells;
                if (this.fluidMesh) {
                    this.scene.remove(this.fluidMesh);
                    this.fluidMesh = null;
                }
                // DO NOT create preview InstancedMesh here — it will be created
                // properly with initialized matrices in initSimulation().
                // Previously, creating it here with uninitialized matrices caused
                // all instances to render at origin (0,0,0).

                this.selectedFaces.clear();
                this.injectionSurfaces = [];
                
                while (this.surfaceArrows.children.length > 0) {
                    this.surfaceArrows.remove(this.surfaceArrows.children[0]);
                }
                
                this.renderSurfaceList();
                this.updateHighlightMesh();

                // Recompute auto dt for new voxel size
                this.computeAutoDt();

                this.hideLoader();
            }
        };

        this.inputVxSize.addEventListener('change', doRevoxelize);
        this.inputPadding.addEventListener('change', doRevoxelize);

        this.btnRecenter.addEventListener('click', () => this.resetCamera());

        this.btnSimulate.addEventListener('click', () => {
            if (!this.simulationData) return;
            this.initSimulation();
        });

        this.btnReset.addEventListener('click', () => {
             if (this.useGPU) {
                 this.gpuRunning = false;
             } else {
                 this.worker.postMessage({ type: 'PAUSE' });
             }
             if (this.fluidMesh) {
                 this.scene.remove(this.fluidMesh);
                 this.fluidMesh = null;
             }
             this.timeDisplay.innerText = "0.0s";
             this.btnSimulate.disabled = false;
             this.playbackUI.style.display = 'none';
             this.legendOverlay.style.display = 'none';
             this.frames = [];
             this.playing = false;
        });

        this.btnToggleMold.addEventListener('click', () => {
            if (this.moldMesh) {
                // Add Gimbal/Axes to world origin mapped relative to their Bounding Box to visualize coordinates
                if(!this.axesHelper) {
                    const axisSize = (this.meshBaseHeight && isFinite(this.meshBaseHeight)) ? this.meshBaseHeight * 1.5 : 1.0;
                    this.axesHelper = new THREE.AxesHelper(axisSize);
                    this.scene.add(this.axesHelper);
                }
                
                this.moldMesh.visible = !this.moldMesh.visible;
                this.btnToggleMold.innerHTML = this.moldMesh.visible ? 
                    '<i data-feather="eye-off" style="width: 14px; height: 14px; vertical-align: middle;"></i> Toggle Mold' : 
                    '<i data-feather="eye" style="width: 14px; height: 14px; vertical-align: middle;"></i> Toggle Mold';
                feather.replace();
            }
        });

        this.btnToggleVoxels.addEventListener('click', () => {
            if (this.fluidMesh) {
                this.fluidMesh.visible = !this.fluidMesh.visible;
                this.btnToggleVoxels.innerHTML = this.fluidMesh.visible ? 
                    '<i data-feather="eye-off" style="width: 14px; height: 14px; vertical-align: middle;"></i> Toggle Voxels' : 
                    '<i data-feather="eye" style="width: 14px; height: 14px; vertical-align: middle;"></i> Toggle Voxels';
                feather.replace();
            }
        });

        // Grid Toggle
        this.btnToggleGrid.addEventListener('click', () => {
            if (this.gridHelper) {
                this.gridHelper.visible = !this.gridHelper.visible;
            }
        });

        // Theme Toggle
        this.btnTheme.addEventListener('click', () => this.toggleTheme());

        this.visButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.visButtons.forEach(b => b.classList.remove('active'));
                const t = e.target.closest('.mode-btn');
                t.classList.add('active');
                this.viewMode = t.getAttribute('data-mode');
                this.drawLegend();
                // Refresh visualization if frozen
                if (!this.playing && this.frames.length > 0) {
                    this.updateFluidMesh(this.frames[parseInt(this.timeSlider.value)]);
                }
            });
        });
        
        this.btnPlayPause.addEventListener('click', () => {
            this.playing = !this.playing;
            if (this.useGPU) {
                this.gpuRunning = this.playing;
                if (this.playing) this.gpuSimLoop();
            } else {
                this.worker.postMessage({ type: this.playing ? 'START' : 'PAUSE' });
            }
            this.btnPlayPause.innerHTML = this.playing ? '<i data-feather="pause"></i>' : '<i data-feather="play"></i>';
            feather.replace();
        });

        this.btnStep.addEventListener('click', () => {
            if (!this.playing) {
                if (this.useGPU) {
                    this.gpuEngine.step(1);
                    this.gpuEngine.readVisualData().then(({ buffer, time }) => this.onSimData(buffer, time));
                } else {
                    this.worker.postMessage({ type: 'STEP' });
                }
            }
        });
        
        this.timeSlider.addEventListener('input', (e) => {
            if (this.frames.length === 0) return;
            const idx = parseInt(e.target.value);
            const buf = this.frames[idx];
            if (buf) this.updateFluidMesh(buf);
        });

        this.btnWireframe.addEventListener('click', () => {
            if (this.moldMesh) {
                this.moldMesh.material.wireframe = !this.moldMesh.material.wireframe;
            }
        });

        this.sliceSlider.addEventListener('input', (e) => {
            const pct = parseFloat(e.target.value) / 100.0;
            if (this.simulationData) {
                const bbox = this.simulationData.bboxMin;
                const height = this.simulationData.ny * this.simulationData.voxelSize;
                this.clipPlane.constant = bbox.y + height * pct;
            } else if (this.meshBaseHeight) {
                this.clipPlane.constant = this.meshBboxMin.y + this.meshBaseHeight * pct;
            }
        });

        // Surface Editor Bindings
        this.btnSurfaceMode.addEventListener('click', () => {
            if (!this.moldMesh) return; // Guard: no geometry loaded yet
            this.surfaceModeActive = !this.surfaceModeActive;
            if(this.surfaceModeActive) {
                this.btnSurfaceMode.classList.remove('btn-secondary');
                this.btnSurfaceMode.classList.add('btn-primary');
                this.surfaceTools.style.display = 'block';
                this.moldMesh.material.opacity = 0.8; 
                this.updateSurfaceUIParams();
            } else {
                this.btnSurfaceMode.classList.remove('btn-primary');
                this.btnSurfaceMode.classList.add('btn-secondary');
                this.surfaceTools.style.display = 'none';
                this.moldMesh.material.opacity = 0.3; // Return to glass view
                this.controls.enabled = true;
                this.isPainting = false;
            }
        });

        this.selectToolMode.addEventListener('change', () => this.updateSurfaceUIParams());
        this.btnSurfaceClear.addEventListener('click', () => {
            this.selectedFaces.clear();
            this.updateHighlightMesh();
        });
        this.btnSurfaceAdd.addEventListener('click', () => this.saveInjectionSurface());
        
        this.unitScaleSelect.addEventListener('change', (e) => {
            this.unitScale = parseFloat(e.target.value);
        });

        // dt Auto/Manual toggle
        this.dtBadge.addEventListener('click', () => {
            this.autoDt = !this.autoDt;
            this.dtBadge.innerText = this.autoDt ? 'AUTO' : 'MANUAL';
            this.dtBadge.classList.toggle('manual', !this.autoDt);
            this.inputDt.readOnly = this.autoDt;
            if (this.autoDt) {
                this.computeAutoDt();
            }
        });

        // Recompute auto dt when pressure or material properties change
        document.getElementById('inject-pressure').addEventListener('change', () => this.computeAutoDt());

        // Export button — download simulation frames as JSON
        document.getElementById('btn-export').addEventListener('click', () => this.exportData());
    }

    updateSurfaceUIParams() {
        const t = this.selectToolMode.value;
        this.brushSizeRow.style.display = (t === 'brush') ? 'flex' : 'none';
        this.angleLimitRow.style.display = (t === 'angle') ? 'flex' : 'none';
    }

    async loadGeometry(file) {
        this.showLoader("Loading Mesh...");
        try {
            const url = URL.createObjectURL(file);
            const ext = file.name.split('.').pop().toLowerCase();
            let geometry;

            if (ext === 'stl') {
                const loader = new STLLoader();
                geometry = await loader.loadAsync(url);
            } else if (ext === 'obj') {
                const loader = new OBJLoader();
                const obj = await loader.loadAsync(url);
                obj.traverse((child) => {
                    if (child.isMesh && !geometry) geometry = child.geometry;
                });
            }

            if (!geometry) throw new Error("Could not extract geometry.");
            
            // Merge vertices to create index for adjacency
            this.showLoader("Optimizing Sub-Geometries...");
            // Use setTimeout to allow DOM to render loader text
            await new Promise(r => setTimeout(r, 50)); 
            
            // For OBJ it might be indexed already, STL is not.
            if (!geometry.index) {
                geometry = BufferGeometryUtils.mergeVertices(geometry);
            }
            geometry.computeVertexNormals();

            geometry.computeBoundingBox();
            const rawW = geometry.boundingBox.max.x - geometry.boundingBox.min.x;
            const rawH = geometry.boundingBox.max.y - geometry.boundingBox.min.y;
            const rawD = geometry.boundingBox.max.z - geometry.boundingBox.min.z;
            
            this.unitScale = parseFloat(this.unitScaleSelect.value);
            const unitLabel = this.unitScaleSelect.options[this.unitScaleSelect.selectedIndex].text;
            
            // Show bounds explicitly in user's UI selection (e.g. mm)
            const w = rawW.toFixed(2);
            const h = rawH.toFixed(2);
            const d = rawD.toFixed(2);
            this.geoBoundsDisplay.innerText = `BBox: ${w} x ${h} x ${d} ${unitLabel}`;

            // Adjust clipping plane constraints
            geometry.scale(this.unitScale, this.unitScale, this.unitScale);
            geometry.computeBoundingBox(); 
            
            const center = new THREE.Vector3();
            geometry.boundingBox.getCenter(center);
            // Ortadan merkeze, ve Y olarak tam taba oturtma
            geometry.translate(-center.x, -geometry.boundingBox.min.y, -center.z); 

            geometry.computeBoundingBox(); 
            this.meshBboxMin = geometry.boundingBox.min;
            this.meshBaseHeight = geometry.boundingBox.max.y - geometry.boundingBox.min.y;
            this.clipPlane.constant = this.meshBboxMin.y + this.meshBaseHeight; // lock to top initially
            
            // Dynamic Grid Generation
            const maxDim = Math.max(geometry.boundingBox.max.x - geometry.boundingBox.min.x, geometry.boundingBox.max.z - geometry.boundingBox.min.z);
            const gridSize = maxDim * 4.0;
            const divisions = Math.max(10, Math.floor(gridSize / (this.unitScale * 10))); // Her kare 10 (mm/cm vb)
            if (this.gridHelper) this.scene.remove(this.gridHelper);
            const isLight = document.body.dataset.theme === 'light';
            this.gridHelper = new THREE.GridHelper(gridSize, divisions, isLight ? 0x2563eb : 0x3b82f6, isLight ? 0x94a3b8 : 0x475569);
            this.gridHelper.position.y = -0.001;
            this.scene.add(this.gridHelper);
            
            geometry.computeBoundingSphere();

            if (this.moldMesh) this.scene.remove(this.moldMesh);
            if (this.highlightMesh) this.scene.remove(this.highlightMesh);
            
            const mat = new THREE.MeshPhongMaterial({ 
                color: 0x94a3b8, 
                transparent: true, 
                opacity: 0.3,
                side: THREE.DoubleSide,
                depthWrite: false,
                clippingPlanes: [this.clipPlane]
            });
            this.moldMesh = new THREE.Mesh(geometry, mat);
            this.scene.add(this.moldMesh);

            // Highlight sub-mesh
            this.highlightMesh = new THREE.Mesh(
                new THREE.BufferGeometry(),
                new THREE.MeshBasicMaterial({ color: 0x3b82f6, side: THREE.DoubleSide, transparent:true, opacity:0.8, polygonOffset: true, polygonOffsetFactor: -1 })
            );
            this.scene.add(this.highlightMesh);
            
            this.camera.position.set(0, geometry.boundingSphere.radius * 2.0, geometry.boundingSphere.radius * 3.0);
            this.controls.target.set(0, geometry.boundingSphere.radius, 0);

            // Build Face Adjacency for flood fill
            this.showLoader("Computing Face Adjacency...");
            await new Promise(r => setTimeout(r, 50)); 
            this.buildAdjacency(geometry);

            this.showLoader("Voxelizing Geometry...");
            const vSizeUI = parseFloat(this.inputVxSize.value);
            const physicsVoxelSize = vSizeUI * this.unitScale;
            const initPad = parseFloat(this.inputPadding.value) || 0;
            
            this.simulationData = await Voxelizer.voxelize(this.moldMesh, physicsVoxelSize, initPad, (prog) => {
                const pct = Math.floor(prog * 100);
                document.getElementById('loader-text').innerText = `Voxelizing ${pct}%`;
            });
            
            // clear old injection lists
            this.injectionSurfaces = [];
            this.selectedFaces.clear();
            this.renderSurfaceList();
            this.surfaceArrows.clear();

            this.btnSimulate.disabled = false;

            // Compute auto dt now that we have geometry + voxel data
            this.computeAutoDt();

        } catch (e) {
            console.error(e);
            alert("Error loading geometry.");
        } finally {
            this.hideLoader();
        }
    }

    buildAdjacency(geometry) {
        const index = geometry.index.array;
        const faceCount = index.length / 3;
        const edges = new Map();

        const addEdge = (v1, v2, faceIdx) => {
            const key = v1 < v2 ? `${v1}_${v2}` : `${v2}_${v1}`;
            if (!edges.has(key)) edges.set(key, []);
            edges.get(key).push(faceIdx);
        };

        for (let i = 0; i < faceCount; i++) {
            const a = index[i*3];
            const b = index[i*3+1];
            const c = index[i*3+2];
            addEdge(a, b, i);
            addEdge(b, c, i);
            addEdge(c, a, i);
        }

        this.faceAdjacency = new Array(faceCount).fill(null).map(() => []);
        edges.forEach((faces) => {
            if (faces.length === 2) {
                this.faceAdjacency[faces[0]].push(faces[1]);
                this.faceAdjacency[faces[1]].push(faces[0]);
            }
        });
        
        // Pre-compute face normals and centers for fast brush/angle
        this.faceNormals = [];
        this.faceCenters = [];
        const pos = geometry.attributes.position.array;
        
        for (let i = 0; i < faceCount; i++) {
            const a = index[i*3];
            const b = index[i*3+1];
            const c = index[i*3+2];
            
            const vA = new THREE.Vector3(pos[a*3], pos[a*3+1], pos[a*3+2]);
            const vB = new THREE.Vector3(pos[b*3], pos[b*3+1], pos[b*3+2]);
            const vC = new THREE.Vector3(pos[c*3], pos[c*3+1], pos[c*3+2]);
            
            const center = new THREE.Vector3().addVectors(vA, vB).add(vC).divideScalar(3);
            this.faceCenters.push(center);
            
            const cb = new THREE.Vector3().subVectors(vC, vB);
            const ab = new THREE.Vector3().subVectors(vA, vB);
            const normal = cb.cross(ab).normalize();
            this.faceNormals.push(normal);
        }
    }

    onPointerDown(e) {
        if (!this.surfaceModeActive || !this.moldMesh) return;
        if (e.target !== this.renderer.domElement) return;

        this.isPainting = true;
        this.controls.enabled = false;
        this.handleRaycast(e, true);
    }

    onPointerMove(e) {
        if (!this.surfaceModeActive || !this.isPainting || !this.moldMesh) return;
        this.handleRaycast(e, false);
    }

    onPointerUp(e) {
        this.isPainting = false;
        this.controls.enabled = true;
    }

    handleRaycast(e, isFirstClick) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.moldMesh, false);

        if (intersects.length > 0) {
            const faceIdx = intersects[0].faceIndex;
            const intersectPt = intersects[0].point;
            const mode = this.selectToolMode.value;

            if (mode === 'single') {
                this.selectedFaces.add(faceIdx);
            } else if (mode === 'brush') {
                const r = parseFloat(document.getElementById('brush-size').value);
                const rSq = r * r;
                for (let i = 0; i < this.faceCenters.length; i++) {
                    if (this.faceCenters[i].distanceToSquared(intersectPt) < rSq) {
                        this.selectedFaces.add(i);
                    }
                }
            } else if (mode === 'angle' && isFirstClick) {
                // Flood fill
                const angleThresh = parseFloat(document.getElementById('angle-limit').value) * Math.PI / 180;
                const cosThresh = Math.cos(angleThresh);
                
                const queue = [faceIdx];
                const visited = new Set();
                
                while(queue.length > 0) {
                    const current = queue.shift();
                    if(visited.has(current)) continue;
                    
                    visited.add(current);
                    this.selectedFaces.add(current);
                    
                    const n1 = this.faceNormals[current];
                    const neighbors = this.faceAdjacency[current];
                    for(let i=0; i<neighbors.length; i++) {
                        const n2 = this.faceNormals[neighbors[i]];
                        if(n1.dot(n2) >= cosThresh) {
                            queue.push(neighbors[i]);
                        }
                    }
                }
            }
            this.updateHighlightMesh();
        }
    }

    updateHighlightMesh() {
        if (!this.moldMesh) return;
        
        // Build a new BufferGeometry holding only the selected faces
        const geom = this.moldMesh.geometry;
        const posView = geom.attributes.position.array;
        const indexView = geom.index.array;
        
        const hlPositions = [];
        this.selectedFaces.forEach(fIdx => {
            const a = indexView[fIdx*3];
            const b = indexView[fIdx*3+1];
            const c = indexView[fIdx*3+2];
            
            hlPositions.push(
                posView[a*3], posView[a*3+1], posView[a*3+2],
                posView[b*3], posView[b*3+1], posView[b*3+2],
                posView[c*3], posView[c*3+1], posView[c*3+2]
            );
        });

        this.highlightMesh.geometry.dispose();
        
        const hlGeom = new THREE.BufferGeometry();
        hlGeom.setAttribute('position', new THREE.Float32BufferAttribute(hlPositions, 3));
        this.highlightMesh.geometry = hlGeom;
    }

    saveInjectionSurface() {
        if (this.selectedFaces.size === 0) return;

        // Calc average normal and center
        let avgNormal = new THREE.Vector3();
        let center = new THREE.Vector3();
        
        this.selectedFaces.forEach(fIdx => {
            avgNormal.add(this.faceNormals[fIdx]);
            center.add(this.faceCenters[fIdx]);
        });
        
        avgNormal.normalize();
        center.divideScalar(this.selectedFaces.size);

        // Map to Voxels
        let inletVoxels = new Set();
        const vs = this.simulationData.voxelSize;
        const { nx, ny, nz, bboxMin } = this.simulationData;
        const grid = this.simulationData.grid;
        
        const IX = (x,y,z) => x + nx * (y + ny * z);

        this.selectedFaces.forEach(fIdx => {
            // Take the centroid and map to grid
            const pt = this.faceCenters[fIdx];
            let vxLoc = Math.floor((pt.x - bboxMin.x) / vs);
            let vyLoc = Math.floor((pt.y - bboxMin.y) / vs);
            let vzLoc = Math.floor((pt.z - bboxMin.z) / vs);
            
            let found = false;
            // Expand search slightly to catch thick cavity
            for(let dx=-1; dx<=1; dx++) {
                for(let dy=-1; dy<=1; dy++) {
                    for(let dz=-1; dz<=1; dz++) {
                        let x = vxLoc+dx, y = vyLoc+dy, z = vzLoc+dz;
                        if(x>=0 && x<nx && y>=0 && y<ny && z>=0 && z<nz) {
                            const idx = IX(x,y,z);
                            if(grid[idx] === 0) {
                                inletVoxels.add(idx); // must be empty mold space
                                found = true;
                            }
                        }
                    }
                }
            }
            
            // Fallback: If Voxelizer perfectly blocked the gap, forcefully carve a hole 
            if (!found && vxLoc>=0 && vxLoc<nx && vyLoc>=0 && vyLoc<ny && vzLoc>=0 && vzLoc<nz) {
                const idx = IX(vxLoc, vyLoc, vzLoc);
                grid[idx] = 0;
                inletVoxels.add(idx);
            }
        });

        const id = 'Sur_' + Math.floor(Math.random()*1000);
        this.injectionSurfaces.push({
            id: id,
            faceIndices: Array.from(this.selectedFaces),
            normal: { x: -avgNormal.x, y: -avgNormal.y, z: -avgNormal.z }, // Invert normal to flow *into* cavity
            voxels: Array.from(inletVoxels),
            displayCenter: center,
            displayNormal: avgNormal
        });

        // Add visual arrow
        const invertedNormal = new THREE.Vector3(-avgNormal.x, -avgNormal.y, -avgNormal.z);
        const arrLength = this.meshBaseHeight ? Math.max(0.01, this.meshBaseHeight * 0.5) : 1.0;
        const arrow = new THREE.ArrowHelper(invertedNormal, center, arrLength, 0xef4444, arrLength*0.3, arrLength*0.15);
        arrow.name = id;
        this.surfaceArrows.add(arrow);

        this.selectedFaces.clear();
        this.updateHighlightMesh();
        this.renderSurfaceList();
    }

    renderSurfaceList() {
        this.surfaceList.innerHTML = '';
        this.injectionSurfaces.forEach((s) => {
            const el = document.createElement('div');
            el.style.background = 'var(--surface-item-bg)';
            el.style.border = '1px solid var(--border-color)';
            el.style.padding = '8px';
            el.style.borderRadius = '4px';
            el.style.display = 'flex';
            el.style.justifyContent = 'space-between';
            el.style.alignItems = 'center';
            el.style.fontSize = '0.85rem';
            
            const nstr = `N: [${s.normal.x.toFixed(1)}, ${s.normal.y.toFixed(1)}, ${s.normal.z.toFixed(1)}]`;
            el.innerHTML = `<div><strong>${s.id}</strong><br><small>${s.voxels.length} Voxels<br>${nstr}</small></div>`;
            
            const delBtn = document.createElement('button');
            delBtn.innerHTML = '<i data-feather="x"></i>';
            delBtn.style.background = 'var(--danger)';
            delBtn.style.color = 'white';
            delBtn.style.border = 'none';
            delBtn.style.borderRadius = '4px';
            delBtn.style.padding = '4px';
            delBtn.style.cursor = 'pointer';
            
            delBtn.onclick = () => {
                this.injectionSurfaces = this.injectionSurfaces.filter(sur => sur.id !== s.id);
                const arr = this.surfaceArrows.getObjectByName(s.id);
                if (arr) this.surfaceArrows.remove(arr);
                this.renderSurfaceList();
            };
            
            el.appendChild(delBtn);
            this.surfaceList.appendChild(el);
        });
        feather.replace();
    }

    initSimulation() {
        this.btnSimulate.disabled = true;
        
        let unifiedInlets = [];
        this.injectionSurfaces.forEach(s => {
            s.voxels.forEach(v => {
                unifiedInlets.push({ idx: v, nx: s.normal.x, ny: s.normal.y, nz: s.normal.z });
            });
        });

        // Fallback default inlet if user skips UI
        if(unifiedInlets.length === 0) {
            alert('No injection surfaces defined. Define a surface first!');
            this.btnSimulate.disabled = false;
            return;
        }

        if (this.fluidMesh) this.scene.remove(this.fluidMesh);
        
        const { grid } = this.simulationData;
        this.fluidCount = 0;
        for(let i=0; i<grid.length; i++) {
            if(grid[i] === 0) this.fluidCount++;
        }

        const vs = this.simulationData.voxelSize;
        const g = new THREE.BoxGeometry(vs, vs, vs);
        const m = new THREE.MeshPhongMaterial({ 
            color: 0xffffff, 
            emissive: 0xff5500,
            emissiveIntensity: 0.7,
            shininess: 80,
            clippingPlanes: [this.clipPlane]
        });
        
        this.fluidMesh = new THREE.InstancedMesh(g, m, this.fluidCount);
        this.fluidMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
        this.fluidMesh.instanceColor = new THREE.InstancedBufferAttribute( new Float32Array( this.fluidCount * 3 ), 3 );
        this.fluidMesh.instanceColor.setUsage( THREE.DynamicDrawUsage );
        this.fluidMesh.frustumCulled = false;
        this.fluidMesh.renderOrder = 1;
        this.scene.add(this.fluidMesh);
        
        for(let i=0; i<this.fluidCount; i++) {
            this.dummy.position.set(9999,9999,9999);
            this.dummy.scale.set(0,0,0);
            this.dummy.updateMatrix();
            this.fluidMesh.setMatrixAt(i, this.dummy.matrix);
        }
        this.fluidMesh.instanceMatrix.needsUpdate = true;

        const simConfig = {
            nx: this.simulationData.nx,
            ny: this.simulationData.ny,
            nz: this.simulationData.nz,
            voxelSize: this.simulationData.voxelSize,
            dt: parseFloat(this.inputDt.value),
            grid: this.simulationData.grid,
            material: this.currentMatLine,
            moldTemp: parseFloat(document.getElementById('mold-temp').value),
            injectTemp: parseFloat(document.getElementById('inject-temp').value),
            injectPressure: parseFloat(document.getElementById('inject-pressure').value),
            solverIters: parseInt(document.getElementById('solver-iters').value),
            compressibility: parseFloat(document.getElementById('compressibility').value)
        };

        if (this.useGPU && this.gpuEngine) {
            // GPU path
            this.gpuEngine.configure(simConfig);
            this.gpuEngine.setInlets(unifiedInlets);
            this.btnSimulate.disabled = false;
            this.btnReset.disabled = false;
            this.btnPlayPause.innerHTML = '<i data-feather="pause"></i>';
            feather.replace();
            this.playbackUI.style.display = 'flex';
            this.legendOverlay.style.display = 'block';
            this.drawLegend();
            this.frames = [];
            this.timeSlider.value = 0;
            this.timeSlider.max = 0;
            this.playing = true;
            this.gpuRunning = true;
            this.gpuSimLoop();
        } else {
            // CPU Worker path
            this.worker.postMessage({ type: 'INIT', config: simConfig });
            this.worker.postMessage({ type: 'SET_INLETS_WITH_NORMALS', inlets: unifiedInlets });
        }
    }

    // ─── GPU Simulation Loop ────────────────────────────────
    async gpuSimLoop() {
        if (!this.gpuRunning) return;

        this.gpuEngine.step(4); // 4 physics steps per render
        const { buffer, time } = await this.gpuEngine.readVisualData();

        this.onSimData(buffer, time);

        if (this.gpuRunning) {
            requestAnimationFrame(() => this.gpuSimLoop());
        }
    }

    // ─── Unified simulation data handler ───────────────────
    onSimData(buffer, time) {
        this.frames.push(buffer);

        if (this.frames.length > 3000) {
            this.frames.splice(0, 1000);
            const curVal = parseInt(this.timeSlider.value);
            this.timeSlider.value = Math.max(0, Math.min(curVal - 1000, this.frames.length - 1));
        }

        this.timeSlider.max = this.frames.length - 1;

        if (this.playing) {
            this.timeSlider.value = this.frames.length - 1;
            this.updateFluidMesh(buffer);
            this.timeDisplay.innerText = time.toFixed(3) + 's';
        }
    }

    // ─── CRITICAL FIX: Buffer offset correction ──────────────
    updateFluidMesh(buffer) {
        if(!this.fluidMesh) return;
        
        // FIX: Buffer stride is exactly 8 floats per voxel, no header offset
        const numActive = Math.floor(buffer.length / 8);
        const vs = this.simulationData.voxelSize;
        const nx = this.simulationData.nx;
        const ny = this.simulationData.ny;
        const { bboxMin } = this.simulationData;

        this.fluidMesh.count = numActive;

        for (let i = 0; i < numActive; i++) {
            // FIX: offset = i * 8 (no +1 header — sendVisualData has no header)
            const offset = i * 8;
            const index = buffer[offset];
            const fillVal = buffer[offset + 1];
            const T = buffer[offset + 2];
            const vxVal = buffer[offset + 3];
            const vyVal = buffer[offset + 4];
            const vzVal = buffer[offset + 5];
            const pVal = buffer[offset + 6];
            const sFraction = buffer[offset + 7];

            const z = Math.floor(index / (nx * ny));
            const r = index % (nx * ny);
            const y = Math.floor(r / nx);
            const x = r % nx;

            const bx = bboxMin.x + x * vs + vs/2;
            const by = bboxMin.y + y * vs + vs/2;
            const bz = bboxMin.z + z * vs + vs/2;
            
            this.dummy.position.set(bx, by, bz);
            const scaleFill = Math.pow(fillVal, 0.33); 
            this.dummy.scale.set(scaleFill, scaleFill, scaleFill);
            this.dummy.updateMatrix();
            this.fluidMesh.setMatrixAt(i, this.dummy.matrix);

            if (this.viewMode === 'fill') {
                // Bright orange→blue colormap for fill fraction
                this.colorBuffer.setHSL(0.08 + fillVal * 0.52, 0.95, 0.35 + fillVal * 0.3);
            } else if (this.viewMode === 'temperature') {
                const Tmin = this.currentMatLine.solidusTemp - 100;
                const Tmax = this.currentMatLine.liquidusTemp + 100;
                const normT = Math.max(0, Math.min(1, (T - Tmin) / (Tmax - Tmin)));
                this.colorBuffer.setHSL((1.0 - normT) * 0.66, 1.0, 0.5);
            } else if (this.viewMode === 'velocity') {
                const mag = Math.sqrt(vxVal*vxVal + vyVal*vyVal + vzVal*vzVal);
                const normV = Math.min(1.0, mag / 3.0); 
                this.colorBuffer.setHSL(0.3 + (1.0 - normV) * 0.4, 0.8, 0.5);
            } else if (this.viewMode === 'pressure') {
                const maxP = parseFloat(document.getElementById('inject-pressure').value) || 100000;
                const normP = Math.max(0, Math.min(1, pVal / maxP));
                this.colorBuffer.setHSL((1.0 - normP) * 0.66, 1.0, 0.5); 
            } else if (this.viewMode === 'solid') {
                const normS = Math.max(0, Math.min(1, sFraction));
                this.colorBuffer.setHSL(0.1, 1.0 - normS, 0.5);
            }

            this.fluidMesh.setColorAt(i, this.colorBuffer);
        }
        
        this.fluidMesh.instanceMatrix.needsUpdate = true;
        if(this.fluidMesh.instanceColor) this.fluidMesh.instanceColor.needsUpdate = true;
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    showLoader(msg) {
        document.getElementById('loader-text').innerText = msg;
        this.loader.classList.remove('hidden');
    }

    exportData() {
        if (!this.frames || this.frames.length === 0) {
            alert('No simulation data to export. Run a simulation first.');
            return;
        }

        const exportObj = {
            metadata: {
                nx: this.simulationData?.nx,
                ny: this.simulationData?.ny,
                nz: this.simulationData?.nz,
                voxelSize: this.simulationData?.voxelSize,
                bboxMin: this.simulationData?.bboxMin,
                material: this.currentMatLine?.name,
                totalFrames: this.frames.length
            },
            // Export last frame only to keep file size manageable
            lastFrame: Array.from(this.frames[this.frames.length - 1])
        };

        const blob = new Blob([JSON.stringify(exportObj)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `castflow_export_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    hideLoader() {
        this.loader.classList.add('hidden');
    }

    // ─── Legend Drawing ───────────────────────────────────────
    drawLegend() {
        const canvas = this.legendCanvas;
        const ctx = canvas.getContext('2d');
        const h = canvas.height;
        const w = canvas.width;

        ctx.clearRect(0, 0, w, h);

        for (let y = 0; y < h; y++) {
            const t = 1.0 - y / h; // top = max value, bottom = min
            let hue, sat, light;

            if (this.viewMode === 'fill') {
                hue = (0.08 + t * 0.52) * 360;
                sat = 95;
                light = (0.35 + t * 0.3) * 100;
            } else if (this.viewMode === 'temperature') {
                hue = (1.0 - t) * 0.66 * 360;
                sat = 100;
                light = 50;
            } else if (this.viewMode === 'velocity') {
                hue = (0.3 + (1.0 - t) * 0.4) * 360;
                sat = 80;
                light = 50;
            } else if (this.viewMode === 'pressure') {
                hue = (1.0 - t) * 0.66 * 360;
                sat = 100;
                light = 50;
            } else if (this.viewMode === 'solid') {
                hue = 36;
                sat = (1.0 - t) * 100;
                light = 50;
            }

            ctx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
            ctx.fillRect(0, y, w, 1);
        }

        // Update labels
        const mat = this.currentMatLine;
        if (this.viewMode === 'fill') {
            this.legendTitle.innerText = 'Fill Fraction';
            this.legendMax.innerText = '1.0';
            this.legendMin.innerText = '0.0';
        } else if (this.viewMode === 'temperature') {
            const Tmin = mat ? mat.solidusTemp - 100 : 0;
            const Tmax = mat ? mat.liquidusTemp + 100 : 1000;
            this.legendTitle.innerText = 'Temperature';
            this.legendMax.innerText = Tmax + '°C';
            this.legendMin.innerText = Tmin + '°C';
        } else if (this.viewMode === 'velocity') {
            this.legendTitle.innerText = 'Velocity';
            this.legendMax.innerText = '3.0 m/s';
            this.legendMin.innerText = '0.0 m/s';
        } else if (this.viewMode === 'pressure') {
            const maxP = parseFloat(document.getElementById('inject-pressure').value) || 100000;
            this.legendTitle.innerText = 'Pressure';
            this.legendMax.innerText = (maxP / 1000).toFixed(0) + ' kPa';
            this.legendMin.innerText = '0 kPa';
        } else if (this.viewMode === 'solid') {
            this.legendTitle.innerText = 'Solid Fraction';
            this.legendMax.innerText = '1.0 (Solid)';
            this.legendMin.innerText = '0.0 (Liquid)';
        }
    }

    resetCamera() {
        if (!this.moldMesh) {
            this.camera.position.set(0, 5, 10);
            this.controls.target.set(0, 0, 0);
        } else {
            const h = this.meshBaseHeight || 1.0;
            const r = h * 1.25; 
            this.camera.position.set(0, r * 1.5, r * 2.5);
            this.controls.target.set(0, r * 0.5, 0);
        }
        this.controls.update();
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        // Disable controls while painting
        if(this.surfaceModeActive && this.isPainting) {
            this.controls.enabled = false;
        } else if (!this.surfaceModeActive) {
            this.controls.enabled = true;
        }

        this.controls.update();

        // ─── Dynamic near/far to prevent zoom clipping ───────
        const dist = this.camera.position.distanceTo(this.controls.target);
        this.camera.near = Math.max(dist * 0.001, 0.0001);
        this.camera.far = Math.max(dist * 100, 100);
        this.camera.updateProjectionMatrix();

        this.renderer.render(this.scene, this.camera);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new CastFlowApp();
});
