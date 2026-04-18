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
        this.initWorker();
        this.loadMaterials();
        
        this.simulationData = null;
        this.viewMode = 'fill';
        this.moldMesh = null;
        this.fluidMesh = null;
        this.materialsMap = {};
        this.currentMatLine = null;
        this.fluidCount = 0;
        
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
        
        this.inputGeometry = document.getElementById('geometry-upload');
        this.selectMat = document.getElementById('material-select');
        this.inputVxSize = document.getElementById('voxel-size');
        this.inputPadding = document.getElementById('domain-padding');
        this.inputDt = document.getElementById('time-step');
        
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

    initThree() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a);
        
        this.camera = new THREE.PerspectiveCamera(45, this.container.clientWidth / this.container.clientHeight, 0.0001, 1000);
        this.camera.position.set(0, 5, 10);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
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
                this.frames = [];
                this.timeSlider.value = 0;
                this.timeSlider.max = 0;
                this.playing = true;
                this.worker.postMessage({ type: 'START' });
            } else if (data.type === 'RENDER_DATA') {
                this.frames.push(data.buffer);
                this.timeSlider.max = this.frames.length - 1;
                
                // Memory preservation - hard cap history at 3000 frames (~70MB)
                if (this.frames.length > 3000) {
                    this.frames.splice(0, 1000); // discard oldest memory
                }
                
                if (this.playing) {
                   this.timeSlider.value = this.frames.length - 1;
                   this.updateFluidMesh(data.buffer);
                   this.timeDisplay.innerText = data.time.toFixed(3) + 's';
                }
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
    }

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
                }
                const g = new THREE.BoxGeometry(this.simulationData.voxelSize * 0.95, this.simulationData.voxelSize * 0.95, this.simulationData.voxelSize * 0.95);
                const m = new THREE.MeshPhongMaterial({ 
                    color: 0xffffff,
                    emissive: 0xff5500, // Explicit visible glow
                    emissiveIntensity: 0.8,
                    shininess: 100,
                    clippingPlanes: [this.clipPlane]
                });
                
                this.fluidMesh = new THREE.InstancedMesh(g, m, this.fluidCount);
                this.fluidMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
                this.scene.add(this.fluidMesh);

                this.selectedFaces.clear();
                this.injectionSurfaces = [];
                
                while (this.surfaceArrows.children.length > 0) {
                    this.surfaceArrows.remove(this.surfaceArrows.children[0]);
                }
                
                this.renderSurfaceList();
                this.updateHighlightMesh();

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
             this.worker.postMessage({ type: 'PAUSE' });
             if (this.fluidMesh) {
                 this.scene.remove(this.fluidMesh);
                 this.fluidMesh = null;
             }
             this.timeDisplay.innerText = "0.0s";
             this.btnSimulate.disabled = false;
             this.playbackUI.style.display = 'none';
             this.frames = [];
             this.playing = false;
        });

        this.btnToggleMold.addEventListener('click', () => {
            if (this.moldMesh) {
                // Add Gimbal/Axes to world origin mapped relative to their Bounding Box to visualize coordinates
                if(!this.axesHelper) {
                    this.axesHelper = new THREE.AxesHelper( this.meshBaseHeight * 1.5 );
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

        this.visButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.visButtons.forEach(b => b.classList.remove('active'));
                const t = e.target.closest('.mode-btn');
                t.classList.add('active');
                this.viewMode = t.getAttribute('data-mode');
                // Refresh visualization if frozen
                if (!this.playing && this.frames.length > 0) {
                    this.updateFluidMesh(this.frames[parseInt(this.timeSlider.value)]);
                }
            });
        });
        
        this.btnPlayPause.addEventListener('click', () => {
            this.playing = !this.playing;
            this.worker.postMessage({ type: this.playing ? 'START' : 'PAUSE' });
            this.btnPlayPause.innerHTML = this.playing ? '<i data-feather="pause"></i>' : '<i data-feather="play"></i>';
            feather.replace();
        });

        this.btnStep.addEventListener('click', () => {
            if (!this.playing) this.worker.postMessage({ type: 'STEP' });
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
            const label = e.target.options[e.target.selectedIndex].text;
            document.getElementById('voxel-unit-label').innerText = ' ' + label;
            this.unitScale = parseFloat(e.target.value);
            // Optionally scale the input text
            let current = parseFloat(this.inputVxSize.value);
            if(label === 'mm') this.inputVxSize.value = (current >= 1.0 ? current : 2.0).toFixed(1);
            if(label === 'm') this.inputVxSize.value = (current < 1.0 ? current : 0.002).toFixed(3);
        });
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
            this.gridHelper = new THREE.GridHelper(gridSize, divisions, 0x3b82f6, 0x475569);
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
        // Simple approach: test bounding box of surface vs voxels, or just cast grid points onto surface
        // Since we want standard inlet boundaries, any voxel touching a selected face is an inlet.
        // Fast approximation: for each face center, find nearest voxel in cavity
        
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
            el.style.background = 'rgba(255,255,255,0.05)';
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

        const g = new THREE.BoxGeometry(this.simulationData.voxelSize * 0.95, this.simulationData.voxelSize * 0.95, this.simulationData.voxelSize * 0.95);
        const m = new THREE.MeshStandardMaterial({ 
            color: 0xffffff, roughness: 0.2, metalness: 0.8,
            clippingPlanes: [this.clipPlane]
        });
        
        this.fluidMesh = new THREE.InstancedMesh(g, m, this.fluidCount);
        this.fluidMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
        this.fluidMesh.instanceColor = new THREE.InstancedBufferAttribute( new Float32Array( this.fluidCount * 3 ), 3 );
        this.fluidMesh.instanceColor.setUsage( THREE.DynamicDrawUsage );
        this.scene.add(this.fluidMesh);
        
        for(let i=0; i<this.fluidCount; i++) {
            this.dummy.position.set(9999,9999,9999);
            this.dummy.scale.set(0,0,0);
            this.dummy.updateMatrix();
            this.fluidMesh.setMatrixAt(i, this.dummy.matrix);
        }
        this.fluidMesh.instanceMatrix.needsUpdate = true;

        this.worker.postMessage({
            type: 'INIT',
            config: {
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
            }
        });

        this.worker.postMessage({
            type: 'SET_INLETS_WITH_NORMALS',
            inlets: unifiedInlets
        });
    }

    updateFluidMesh(buffer) {
        if(!this.fluidMesh) return;
        
        const numActive = Math.floor((buffer.length - 1) / 8);
        const vs = this.simulationData.voxelSize;
        const nx = this.simulationData.nx;
        const ny = this.simulationData.ny;
        const { bboxMin } = this.simulationData;

        this.fluidMesh.count = numActive;

        for (let i = 0; i < numActive; i++) {
            let offset = i * 8 + 1; // +1 mathematically offsets the step time encoded at buffer[0]
            let index = buffer[offset];
            let fill = buffer[offset + 1];
            let T = buffer[offset + 2];
            let vx = buffer[offset + 3];
            let vy = buffer[offset + 4];
            let vz = buffer[offset + 5];
            let pVal = buffer[offset + 6];
            let sFraction = buffer[offset + 7];

            let z = Math.floor(index / (nx * ny));
            let r = index % (nx * ny);
            let y = Math.floor(r / nx);
            let x = r % nx;

            let bx = bboxMin.x + x * vs + vs/2;
            let by = bboxMin.y + y * vs + vs/2;
            let bz = bboxMin.z + z * vs + vs/2;
            
            this.dummy.position.set(bx, by, bz);
            let scaleFill = Math.pow(fill, 0.33); 
            this.dummy.scale.set(scaleFill, scaleFill, scaleFill);
            this.dummy.updateMatrix();
            this.fluidMesh.setMatrixAt(i, this.dummy.matrix);

            if (this.viewMode === 'fill') {
                this.colorBuffer.setHSL(0.6, 0.9, 0.2 + fill * 0.6); 
            } else if (this.viewMode === 'temperature') {
                let Tmin = this.currentMatLine.solidusTemp - 100;
                let Tmax = this.currentMatLine.liquidusTemp + 100;
                let normT = Math.max(0, Math.min(1, (T - Tmin) / (Tmax - Tmin)));
                this.colorBuffer.setHSL((1.0 - normT) * 0.66, 1.0, 0.5);
            } else if (this.viewMode === 'velocity') {
                let mag = Math.sqrt(vx*vx + vy*vy + vz*vz);
                let normV = Math.min(1.0, mag / 3.0); 
                this.colorBuffer.setHSL(0.3 + (1.0 - normV) * 0.4, 0.8, 0.5);
            } else if (this.viewMode === 'pressure') {
                let maxP = parseFloat(document.getElementById('inject-pressure').value) || 100000;
                let normP = Math.max(0, Math.min(1, pVal / maxP));
                this.colorBuffer.setHSL((1.0 - normP) * 0.66, 1.0, 0.5); 
            } else if (this.viewMode === 'solid') {
                let normS = Math.max(0, Math.min(1, sFraction));
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

    hideLoader() {
        this.loader.classList.add('hidden');
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
        this.renderer.render(this.scene, this.camera);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new CastFlowApp();
});
