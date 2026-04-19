/**
 * worker.js — CastFlow Simulation Engine
 * 
 * CFD Architecture:
 *  - Incompressible Navier-Stokes on collocated grid
 *  - Chorin projection method for pressure-velocity coupling
 *  - Fill fraction transport via Upwind Finite-Volume (donor-cell)
 *  - Velocity extension at free surface for fill front propagation
 *  - Red-Black Gauss-Seidel pressure solver
 *  - Integrated heat transfer (explicit diffusion) & solidification
 *  - Adaptive CFL sub-stepping
 */

let nx, ny, nz, nxny;
let totalCells;
let voxelSize = 0.01;
let dt = 0.01;
let configDt = 0.01;

// Simulation Arrays (Collocated grid)
let moldGrid;   // Uint8Array: 1 = mold, 0 = cavity
let isInlet;    // Uint8Array: 1 = inlet
let fill;       // Float32Array: 0 to 1
let fillNew;
let T;          // Float32Array: Temperature (°C)
let TNew;
let sFraction;  // Float32Array: Solid fraction (0 to 1)
let p;          // Float32Array: Pressure (solver units)
let div;        // Float32Array: Divergence
let vx, vy, vz; // Float32Array: Velocity (m/s)
let vxNew, vyNew, vzNew;

// Properties
let mat;
let moldTemp;
let injectTemp;
let injectPressure;
let solverIters;
let compressibility = 0.0001;
let inletIndices = [];
let inletNormals = new Map();
let inletSpeed = 1.0;

// Performance: cavity index list
let cavityIndices = null;
let cavityCount = 0;

let running = false;
let stepCount = 0;

// Batch stepping: physics steps per render callback
const STEPS_PER_RENDER = 4;

self.onmessage = function(e) {
    const msg = e.data;
    if (msg.type === 'INIT') {
        initSimulation(msg.config);
    } else if (msg.type === 'START') {
        running = true;
        simulateLoop();
    } else if (msg.type === 'PAUSE') {
        running = false;
    } else if (msg.type === 'STEP') {
        stepSimulation();
        sendVisualData();
    } else if (msg.type === 'SET_INLETS_WITH_NORMALS') {
        inletIndices = [];
        inletNormals = new Map();
        const inlets = msg.inlets;
        for (let i = 0; i < inlets.length; i++) {
            const inf = inlets[i];
            inletIndices.push(inf.idx);
            isInlet[inf.idx] = 1;
            moldGrid[inf.idx] = 0; // Ensure physical volume exists
            inletNormals.set(inf.idx, { nx: inf.nx, ny: inf.ny, nz: inf.nz });
        }
        // Rebuild cavity list after inlet modification
        buildCavityIndex();
    }
};

function initSimulation(config) {
    nx = config.nx;
    ny = config.ny;
    nz = config.nz;
    nxny = nx * ny;
    totalCells = nx * ny * nz;
    voxelSize = config.voxelSize;
    dt = config.dt;
    configDt = config.dt;
    
    mat = config.material;
    moldTemp = config.moldTemp;
    injectTemp = config.injectTemp;
    injectPressure = config.injectPressure || 100000;
    solverIters = config.solverIters || 20;
    compressibility = config.compressibility || 0.0001;

    // Compute inlet speed from Bernoulli: v = sqrt(2P/ρ), cap for stability
    inletSpeed = Math.min(10.0, Math.sqrt(2.0 * injectPressure / mat.density));

    // Load geometry
    moldGrid = config.grid;

    // Alloc arrays
    isInlet = new Uint8Array(totalCells);
    fill = new Float32Array(totalCells);
    fillNew = new Float32Array(totalCells);
    T = new Float32Array(totalCells);
    TNew = new Float32Array(totalCells);
    sFraction = new Float32Array(totalCells);
    p = new Float32Array(totalCells);
    div = new Float32Array(totalCells);
    vx = new Float32Array(totalCells);
    vy = new Float32Array(totalCells);
    vz = new Float32Array(totalCells);
    vxNew = new Float32Array(totalCells);
    vyNew = new Float32Array(totalCells);
    vzNew = new Float32Array(totalCells);

    // Initialize temperature
    for (let i = 0; i < totalCells; i++) {
        T[i] = moldTemp;
    }

    // Build cavity index
    buildCavityIndex();
    
    stepCount = 0;
    self.postMessage({ type: 'INIT_DONE' });
}

function buildCavityIndex() {
    const tmp = [];
    for (let i = 0; i < totalCells; i++) {
        if (moldGrid[i] === 0) tmp.push(i);
    }
    cavityIndices = new Int32Array(tmp);
    cavityCount = tmp.length;
}

function simulateLoop() {
    if (!running) return;
    
    for (let b = 0; b < STEPS_PER_RENDER; b++) {
        stepSimulation();
    }
    
    sendVisualData();
    if (running) {
        setTimeout(simulateLoop, 0);
    }
}

function stepSimulation() {
    // 1. Apply inlet boundary conditions
    applySources();

    // 2. Adaptive CFL sub-stepping
    let maxV = 1.0;
    for (let c = 0; c < cavityCount; c++) {
        const i = cavityIndices[c];
        const vel = Math.abs(vx[i]) + Math.abs(vy[i]) + Math.abs(vz[i]);
        if (vel > maxV) maxV = vel;
    }

    let safeDt = (voxelSize * 0.35) / maxV;
    safeDt = Math.min(safeDt, configDt); // Never exceed user dt
    
    let substeps = Math.ceil(configDt / safeDt);
    substeps = Math.min(substeps, 200);
    dt = configDt / substeps;

    for (let s = 0; s < substeps; s++) {
        // Gravity on filled cells
        addGravity();
        
        // Pressure projection (enforces incompressibility in filled region)
        project();
        
        // Extend velocity field one cell beyond fill front
        extendVelocity();
        
        // Transport fill fraction via upwind finite-volume
        advectFill();
        
        // Advect velocity & temperature via semi-Lagrangian (in filled region)
        advectVelocityTemperature();
        
        // Re-enforce inlet BCs
        applySources();
    }
    
    // Thermal physics (once per outer step)
    computeHeatTransfer();
    computeSolidification();
    
    stepCount++;
}

// ─── Index Helpers ──────────────────────────────────────────

function IX(x, y, z) {
    if (x < 0) x = 0; else if (x > nx-1) x = nx-1;
    if (y < 0) y = 0; else if (y > ny-1) y = ny-1;
    if (z < 0) z = 0; else if (z > nz-1) z = nz-1;
    return x + nx * (y + ny * z);
}

// ─── Source Terms & BCs ─────────────────────────────────────

function applySources() {
    for (let k = 0; k < inletIndices.length; k++) {
        const i = inletIndices[k];
        fill[i] = 1.0;
        T[i] = injectTemp;
        sFraction[i] = 0;
        
        const n = inletNormals.get(i);
        if (n) {
            vx[i] = n.nx * inletSpeed;
            vy[i] = n.ny * inletSpeed;
            vz[i] = n.nz * inletSpeed;
        }
    }
}

function addGravity() {
    const gdt = -9.81 * dt;
    for (let c = 0; c < cavityCount; c++) {
        const i = cavityIndices[c];
        if (fill[i] > 0.01 && sFraction[i] < 0.99) {
            vy[i] += gdt;
        }
    }
}

// ─── Velocity Extension ─────────────────────────────────────
// Extrapolate velocity from filled cells to adjacent empty cavity cells
// so that upwind fill transport has non-zero face velocities at the front.

function extendVelocity() {
    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                // Only extend to empty cavity cells
                if (moldGrid[idx] === 0 && fill[idx] < 0.01 && isInlet[idx] === 0) {
                    let sumVx = 0, sumVy = 0, sumVz = 0;
                    let count = 0;
                    
                    // Check 6 neighbors for filled cells
                    if (x > 0) {
                        const ni = idx - 1;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    if (x < nx-1) {
                        const ni = idx + 1;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    if (y > 0) {
                        const ni = idx - nx;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    if (y < ny-1) {
                        const ni = idx + nx;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    if (z > 0) {
                        const ni = idx - nxny;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    if (z < nz-1) {
                        const ni = idx + nxny;
                        if (moldGrid[ni] === 0 && fill[ni] > 0.01) {
                            sumVx += vx[ni]; sumVy += vy[ni]; sumVz += vz[ni]; count++;
                        }
                    }
                    
                    if (count > 0) {
                        const inv = 1.0 / count;
                        vx[idx] = sumVx * inv;
                        vy[idx] = sumVy * inv;
                        vz[idx] = sumVz * inv;
                    }
                }
                idx++;
            }
        }
    }
}

// ─── Fill Transport (Upwind Finite Volume) ──────────────────
// Donor-cell method: compute face fluxes using upwind direction.
// This correctly propagates fill from filled cells into empty neighbors,
// unlike semi-Lagrangian which requires non-zero velocity in the target cell.

function advectFill() {
    const dtOverDx = dt / voxelSize;
    
    // Copy current fill
    fillNew.set(fill);
    
    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (moldGrid[idx] === 1 || isInlet[idx] === 1) {
                    idx++;
                    continue;
                }
                
                let netFlux = 0;
                
                // X-direction faces
                // Left face (x - 1/2): flux INTO this cell from left
                if (x > 0 && moldGrid[idx - 1] === 0) {
                    const vFace = 0.5 * (vx[idx] + vx[idx - 1]);
                    if (vFace > 0) {
                        // Flow from left to right: donor is left cell
                        netFlux += fill[idx - 1] * vFace * dtOverDx;
                    } else {
                        // Flow from right to left: donor is this cell (outgoing)
                        netFlux += fill[idx] * vFace * dtOverDx;
                    }
                }
                
                // Right face (x + 1/2): flux OUT of this cell to right
                if (x < nx-1 && moldGrid[idx + 1] === 0) {
                    const vFace = 0.5 * (vx[idx] + vx[idx + 1]);
                    if (vFace > 0) {
                        // Flow from left to right: donor is this cell (outgoing)
                        netFlux -= fill[idx] * vFace * dtOverDx;
                    } else {
                        // Flow from right to left: donor is right cell (incoming)
                        netFlux -= fill[idx + 1] * vFace * dtOverDx;
                    }
                }
                
                // Y-direction faces
                // Bottom face (y - 1/2)
                if (y > 0 && moldGrid[idx - nx] === 0) {
                    const vFace = 0.5 * (vy[idx] + vy[idx - nx]);
                    if (vFace > 0) {
                        netFlux += fill[idx - nx] * vFace * dtOverDx;
                    } else {
                        netFlux += fill[idx] * vFace * dtOverDx;
                    }
                }
                
                // Top face (y + 1/2)
                if (y < ny-1 && moldGrid[idx + nx] === 0) {
                    const vFace = 0.5 * (vy[idx] + vy[idx + nx]);
                    if (vFace > 0) {
                        netFlux -= fill[idx] * vFace * dtOverDx;
                    } else {
                        netFlux -= fill[idx + nx] * vFace * dtOverDx;
                    }
                }
                
                // Z-direction faces
                // Back face (z - 1/2)
                if (z > 0 && moldGrid[idx - nxny] === 0) {
                    const vFace = 0.5 * (vz[idx] + vz[idx - nxny]);
                    if (vFace > 0) {
                        netFlux += fill[idx - nxny] * vFace * dtOverDx;
                    } else {
                        netFlux += fill[idx] * vFace * dtOverDx;
                    }
                }
                
                // Front face (z + 1/2)
                if (z < nz-1 && moldGrid[idx + nxny] === 0) {
                    const vFace = 0.5 * (vz[idx] + vz[idx + nxny]);
                    if (vFace > 0) {
                        netFlux -= fill[idx] * vFace * dtOverDx;
                    } else {
                        netFlux -= fill[idx + nxny] * vFace * dtOverDx;
                    }
                }
                
                fillNew[idx] = Math.max(0.0, Math.min(1.0, fill[idx] + netFlux));
                idx++;
            }
        }
    }
    
    // Swap
    const tmp = fill;
    fill = fillNew;
    fillNew = tmp;
}

// ─── Velocity & Temperature Advection (Semi-Lagrangian) ─────
// Applied only to filled cells. Fill transport is handled separately above.

function advectVelocityTemperature() {
    const dtInvDx = dt / voxelSize;
    
    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                // Skip mold walls and inlets
                if (moldGrid[idx] === 1 || isInlet[idx] === 1) {
                    vxNew[idx] = vx[idx];
                    vyNew[idx] = vy[idx];
                    vzNew[idx] = vz[idx];
                    TNew[idx] = T[idx];
                    idx++;
                    continue;
                }
                
                // Solidified — freeze velocity
                if (sFraction[idx] >= 0.99) {
                    vxNew[idx] = 0;
                    vyNew[idx] = 0;
                    vzNew[idx] = 0;
                    TNew[idx] = T[idx];
                    idx++;
                    continue;
                }
                
                // Only advect velocity for cells with fluid
                if (fill[idx] > 0.01) {
                    const px = x - dtInvDx * vx[idx];
                    const py = y - dtInvDx * vy[idx];
                    const pz = z - dtInvDx * vz[idx];
                    
                    vxNew[idx] = interpolate(vx, px, py, pz) * 0.998;
                    vyNew[idx] = interpolate(vy, px, py, pz) * 0.998;
                    vzNew[idx] = interpolate(vz, px, py, pz) * 0.998;
                    TNew[idx] = interpolate(T, px, py, pz);
                } else {
                    vxNew[idx] = 0;
                    vyNew[idx] = 0;
                    vzNew[idx] = 0;
                    TNew[idx] = T[idx];
                }
                
                idx++;
            }
        }
    }
    
    // Swap
    let tmp;
    tmp = vx; vx = vxNew; vxNew = tmp;
    tmp = vy; vy = vyNew; vyNew = tmp;
    tmp = vz; vz = vzNew; vzNew = tmp;
    tmp = T;  T  = TNew;  TNew  = tmp;
}

function interpolate(array, cx, cy, cz) {
    let x0 = Math.floor(cx); let x1 = x0 + 1;
    let y0 = Math.floor(cy); let y1 = y0 + 1;
    let z0 = Math.floor(cz); let z1 = z0 + 1;

    const sx1 = cx - x0; const sx0 = 1.0 - sx1;
    const sy1 = cy - y0; const sy0 = 1.0 - sy1;
    const sz1 = cz - z0; const sz0 = 1.0 - sz1;
    
    return sx0 * (
        sy0 * (sz0 * array[IX(x0, y0, z0)] + sz1 * array[IX(x0, y0, z1)]) +
        sy1 * (sz0 * array[IX(x0, y1, z0)] + sz1 * array[IX(x0, y1, z1)])
    ) + sx1 * (
        sy0 * (sz0 * array[IX(x1, y0, z0)] + sz1 * array[IX(x1, y0, z1)]) +
        sy1 * (sz0 * array[IX(x1, y1, z0)] + sz1 * array[IX(x1, y1, z1)])
    );
}

// ─── Pressure Projection ────────────────────────────────────
// Chorin fractional-step method:
//   1. Compute divergence of intermediate velocity field
//   2. Solve Poisson equation for pressure (RB Gauss-Seidel)
//   3. Correct velocity to be divergence-free
// 
// NO physical pressure units. Solver works in Stam/grid units.
// Inlet velocity is the driving BC, not inlet pressure.

function project() {
    const halfDx = 0.5 * voxelSize;
    const invDx = 1.0 / voxelSize;
    const fillThreshold = 0.01; // Include partially filled cells
    
    // ── Divergence computation ──
    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                p[idx] = 0;
                
                if (moldGrid[idx] === 1 || fill[idx] < fillThreshold) {
                    div[idx] = 0;
                    idx++;
                    continue;
                }
                
                // Face velocities with wall BCs (zero velocity at mold walls)
                const vxR = (x < nx-1 && moldGrid[idx + 1]     === 0) ? vx[idx + 1]    : 0;
                const vxL = (x > 0    && moldGrid[idx - 1]     === 0) ? vx[idx - 1]    : 0;
                const vyU = (y < ny-1 && moldGrid[idx + nx]    === 0) ? vy[idx + nx]   : 0;
                const vyD = (y > 0    && moldGrid[idx - nx]    === 0) ? vy[idx - nx]   : 0;
                const vzF = (z < nz-1 && moldGrid[idx + nxny]  === 0) ? vz[idx + nxny] : 0;
                const vzB = (z > 0    && moldGrid[idx - nxny]  === 0) ? vz[idx - nxny] : 0;
                
                div[idx] = -halfDx * ((vxR - vxL) + (vyU - vyD) + (vzF - vzB));
                
                idx++;
            }
        }
    }

    // ── Red-Black Gauss-Seidel Pressure Solve ──
    for (let k = 0; k < solverIters; k++) {
        for (let color = 0; color < 2; color++) {
            let i = 0;
            for (let z = 0; z < nz; z++) {
                for (let y = 0; y < ny; y++) {
                    for (let x = 0; x < nx; x++) {
                        if (((x + y + z) & 1) === color) {
                            if (moldGrid[i] === 0 && fill[i] > fillThreshold) {
                                // Count valid (non-wall) neighbors and sum their pressures
                                let sumP = 0;
                                let nCount = 0;
                                
                                if (x < nx-1 && moldGrid[i + 1] === 0) { sumP += p[i + 1]; nCount++; }
                                else nCount++; // Wall neighbor contributes p=0
                                
                                if (x > 0 && moldGrid[i - 1] === 0) { sumP += p[i - 1]; nCount++; }
                                else nCount++;
                                
                                if (y < ny-1 && moldGrid[i + nx] === 0) { sumP += p[i + nx]; nCount++; }
                                else nCount++;
                                
                                if (y > 0 && moldGrid[i - nx] === 0) { sumP += p[i - nx]; nCount++; }
                                else nCount++;
                                
                                if (z < nz-1 && moldGrid[i + nxny] === 0) { sumP += p[i + nxny]; nCount++; }
                                else nCount++;
                                
                                if (z > 0 && moldGrid[i - nxny] === 0) { sumP += p[i - nxny]; nCount++; }
                                else nCount++;
                                
                                if (nCount > 0) {
                                    p[i] = (div[i] + sumP) / nCount;
                                }
                            } else {
                                p[i] = 0;
                            }
                        }
                        i++;
                    }
                }
            }
        }
    }

    // ── Velocity Correction ──
    idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (moldGrid[idx] === 0 && fill[idx] > fillThreshold && 
                    sFraction[idx] < 0.99 && isInlet[idx] === 0) {
                    
                    // Pressure gradient (using same values as neighbors, p=0 at walls/empty)
                    const pR = (x < nx-1 && moldGrid[idx + 1] === 0)    ? p[idx + 1]    : p[idx];
                    const pL = (x > 0    && moldGrid[idx - 1] === 0)    ? p[idx - 1]    : p[idx];
                    const pU = (y < ny-1 && moldGrid[idx + nx] === 0)   ? p[idx + nx]   : p[idx];
                    const pD = (y > 0    && moldGrid[idx - nx] === 0)   ? p[idx - nx]   : p[idx];
                    const pF = (z < nz-1 && moldGrid[idx + nxny] === 0) ? p[idx + nxny] : p[idx];
                    const pB = (z > 0    && moldGrid[idx - nxny] === 0) ? p[idx - nxny] : p[idx];
                    
                    const gradScale = 0.5 * invDx;
                    vx[idx] -= (pR - pL) * gradScale;
                    vy[idx] -= (pU - pD) * gradScale;
                    vz[idx] -= (pF - pB) * gradScale;
                }
                idx++;
            }
        }
    }
}

// ─── Heat Transfer ──────────────────────────────────────────

function computeHeatTransfer() {
    if (!mat || !mat.thermalConductivity) return;
    
    const alpha = mat.thermalConductivity / (mat.density * mat.specificHeat);
    const dFactor = Math.min(alpha * configDt / (voxelSize * voxelSize), 0.15);

    TNew.set(T);

    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {
                    if (fill[idx] > 0.01 || moldGrid[idx] === 1) {
                        const laplacian = T[idx - 1] + T[idx + 1] +
                                          T[idx - nx] + T[idx + nx] +
                                          T[idx - nxny] + T[idx + nxny] -
                                          6.0 * T[idx];
                        TNew[idx] = T[idx] + dFactor * laplacian;
                    }
                }
                idx++;
            }
        }
    }
    
    const tmp = T; T = TNew; TNew = tmp;
}

// ─── Solidification (with Latent Heat) ─────────────────────

function computeSolidification() {
    if (!mat) return;
    
    const solidusTemp = mat.solidusTemp;
    const liquidusTemp = mat.liquidusTemp;
    const range = liquidusTemp - solidusTemp;
    if (range <= 0) return;
    const invRange = 1.0 / range;
    
    // Latent heat: energy released during solidification absorbs cooling
    // dT_latent = L * dfs / Cp  (fs = solid fraction change)
    const latentHeat = mat.latentHeat || 0;
    const specificHeat = mat.specificHeat || 1;

    for (let c = 0; c < cavityCount; c++) {
        const i = cavityIndices[c];
        if (fill[i] > 0.1) {
            const oldSFrac = sFraction[i];
            
            if (T[i] <= solidusTemp) {
                sFraction[i] = 1.0;
                vx[i] = 0; vy[i] = 0; vz[i] = 0;
            } else if (T[i] < liquidusTemp) {
                sFraction[i] = 1.0 - ((T[i] - solidusTemp) * invRange);
                const damp = 1.0 - sFraction[i];
                vx[i] *= damp;
                vy[i] *= damp;
                vz[i] *= damp;
            } else {
                sFraction[i] = 0.0;
            }
            
            // Apply latent heat release: solidification releases energy,
            // slowing down cooling in the mushy zone
            if (latentHeat > 0) {
                const dSFrac = sFraction[i] - oldSFrac;
                if (dSFrac > 0) {
                    // Energy released = L * dfs, temperature rise = L * dfs / Cp
                    T[i] += (latentHeat * dSFrac) / specificHeat;
                }
            }
        }
    }
}

// ─── Visualization Data ─────────────────────────────────────

function sendVisualData() {
    let activeCount = 0;
    for (let c = 0; c < cavityCount; c++) {
        const i = cavityIndices[c];
        if (fill[i] > 0.02 || sFraction[i] > 0) activeCount++;
    }

    const buf = new Float32Array(activeCount * 8);
    let ptr = 0;
    for (let c = 0; c < cavityCount; c++) {
        const i = cavityIndices[c];
        if (fill[i] > 0.02 || sFraction[i] > 0) {
            buf[ptr++] = i;
            buf[ptr++] = fill[i];
            buf[ptr++] = T[i];
            buf[ptr++] = vx[i];
            buf[ptr++] = vy[i];
            buf[ptr++] = vz[i];
            buf[ptr++] = p[i];
            buf[ptr++] = sFraction[i];
        }
    }

    self.postMessage({
        type: 'RENDER_DATA',
        time: stepCount * configDt,
        buffer: buf
    }, [buf.buffer]);
}
