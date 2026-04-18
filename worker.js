/**
 * worker.js
 * High-Pressure Physics Simulation Engine for CastFlow
 */

let nx, ny, nz;
let totalCells;
let voxelSize = 0.01;
let dt = 0.01;
let configDt = 0.01;

// Simulation Arrays (Collocated grid)
let moldGrid;   // Uint8Array: 1 = mold, 0 = cavity
let openness;   // Uint8Array: 0-6 free neighbors
let isInlet;    // Uint8Array: 1 = inlet
let fill;       // Float32Array: 0 to 1
let fillNew;
let T;          // Float32Array: Temperature
let TNew;
let sFraction;  // Float32Array: Solid fraction (0 to 1)
let p;          // Float32Array: Pressure
let pNew;
let div;        // Float32Array: Divergence
let vx, vy, vz; // Float32Array: Velocity
let vxNew, vyNew, vzNew;

// Properties
let mat;
let moldTemp;
let injectTemp;
let injectPressure;
let solverIters;
let kCompressibility;
let inletIndices = [];
let inletNormals = [];

let running = false;
let stepCount = 0;

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
        inletNormals = new Array(totalCells).fill(null);
        let inlets = msg.inlets;
        for(let i=0; i<inlets.length; i++) {
            let inf = inlets[i];
            inletIndices.push(inf.idx);
            isInlet[inf.idx] = 1;
            moldGrid[inf.idx] = 0; // Ensure physical volume exists
            inletNormals[inf.idx] = { nx: inf.nx, ny: inf.ny, nz: inf.nz };
        }
    }
};

function initSimulation(config) {
    nx = config.nx;
    ny = config.ny;
    nz = config.nz;
    totalCells = nx * ny * nz;
    voxelSize = config.voxelSize;
    dt = config.dt;
    configDt = config.dt;
    
    mat = config.material;
    moldTemp = config.moldTemp;
    injectTemp = config.injectTemp;
    injectPressure = config.injectPressure || 100000;
    solverIters = config.solverIters || 20;
    kCompressibility = config.compressibility || 0.0001;

    // Load geometry
    moldGrid = config.grid;

    // Alloc arrays
    openness = new Uint8Array(totalCells);
    isInlet = new Uint8Array(totalCells);
    fill = new Float32Array(totalCells);
    fillNew = new Float32Array(totalCells);
    T = new Float32Array(totalCells);
    TNew = new Float32Array(totalCells);
    sFraction = new Float32Array(totalCells);
    p = new Float32Array(totalCells);
    pNew = new Float32Array(totalCells);
    div = new Float32Array(totalCells);
    vx = new Float32Array(totalCells);
    vy = new Float32Array(totalCells);
    vz = new Float32Array(totalCells);
    vxNew = new Float32Array(totalCells);
    vyNew = new Float32Array(totalCells);
    vzNew = new Float32Array(totalCells);

    // Initialize conditions
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                let idx = IX(x, y, z);
                if (moldGrid[idx] === 1) {
                    T[idx] = moldTemp;
                } else {
                    T[idx] = moldTemp; // air temperature initially
                    fill[idx] = 0;
                    
                    // Precompute geometry openness
                    let o = 6;
                    if (x===0 || moldGrid[IX(x-1,y,z)] === 1) o--;
                    if (x===nx-1 || moldGrid[IX(x+1,y,z)] === 1) o--;
                    if (y===0 || moldGrid[IX(x,y-1,z)] === 1) o--;
                    if (y===ny-1 || moldGrid[IX(x,y+1,z)] === 1) o--;
                    if (z===0 || moldGrid[IX(x,y,z-1)] === 1) o--;
                    if (z===nz-1 || moldGrid[IX(x,y,z+1)] === 1) o--;
                    openness[idx] = o;
                }
            }
        }
    }
    
    stepCount = 0;
    self.postMessage({ type: 'INIT_DONE' });
}

function simulateLoop() {
    if (!running) return;
    
    stepSimulation();
    
    sendVisualData();
    if (running) {
        setTimeout(simulateLoop, 0); 
    }
}

function stepSimulation() {
    applySources();

    let maxV = 1.0; // Base margin to account for gravity jumps
    for(let i=0; i<totalCells; i++) {
        const vel = Math.sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        if(vel > maxV) maxV = vel;
    }

    // Strict CFL: don't move more than 0.5 voxels per substep
    let safeDt = (voxelSize * 0.5) / maxV;
    safeDt = Math.min(safeDt, 0.002); // absolute physical ceiling

    let substeps = Math.ceil(configDt / safeDt);
    substeps = Math.min(substeps, 100); // Prevent total lockup
    
    dt = configDt / substeps;

    for (let s = 0; s < substeps; s++) {
        if (stepCount > 0) {
            addGravity();
        }
        
        project();
        
        advectVelocity();
        advectScalars();
        
        applySources(); // Re-inforce boundaries after advection
    }
    
    stepCount++;
}

function IX(x, y, z) {
    if (x < 0) x = 0; if (x > nx-1) x = nx-1;
    if (y < 0) y = 0; if (y > ny-1) y = ny-1;
    if (z < 0) z = 0; if (z > nz-1) z = nz-1;
    return x + nx * (y + ny * z);
}

function applySources() {
    let biasStrength = Math.min(5.0, injectPressure * 0.00001); 
    
    for (let i = 0; i < totalCells; i++) {
        if (isInlet[i] === 1) {
            fill[i] = 1.0;
            T[i] = injectTemp;
            sFraction[i] = 0;
            
            let n = inletNormals[i];
            if (n) {
                vx[i] = n.nx * biasStrength;
                vy[i] = n.ny * biasStrength;
                vz[i] = n.nz * biasStrength;
            }
        }
    }
}

function addGravity() {
    const g = -9.81;
    for (let i = 0; i < totalCells; i++) {
        if (fill[i] > 0.05 && sFraction[i] < 0.99 && moldGrid[i] === 0) {
            vy[i] += g * dt;
        }
    }
}

function interpolate(array, cx, cy, cz) {
    let x0 = Math.floor(cx); let x1 = x0 + 1;
    let y0 = Math.floor(cy); let y1 = y0 + 1;
    let z0 = Math.floor(cz); let z1 = z0 + 1;

    let sx1 = cx - x0; let sx0 = 1.0 - sx1;
    let sy1 = cy - y0; let sy0 = 1.0 - sy1;
    let sz1 = cz - z0; let sz0 = 1.0 - sz1;
    
    return sx0 * (
        sy0 * (sz0 * array[IX(x0, y0, z0)] + sz1 * array[IX(x0, y0, z1)]) +
        sy1 * (sz0 * array[IX(x0, y1, z0)] + sz1 * array[IX(x0, y1, z1)])
    ) + sx1 * (
        sy0 * (sz0 * array[IX(x1, y0, z0)] + sz1 * array[IX(x1, y0, z1)]) +
        sy1 * (sz0 * array[IX(x1, y1, z0)] + sz1 * array[IX(x1, y1, z1)])
    );
}

function advectScalars() {
    let idx = 0;
    const invDx = 1.0 / voxelSize;
    
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (moldGrid[idx] === 1 || sFraction[idx] >= 0.99 || isInlet[idx] === 1) {
                    fillNew[idx] = fill[idx];
                    TNew[idx] = T[idx];
                    idx++;
                    continue;
                }
                
                let px = x - dt * vx[idx] * invDx;
                let py = y - dt * vy[idx] * invDx;
                let pz = z - dt * vz[idx] * invDx;
                
                fillNew[idx] = interpolate(fill, px, py, pz);
                TNew[idx] = interpolate(T, px, py, pz);
                
                idx++;
            }
        }
    }
    
    let tmpf = fill; fill = fillNew; fillNew = tmpf;
    let tmpt = T; T = TNew; TNew = tmpt;
}

function advectVelocity() {
    let idx = 0;
    const invDx = 1.0 / voxelSize;
    
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (moldGrid[idx] === 1 || sFraction[idx] >= 0.99 || fill[idx] < 0.05) {
                    vxNew[idx] = vx[idx];
                    vyNew[idx] = vy[idx];
                    vzNew[idx] = vz[idx];
                    idx++;
                    continue;
                }
                
                let px = x - dt * vx[idx] * invDx;
                let py = y - dt * vy[idx] * invDx;
                let pz = z - dt * vz[idx] * invDx;
                
                vxNew[idx] = interpolate(vx, px, py, pz);
                vyNew[idx] = interpolate(vy, px, py, pz);
                vzNew[idx] = interpolate(vz, px, py, pz);
                
                // Dissipation
                vxNew[idx] *= 0.99;
                vyNew[idx] *= 0.99;
                vzNew[idx] *= 0.99;
                
                idx++;
            }
        }
    }
    
    let tmp1 = vx; vx = vxNew; vxNew = tmp1;
    let tmp2 = vy; vy = vyNew; vyNew = tmp2;
    let tmp3 = vz; vz = vzNew; vzNew = tmp3;
}

function computeHeatTransfer() {
    const alphaMetal = mat.thermalConductivity / (mat.density * mat.specificHeat);
    const diffMetal = alphaMetal * dt / (voxelSize * voxelSize) * 1000;
    const dFactor = Math.min(diffMetal, 0.1);

    for(let i=0; i<totalCells; i++) TNew[i] = T[i];

    let idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if(x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
                    let sum = T[IX(x-1,y,z)] + T[IX(x+1,y,z)] + T[IX(x,y-1,z)] + T[IX(x,y+1,z)] + T[IX(x,y,z-1)] + T[IX(x,y,z+1)];
                    TNew[idx] = T[idx] + dFactor * (sum - 6 * T[idx]);
                }
                idx++;
            }
        }
    }
    
    let tmpt = T; T = TNew; TNew = tmpt;
}

function computeSolidification() {
    for (let i = 0; i < totalCells; i++) {
        if (moldGrid[i] === 0 && fill[i] > 0.1) {
            if (T[i] <= mat.solidusTemp) {
                sFraction[i] = 1.0;
                vx[i] = 0; vy[i] = 0; vz[i] = 0;
            } else if (T[i] < mat.liquidusTemp) {
                let range = mat.liquidusTemp - mat.solidusTemp;
                sFraction[i] = 1.0 - ((T[i] - mat.solidusTemp) / range);
                let damp = 1.0 - sFraction[i];
                vx[i] *= damp;
                vy[i] *= damp;
                vz[i] *= damp;
            } else {
                sFraction[i] = 0.0;
            }
        }
    }
}

// Pressure projection for High Injection Modeling
function project() {
    const invDx = 1.0 / voxelSize;
    let idx = 0;
    
    // Divergence Step
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                p[idx] = 0; // initialize
                if (moldGrid[idx] === 1 || fill[idx] < 0.1) {
                    div[idx] = 0;
                } else {
                    let vX1 = (x<nx-1) ? vx[IX(x+1,y,z)] : 0;
                    let vX0 = (x>0)    ? vx[IX(x-1,y,z)] : 0;
                    let vY1 = (y<ny-1) ? vy[IX(x,y+1,z)] : 0;
                    let vY0 = (y>0)    ? vy[IX(x,y-1,z)] : 0;
                    let vZ1 = (z<nz-1) ? vz[IX(x,y,z+1)] : 0;
                    let vZ0 = (z>0)    ? vz[IX(x,y,z-1)] : 0;
                    
                    div[idx] = -0.5 * voxelSize * ((vX1 - vX0) + (vY1 - vY0) + (vZ1 - vZ0));
                    
                    // Boundary zeros
                    if(x===0 || x===nx-1 || moldGrid[IX(x+1,y,z)]===1 || moldGrid[IX(x-1,y,z)]===1) div[idx] = 0;
                }
                idx++;
            }
        }
    }

    // Jacobi Solve
    for (let k = 0; k < solverIters; k++) {
        let i = 0;
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    if (isInlet[i] === 1) {
                        pNew[i] = injectPressure; // Preserve P_in boundary
                    } else if (moldGrid[i] === 0 && fill[i] > 0.1) {
                        let pX1 = (x<nx-1) ? p[IX(x+1,y,z)] : 0;
                        let pX0 = (x>0)    ? p[IX(x-1,y,z)] : 0;
                        let pY1 = (y<ny-1) ? p[IX(x,y+1,z)] : 0;
                        let pY0 = (y>0)    ? p[IX(x,y-1,z)] : 0;
                        let pZ1 = (z<nz-1) ? p[IX(x,y,z+1)] : 0;
                        let pZ0 = (z>0)    ? p[IX(x,y,z-1)] : 0;
                        
                        pNew[i] = (div[i] + pX1 + pX0 + pY1 + pY0 + pZ1 + pZ0) / 6.0;
                    } else {
                        pNew[i] = 0;
                    }
                    i++;
                }
            }
        }
        let t = p; p = pNew; pNew = t;
    }

    // Velocity Subtraction (Flow Update)
    idx = 0;
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                if (moldGrid[idx] === 0 && fill[idx] > 0.1 && sFraction[idx] < 0.99) {
                    let pX1 = (x<nx-1) ? p[IX(x+1,y,z)] : p[idx];
                    let pX0 = (x>0)    ? p[IX(x-1,y,z)] : p[idx];
                    let pY1 = (y<ny-1) ? p[IX(x,y+1,z)] : p[idx];
                    let pY0 = (y>0)    ? p[IX(x,y-1,z)] : p[idx];
                    let pZ1 = (z<nz-1) ? p[IX(x,y,z+1)] : p[idx];
                    let pZ0 = (z>0)    ? p[IX(x,y,z-1)] : p[idx];
                    
                    // Modifiers
                    let flowMod = 1.0;
                    // Geometry adaptive acceleration
                    if (openness[idx] < 6) {
                        flowMod = 1.0 + (6 - openness[idx]) * 0.15; 
                    }
                    
                    // Density compress scale
                    let localRhoScale = 1.0 + kCompressibility * p[idx];
                    
                    // Base grad factor adjusted
                    let scale = (0.5 * invDx * flowMod) / localRhoScale;

                    if(isInlet[idx] === 0) { // Dont override source explicitly here, let neighbors pull
                        vx[idx] -= (pX1 - pX0) * scale;
                        vy[idx] -= (pY1 - pY0) * scale;
                        vz[idx] -= (pZ1 - pZ0) * scale;
                    }
                }
                idx++;
            }
        }
    }
}

// Ensure 8 length per voxel
function sendVisualData() {
    let activeCount = 0;
    for (let i = 0; i < totalCells; i++) {
        if (moldGrid[i] === 0 && (fill[i] > 0.05 || sFraction[i] > 0)) activeCount++;
    }

    const renderBuffer = new Float32Array(activeCount * 8);
    let ptr = 0;
    for (let i = 0; i < totalCells; i++) {
        if (moldGrid[i] === 0 && (fill[i] > 0.05 || sFraction[i] > 0)) {
            renderBuffer[ptr++] = i;
            renderBuffer[ptr++] = fill[i];
            renderBuffer[ptr++] = T[i];
            renderBuffer[ptr++] = vx[i];
            renderBuffer[ptr++] = vy[i];
            renderBuffer[ptr++] = vz[i];
            renderBuffer[ptr++] = p[i];
            renderBuffer[ptr++] = sFraction[i];
        }
    }

    self.postMessage({
        type: 'RENDER_DATA',
        time: stepCount * configDt, // keep UI time simple
        buffer: renderBuffer
    }, [renderBuffer.buffer]);
}
