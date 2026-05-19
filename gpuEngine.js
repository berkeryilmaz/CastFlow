/**
 * gpuEngine.js — WebGPU Compute Engine for CastFlow
 * 
 * Full GPU-accelerated CFD simulation using WebGPU compute shaders.
 * Replicates all CPU worker physics: Navier-Stokes, fill transport,
 * heat transfer, and solidification with latent heat.
 */

const WG_SIZE = 64;
const STEPS_PER_RENDER = 4;

// ─── WGSL Physics Shader ────────────────────────────────────
const PHYSICS_SHADER = `
struct SimParams {
    nx: u32, ny: u32, nz: u32, totalCells: u32,
    dt: f32, voxelSize: f32, injectTemp: f32, moldTemp: f32,
    inletSpeed: f32, inletCount: u32, gravity: f32, thermalDiffFactor: f32,
    latentHeat: f32, specificHeat: f32, solidusTemp: f32, liquidusTemp: f32,
    thermalDiffFactorMold: f32, injectPressure: f32, _pad1: u32, _pad2: u32,
    compressibility: f32, density: f32, showAir: f32, _pad4: u32,
};
@group(0) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(0) var<storage, read_write> gridFlags: array<u32>;
@group(1) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> velocityNew: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> scalars: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> scalarsNew: array<vec4<f32>>;
@group(2) @binding(0) var<storage, read> inletData: array<vec4<f32>>;

fn IX(x: i32, y: i32, z: i32) -> u32 {
    let cx = clamp(x, 0, i32(params.nx)-1);
    let cy = clamp(y, 0, i32(params.ny)-1);
    let cz = clamp(z, 0, i32(params.nz)-1);
    return u32(cx) + params.nx * (u32(cy) + params.ny * u32(cz));
}
fn toXYZ(idx: u32) -> vec3<u32> {
    let nxny = params.nx * params.ny;
    return vec3<u32>(idx % params.nx, (idx % nxny) / params.nx, idx / nxny);
}
fn isMold(idx: u32) -> bool { return (gridFlags[idx] & 1u) != 0u; }
fn isInletF(idx: u32) -> bool { return (gridFlags[idx] & 2u) != 0u; }
fn isOutletF(idx: u32) -> bool { return (gridFlags[idx] & 4u) != 0u; }

fn interpVel(cx: f32, cy: f32, cz: f32) -> vec3<f32> {
    let x0=i32(floor(cx)); let y0=i32(floor(cy)); let z0=i32(floor(cz));
    let sx=cx-f32(x0); let rx=1.0-sx;
    let sy=cy-f32(y0); let ry=1.0-sy;
    let sz=cz-f32(z0); let rz=1.0-sz;
    return rx*(ry*(rz*velocity[IX(x0,y0,z0)].xyz+sz*velocity[IX(x0,y0,z0+1)].xyz)
              +sy*(rz*velocity[IX(x0,y0+1,z0)].xyz+sz*velocity[IX(x0,y0+1,z0+1)].xyz))
          +sx*(ry*(rz*velocity[IX(x0+1,y0,z0)].xyz+sz*velocity[IX(x0+1,y0,z0+1)].xyz)
              +sy*(rz*velocity[IX(x0+1,y0+1,z0)].xyz+sz*velocity[IX(x0+1,y0+1,z0+1)].xyz));
}
fn interpTemp(cx: f32, cy: f32, cz: f32) -> f32 {
    let x0=i32(floor(cx)); let y0=i32(floor(cy)); let z0=i32(floor(cz));
    let sx=cx-f32(x0); let rx=1.0-sx;
    let sy=cy-f32(y0); let ry=1.0-sy;
    let sz=cz-f32(z0); let rz=1.0-sz;
    return rx*(ry*(rz*scalars[IX(x0,y0,z0)].y+sz*scalars[IX(x0,y0,z0+1)].y)
              +sy*(rz*scalars[IX(x0,y0+1,z0)].y+sz*scalars[IX(x0,y0+1,z0+1)].y))
          +sx*(ry*(rz*scalars[IX(x0+1,y0,z0)].y+sz*scalars[IX(x0+1,y0,z0+1)].y)
              +sy*(rz*scalars[IX(x0+1,y0+1,z0)].y+sz*scalars[IX(x0+1,y0+1,z0+1)].y));
}

// ─── Kernels ────────────────────────────────────────────────
@compute @workgroup_size(${WG_SIZE})
fn apply_sources(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params._pad2;
    if (i >= params.inletCount) { return; }
    let d = inletData[i]; let ci = u32(d.x);
    let physicalPressure = abs(velocity[ci].w) * params.density / params.dt;
    let backPressureRatio = max(0.0, 1.0 - physicalPressure / (params.injectPressure + 0.000001));
    let effectiveSpeed = params.inletSpeed * backPressureRatio;
    
    if (effectiveSpeed > 0.01 * params.inletSpeed) {
        scalars[ci] = vec4<f32>(1.0, params.injectTemp, 0.0, scalars[ci].w);
        velocity[ci] = vec4<f32>(d.y*effectiveSpeed, d.z*effectiveSpeed, d.w*effectiveSpeed, velocity[ci].w);
    } else {
        velocity[ci] = vec4<f32>(0.0, 0.0, 0.0, velocity[ci].w);
    }
}

@compute @workgroup_size(${WG_SIZE})
fn add_gravity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells || isMold(idx)) { return; }
    let s = scalars[idx];
    if (s.x > 0.01 && s.z < 0.99) {
        var v = velocity[idx];
        let liquidFrac = 1.0 - s.z;
        v.y += params.gravity * params.dt * (liquidFrac * liquidFrac * liquidFrac);
        velocity[idx] = v;
    }
}

@compute @workgroup_size(${WG_SIZE})
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var vn = velocityNew[idx];
    if (isMold(idx)) {
        vn.w = 0.0; velocityNew[idx] = vn;
        var v = velocity[idx]; v.w = 0.0; velocity[idx] = v;
        return;
    }
    let p = toXYZ(idx); let x=i32(p.x); let y=i32(p.y); let z=i32(p.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    let halfDx = 0.5 * params.voxelSize;
    let vxR = select(0.0, velocity[IX(x+1,y,z)].x, x<nx-1 && !isMold(IX(x+1,y,z)));
    let vxL = select(0.0, velocity[IX(x-1,y,z)].x, x>0 && !isMold(IX(x-1,y,z)));
    let vyU = select(0.0, velocity[IX(x,y+1,z)].y, y<ny-1 && !isMold(IX(x,y+1,z)));
    let vyD = select(0.0, velocity[IX(x,y-1,z)].y, y>0 && !isMold(IX(x,y-1,z)));
    let vzF = select(0.0, velocity[IX(x,y,z+1)].z, z<nz-1 && !isMold(IX(x,y,z+1)));
    let vzB = select(0.0, velocity[IX(x,y,z-1)].z, z>0 && !isMold(IX(x,y,z-1)));
    
    var divVal = -halfDx * ((vxR-vxL) + (vyU-vyD) + (vzF-vzB));
    
    // Add air compression source term for empty/partially-filled cells
    let fillFraction = scalars[idx].x;
    if (fillFraction < 0.99 && !isOutletF(idx)) {
        let airFraction = max(0.005, 1.0 - fillFraction);
        let airP = 101325.0 / airFraction;
        if (airP > 101325.0) {
            let pExcess = (airP - 101325.0) / 101325.0;
            divVal += pExcess * params.compressibility * params.voxelSize;
        }
    }
    
    vn.w = divVal;
    velocityNew[idx] = vn;
}

@compute @workgroup_size(${WG_SIZE})
fn jacobi_a_to_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var sn = scalarsNew[idx];
    if (isMold(idx) || isOutletF(idx)) { sn.w = 0.0; scalarsNew[idx] = sn; return; }
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    var sumP: f32 = 0.0;
    sumP += select(0.0, velocity[IX(x+1,y,z)].w, x<nx-1 && !isMold(IX(x+1,y,z)));
    sumP += select(0.0, velocity[IX(x-1,y,z)].w, x>0 && !isMold(IX(x-1,y,z)));
    sumP += select(0.0, velocity[IX(x,y+1,z)].w, y<ny-1 && !isMold(IX(x,y+1,z)));
    sumP += select(0.0, velocity[IX(x,y-1,z)].w, y>0 && !isMold(IX(x,y-1,z)));
    sumP += select(0.0, velocity[IX(x,y,z+1)].w, z<nz-1 && !isMold(IX(x,y,z+1)));
    sumP += select(0.0, velocity[IX(x,y,z-1)].w, z>0 && !isMold(IX(x,y,z-1)));
    sn.w = (velocityNew[idx].w + sumP) / 6.0;
    scalarsNew[idx] = sn;
}

@compute @workgroup_size(${WG_SIZE})
fn jacobi_b_to_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var v = velocity[idx];
    if (isMold(idx) || isOutletF(idx)) { v.w = 0.0; velocity[idx] = v; return; }
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    var sumP: f32 = 0.0;
    sumP += select(0.0, scalarsNew[IX(x+1,y,z)].w, x<nx-1 && !isMold(IX(x+1,y,z)));
    sumP += select(0.0, scalarsNew[IX(x-1,y,z)].w, x>0 && !isMold(IX(x-1,y,z)));
    sumP += select(0.0, scalarsNew[IX(x,y+1,z)].w, y<ny-1 && !isMold(IX(x,y+1,z)));
    sumP += select(0.0, scalarsNew[IX(x,y-1,z)].w, y>0 && !isMold(IX(x,y-1,z)));
    sumP += select(0.0, scalarsNew[IX(x,y,z+1)].w, z<nz-1 && !isMold(IX(x,y,z+1)));
    sumP += select(0.0, scalarsNew[IX(x,y,z-1)].w, z>0 && !isMold(IX(x,y,z-1)));
    v.w = (velocityNew[idx].w + sumP) / 6.0;
    velocity[idx] = v;
}

@compute @workgroup_size(${WG_SIZE})
fn ensure_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var v = velocity[idx]; v.w = scalarsNew[idx].w; velocity[idx] = v;
}

@compute @workgroup_size(${WG_SIZE})
fn pressure_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    if (isMold(idx) || scalars[idx].z >= 0.99 || isInletF(idx) || isOutletF(idx)) { return; }
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    var v = velocity[idx]; let p0 = v.w;
    let gradScale = 0.5 / params.voxelSize;
    let pR=select(p0, velocity[IX(x+1,y,z)].w, x<nx-1 && !isMold(IX(x+1,y,z)));
    let pL=select(p0, velocity[IX(x-1,y,z)].w, x>0 && !isMold(IX(x-1,y,z)));
    let pU=select(p0, velocity[IX(x,y+1,z)].w, y<ny-1 && !isMold(IX(x,y+1,z)));
    let pD=select(p0, velocity[IX(x,y-1,z)].w, y>0 && !isMold(IX(x,y-1,z)));
    let pF=select(p0, velocity[IX(x,y,z+1)].w, z<nz-1 && !isMold(IX(x,y,z+1)));
    let pB=select(p0, velocity[IX(x,y,z-1)].w, z>0 && !isMold(IX(x,y,z-1)));
    v.x -= (pR-pL)*gradScale; v.y -= (pU-pD)*gradScale; v.z -= (pF-pB)*gradScale;
    
    // Darcy damping after correction
    let fs = scalars[idx].z;
    if (fs > 0.01) {
        let lf = 1.0 - fs;
        let d = lf * lf * lf;
        v.x *= d; v.y *= d; v.z *= d;
    }
    
    velocity[idx] = v;
}

@compute @workgroup_size(${WG_SIZE})
fn extend_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells || isMold(idx)) { return; }
    if (scalars[idx].x >= 0.01 || isInletF(idx)) { return; }
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    var sumV = vec3<f32>(0.0); var cnt: f32 = 0.0; var sumT: f32 = 0.0;
    if (x>0)    { let ni=IX(x-1,y,z); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (x<nx-1) { let ni=IX(x+1,y,z); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (y>0)    { let ni=IX(x,y-1,z); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (y<ny-1) { let ni=IX(x,y+1,z); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (z>0)    { let ni=IX(x,y,z-1); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (z<nz-1) { let ni=IX(x,y,z+1); if (!isMold(ni) && scalars[ni].x>0.01) { sumV+=velocity[ni].xyz; sumT+=scalars[ni].y; cnt+=1.0; } }
    if (cnt > 0.0) { 
        velocity[idx] = vec4<f32>(sumV/cnt, velocity[idx].w); 
        var s = scalars[idx]; s.y = sumT/cnt; scalars[idx] = s;
    }
}

@compute @workgroup_size(${WG_SIZE})
fn advect_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var sn = scalarsNew[idx];
    if (isMold(idx) || isInletF(idx) || scalars[idx].z >= 0.99) { sn.x = scalars[idx].x; scalarsNew[idx] = sn; return; }
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    let dtDx = params.dt / params.voxelSize;
    let myF = scalars[idx].x;
    var nf: f32 = 0.0;
    // X-
    if (x>0) { let ni=IX(x-1,y,z); if (!isMold(ni)) { let vf=0.5*(velocity[idx].x+velocity[ni].x);
        nf += select(myF*vf*dtDx, scalars[ni].x*vf*dtDx, vf>0.0); } }
    // X+
    if (x<nx-1) { let ni=IX(x+1,y,z); if (!isMold(ni)) { let vf=0.5*(velocity[idx].x+velocity[ni].x);
        nf -= select(scalars[ni].x*vf*dtDx, myF*vf*dtDx, vf>0.0); } }
    // Y-
    if (y>0) { let ni=IX(x,y-1,z); if (!isMold(ni)) { let vf=0.5*(velocity[idx].y+velocity[ni].y);
        nf += select(myF*vf*dtDx, scalars[ni].x*vf*dtDx, vf>0.0); } }
    // Y+
    if (y<ny-1) { let ni=IX(x,y+1,z); if (!isMold(ni)) { let vf=0.5*(velocity[idx].y+velocity[ni].y);
        nf -= select(scalars[ni].x*vf*dtDx, myF*vf*dtDx, vf>0.0); } }
    // Z-
    if (z>0) { let ni=IX(x,y,z-1); if (!isMold(ni)) { let vf=0.5*(velocity[idx].z+velocity[ni].z);
        nf += select(myF*vf*dtDx, scalars[ni].x*vf*dtDx, vf>0.0); } }
    // Z+
    if (z<nz-1) { let ni=IX(x,y,z+1); if (!isMold(ni)) { let vf=0.5*(velocity[idx].z+velocity[ni].z);
        nf -= select(scalars[ni].x*vf*dtDx, myF*vf*dtDx, vf>0.0); } }
    var fillVal = clamp(myF + nf, 0.0, 1.0);
    if (isOutletF(idx)) {
        fillVal = min(fillVal, 0.95);
    }
    sn.x = fillVal;
    scalarsNew[idx] = sn;
}

@compute @workgroup_size(${WG_SIZE})
fn advect_velocity_temp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    if (isMold(idx) || isInletF(idx)) {
        velocityNew[idx] = velocity[idx];
        var sn=scalarsNew[idx]; sn.y=scalars[idx].y; scalarsNew[idx]=sn; return;
    }
    if (scalars[idx].z >= 0.99) {
        velocityNew[idx] = vec4<f32>(0.0, 0.0, 0.0, velocity[idx].w);
        var sn=scalarsNew[idx]; sn.y=scalars[idx].y; scalarsNew[idx]=sn; return;
    }
    if (scalars[idx].x > 0.01) {
        let pos = toXYZ(idx);
        let dtInvDx = params.dt / params.voxelSize;
        let px = f32(pos.x) - dtInvDx * velocity[idx].x;
        let py = f32(pos.y) - dtInvDx * velocity[idx].y;
        let pz = f32(pos.z) - dtInvDx * velocity[idx].z;
        
        let fs = scalars[idx].z;
        let lf = 1.0 - fs;
        let darcyDamp = lf * lf * lf;
        let visDamp = 0.998 * darcyDamp;
        
        velocityNew[idx] = vec4<f32>(interpVel(px,py,pz) * visDamp, velocity[idx].w);
        var sn=scalarsNew[idx]; sn.y=interpTemp(px,py,pz); scalarsNew[idx]=sn;
    } else {
        velocityNew[idx] = vec4<f32>(0.0, 0.0, 0.0, velocity[idx].w);
        var sn=scalarsNew[idx]; sn.y=scalars[idx].y; scalarsNew[idx]=sn;
    }
}

@compute @workgroup_size(${WG_SIZE})
fn swap_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var s = scalars[idx]; s.x = scalarsNew[idx].x; scalars[idx] = s;
}

@compute @workgroup_size(${WG_SIZE})
fn swap_velocity_temp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var v = velocity[idx]; let vn = velocityNew[idx];
    v.x = vn.x; v.y = vn.y; v.z = vn.z; velocity[idx] = v;
    var s = scalars[idx]; s.y = scalarsNew[idx].y; scalars[idx] = s;
}

@compute @workgroup_size(${WG_SIZE})
fn heat_transfer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var sn = scalarsNew[idx]; let s = scalars[idx];
    let pos = toXYZ(idx); let x=i32(pos.x); let y=i32(pos.y); let z=i32(pos.z);
    let nx=i32(params.nx); let ny=i32(params.ny); let nz=i32(params.nz);
    
    // Mold cells: conjugate heat transfer
    if (isMold(idx)) {
        // Grid boundary: Dirichlet BC
        if (x==0 || x==nx-1 || y==0 || y==ny-1 || z==0 || z==nz-1) {
            sn.y = params.moldTemp; scalarsNew[idx] = sn; return;
        }
        // Interior mold: diffuse with mold properties
        let lap = scalars[IX(x-1,y,z)].y + scalars[IX(x+1,y,z)].y
                + scalars[IX(x,y-1,z)].y + scalars[IX(x,y+1,z)].y
                + scalars[IX(x,y,z-1)].y + scalars[IX(x,y,z+1)].y - 6.0*s.y;
        sn.y = s.y + params.thermalDiffFactorMold * lap;
        scalarsNew[idx] = sn; return;
    }
    
    // Casting fluid cells: diffuse with casting properties
    if (x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1) {
        if (s.x > 0.01) {
            let lap = scalars[IX(x-1,y,z)].y + scalars[IX(x+1,y,z)].y
                    + scalars[IX(x,y-1,z)].y + scalars[IX(x,y+1,z)].y
                    + scalars[IX(x,y,z-1)].y + scalars[IX(x,y,z+1)].y - 6.0*s.y;
            let range = params.liquidusTemp - params.solidusTemp;
            var apparentC = 1.0;
            if (range > 0.0 && s.y >= params.solidusTemp && s.y <= params.liquidusTemp) {
                apparentC = 1.0 + params.latentHeat / (params.specificHeat * range + 0.0001);
            }
            sn.y = s.y + (params.thermalDiffFactor / apparentC) * lap;
            scalarsNew[idx] = sn; return;
        }
    }
    sn.y = s.y; scalarsNew[idx] = sn;
}

@compute @workgroup_size(${WG_SIZE})
fn swap_temp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    var s = scalars[idx]; s.y = scalarsNew[idx].y; scalars[idx] = s;
}

@compute @workgroup_size(${WG_SIZE})
fn solidification(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells || isMold(idx)) { return; }
    var s = scalars[idx];
    if (s.x <= 0.1) { return; }
    let range = params.liquidusTemp - params.solidusTemp;
    if (range <= 0.0) { return; }
    let oldSF = s.z;
    if (s.y <= params.solidusTemp) {
        s.z = 1.0; velocity[idx] = vec4<f32>(0.0, 0.0, 0.0, velocity[idx].w);
    } else if (s.y < params.liquidusTemp) {
        s.z = 1.0 - ((s.y - params.solidusTemp) / range);
        let damp = 1.0 - s.z;
        var v = velocity[idx]; v.x *= damp; v.y *= damp; v.z *= damp; velocity[idx] = v;
    } else { s.z = 0.0; }
    scalars[idx] = s;
}
`;

// ─── WGSL Extraction Shader ────────────────────────────────
const EXTRACT_SHADER = `
struct SimParams {
    nx: u32, ny: u32, nz: u32, totalCells: u32,
    dt: f32, voxelSize: f32, injectTemp: f32, moldTemp: f32,
    inletSpeed: f32, inletCount: u32, gravity: f32, thermalDiffFactor: f32,
    latentHeat: f32, specificHeat: f32, solidusTemp: f32, liquidusTemp: f32,
    thermalDiffFactorMold: f32, injectPressure: f32, _pad1: u32, _pad2: u32,
    compressibility: f32, density: f32, showAir: f32, _pad4: u32,
};
@group(0) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(0) var<storage, read> gridFlags: array<u32>;
@group(1) @binding(1) var<storage, read> velocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> scalars: array<vec4<f32>>;
@group(2) @binding(0) var<storage, read_write> output: array<f32>;
@group(2) @binding(1) var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(1)
fn reset_counter() { atomicStore(&counter[0], 0u); }

@compute @workgroup_size(${WG_SIZE})
fn extract_visual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params._pad1;
    if (idx >= params.totalCells) { return; }
    if ((gridFlags[idx] & 1u) != 0u) { return; }
    
    let s = scalars[idx]; let v = velocity[idx];
    let hasFluid = s.x > 0.02 || s.z > 0.0;
    
    if (hasFluid || params.showAir > 0.5) {
        let pos = atomicAdd(&counter[0], 1u);
        let b = pos * 8u;
        output[b] = f32(idx);
        output[b+1u] = select(-1.0, s.x, hasFluid);
        output[b+2u] = s.y;
        output[b+3u] = v.x;
        output[b+4u] = v.y;
        output[b+5u] = v.z;
        
        // Convert solver pressure to physical pressure in Pascals for visual rendering
        let physicalPressure = abs(v.w) * params.density / params.dt;
        output[b+6u] = physicalPressure;
        
        output[b+7u] = s.z;
    }
}
`;

// ─── GPU Simulation Engine ──────────────────────────────────
export class GPUSimEngine {
    constructor() {
        this.device = null;
        this.stepCount = 0;
    }

    static async isSupported() {
        if (!navigator.gpu) return false;
        try {
            const adapter = await navigator.gpu.requestAdapter();
            return !!adapter;
        } catch { return false; }
    }

    static async create() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('No GPU adapter found');
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension
            }
        });
        const engine = new GPUSimEngine();
        engine.device = device;
        return engine;
    }

    async configure(config) {
        const d = this.device;
        this.nx = config.nx; this.ny = config.ny; this.nz = config.nz;
        this.totalCells = config.nx * config.ny * config.nz;
        this.dt = config.dt;
        this.voxelSize = config.voxelSize;
        this.moldTemp = config.moldTemp;
        this.injectTemp = config.injectTemp;
        this.solverIters = config.solverIters || 20;
        this.mat = config.material;
        this.inletCount = 0;
        // Check config object safely
        this.injectPressure = config.injectPressure || 100000;
        this.inletSpeed = Math.min(10.0, Math.sqrt(2.0 * this.injectPressure / this.mat.density));

        // Physics and air parameters
        this.compressibility = config.compressibility || 0.0001;
        this.density = this.mat.density;
        this.showAir = config.showAir ? 1.0 : 0.0;

        // Thermal diffusion factors
        const alpha = this.mat.thermalConductivity / (this.mat.density * this.mat.specificHeat);
        this.thermalDiffFactor = Math.min(alpha * this.dt / (this.voxelSize * this.voxelSize), 0.15);
        
        const moldMat = config.moldMaterial || { thermalConductivity: 24.6, density: 7800, specificHeat: 460 };
        const alphaMold = moldMat.thermalConductivity / (moldMat.density * moldMat.specificHeat);
        this.thermalDiffFactorMold = Math.min(alphaMold * this.dt / (this.voxelSize * this.voxelSize), 0.15);

        // Count cavity cells for output sizing
        let cavityCount = 0;
        for (let i = 0; i < this.totalCells; i++) { if (config.grid[i] === 0) cavityCount++; }
        this.maxOutput = Math.max(cavityCount, 1024);

        const tc = this.totalCells;

        // ── Create Buffers ──
        this.uniformsBuffer = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.gridFlagsBuffer = d.createBuffer({ size: tc * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.velocityBuffer = d.createBuffer({ size: tc * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.velocityNewBuffer = d.createBuffer({ size: tc * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.scalarsBuffer = d.createBuffer({ size: tc * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.scalarsNewBuffer = d.createBuffer({ size: tc * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.inletDataBuffer = d.createBuffer({ size: Math.max(16, 131072 * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.outputBuffer = d.createBuffer({ size: this.maxOutput * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.counterBuffer = d.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.outputStaging = d.createBuffer({ size: this.maxOutput * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.counterStaging = d.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        // ── Clean zero-initialization of buffers to prevent GPU garbage values ──
        const zeroInit = new Float32Array(tc * 4);
        d.queue.writeBuffer(this.velocityBuffer, 0, zeroInit);
        d.queue.writeBuffer(this.velocityNewBuffer, 0, zeroInit);
        d.queue.writeBuffer(this.scalarsNewBuffer, 0, zeroInit);

        // ── Init grid flags ──
        const gridFlagsData = new Uint32Array(tc);
        for (let i = 0; i < tc; i++) gridFlagsData[i] = config.grid[i]; // bit 0 = mold
        d.queue.writeBuffer(this.gridFlagsBuffer, 0, gridFlagsData);
        this.moldGridCPU = config.grid;

        // ── Init scalars (temperature = moldTemp) ──
        const scalarsInit = new Float32Array(tc * 4);
        for (let i = 0; i < tc; i++) scalarsInit[i * 4 + 1] = this.moldTemp;
        d.queue.writeBuffer(this.scalarsBuffer, 0, scalarsInit);

        this.gridWorkgroups = Math.ceil(tc / WG_SIZE);

        // ── Write uniforms ──
        this._writeUniforms();

        // ── Create Pipelines ──
        this._createPipelines();
    }

    _writeUniforms() {
        const buf = new ArrayBuffer(96);
        const v = new DataView(buf);
        v.setUint32(0, this.nx, true);
        v.setUint32(4, this.ny, true);
        v.setUint32(8, this.nz, true);
        v.setUint32(12, this.totalCells, true);
        v.setFloat32(16, this.dt, true);
        v.setFloat32(20, this.voxelSize, true);
        v.setFloat32(24, this.injectTemp, true);
        v.setFloat32(28, this.moldTemp, true);
        v.setFloat32(32, this.inletSpeed, true);
        v.setUint32(36, this.inletCount, true);
        v.setFloat32(40, -9.81, true);
        v.setFloat32(44, this.thermalDiffFactor, true);
        v.setFloat32(48, this.mat.latentHeat || 0, true);
        v.setFloat32(52, this.mat.specificHeat || 1, true);
        v.setFloat32(56, this.mat.solidusTemp, true);
        v.setFloat32(60, this.mat.liquidusTemp, true);
        v.setFloat32(64, this.thermalDiffFactorMold, true);
        v.setFloat32(68, this.injectPressure, true);
        
        const pad1 = Math.min(this.gridWorkgroups || 0, 65535) * WG_SIZE;
        const pad2 = Math.min(Math.max(1, Math.ceil(this.inletCount / WG_SIZE)), 65535) * WG_SIZE;
        v.setUint32(72, pad1, true);
        v.setUint32(76, pad2, true);
        
        v.setFloat32(80, this.compressibility, true);
        v.setFloat32(84, this.density, true);
        v.setFloat32(88, this.showAir, true);
        v.setUint32(92, 0, true); // _pad4
        
        this.device.queue.writeBuffer(this.uniformsBuffer, 0, buf);
    }

    setShowAir(value) {
        this.showAir = value ? 1.0 : 0.0;
        if (this.device && this.uniformsBuffer) {
            this._writeUniforms();
        }
    }

    _createPipelines() {
        const d = this.device;

        // ── Bind Group Layouts ──
        const uniformBGL = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
        ]});
        const physicsBGL = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]});
        const inletBGL = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
        ]});
        const extractReadBGL = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        ]});
        const extractWriteBGL = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]});

        const physicsLayout = d.createPipelineLayout({ bindGroupLayouts: [uniformBGL, physicsBGL, inletBGL] });
        const extractLayout = d.createPipelineLayout({ bindGroupLayouts: [uniformBGL, extractReadBGL, extractWriteBGL] });

        // ── Bind Groups ──
        this.uniformsBG = d.createBindGroup({ layout: uniformBGL, entries: [
            { binding: 0, resource: { buffer: this.uniformsBuffer } }
        ]});
        this.physicsBG = d.createBindGroup({ layout: physicsBGL, entries: [
            { binding: 0, resource: { buffer: this.gridFlagsBuffer } },
            { binding: 1, resource: { buffer: this.velocityBuffer } },
            { binding: 2, resource: { buffer: this.velocityNewBuffer } },
            { binding: 3, resource: { buffer: this.scalarsBuffer } },
            { binding: 4, resource: { buffer: this.scalarsNewBuffer } },
        ]});
        this.inletBG = d.createBindGroup({ layout: inletBGL, entries: [
            { binding: 0, resource: { buffer: this.inletDataBuffer } }
        ]});
        this.extractReadBG = d.createBindGroup({ layout: extractReadBGL, entries: [
            { binding: 0, resource: { buffer: this.gridFlagsBuffer } },
            { binding: 1, resource: { buffer: this.velocityBuffer } },
            { binding: 2, resource: { buffer: this.scalarsBuffer } },
        ]});
        this.extractWriteBG = d.createBindGroup({ layout: extractWriteBGL, entries: [
            { binding: 0, resource: { buffer: this.outputBuffer } },
            { binding: 1, resource: { buffer: this.counterBuffer } },
        ]});

        // ── Compile Shader Modules ──
        const physicsModule = d.createShaderModule({ code: PHYSICS_SHADER });
        const extractModule = d.createShaderModule({ code: EXTRACT_SHADER });

        // ── Physics Pipelines ──
        const physicsEntries = [
            'apply_sources', 'add_gravity', 'compute_divergence',
            'jacobi_a_to_b', 'jacobi_b_to_a', 'ensure_pressure',
            'pressure_correct', 'extend_velocity', 'advect_fill',
            'advect_velocity_temp', 'swap_fill', 'swap_velocity_temp',
            'heat_transfer', 'swap_temp', 'solidification'
        ];
        this.pp = {};
        for (const entry of physicsEntries) {
            this.pp[entry] = d.createComputePipeline({
                layout: physicsLayout,
                compute: { module: physicsModule, entryPoint: entry }
            });
        }

        // ── Extract Pipelines ──
        this.resetCounterPipeline = d.createComputePipeline({
            layout: extractLayout,
            compute: { module: extractModule, entryPoint: 'reset_counter' }
        });
        this.extractPipeline = d.createComputePipeline({
            layout: extractLayout,
            compute: { module: extractModule, entryPoint: 'extract_visual' }
        });

        this.physicsGroups = [this.uniformsBG, this.physicsBG, this.inletBG];
        this.extractGroups = [this.uniformsBG, this.extractReadBG, this.extractWriteBG];
    }

    setInlets(inlets) {
        // Rebuild gridFlags with inlet bits (preserve outlet bits)
        const gf = new Uint32Array(this.totalCells);
        for (let i = 0; i < this.totalCells; i++) gf[i] = this.moldGridCPU[i];
        for (const inf of inlets) { gf[inf.idx] |= 2; gf[inf.idx] &= ~1; }
        this._cachedGridFlags = gf;
        this.device.queue.writeBuffer(this.gridFlagsBuffer, 0, gf);

        // Upload inlet data
        const arr = new Float32Array(Math.max(inlets.length, 1) * 4);
        for (let i = 0; i < inlets.length; i++) {
            arr[i*4] = inlets[i].idx; arr[i*4+1] = inlets[i].nx;
            arr[i*4+2] = inlets[i].ny; arr[i*4+3] = inlets[i].nz;
        }
        this.device.queue.writeBuffer(this.inletDataBuffer, 0, arr);
        this.inletCount = inlets.length;
        this._writeUniforms();
    }

    setOutlets(outlets) {
        // Add outlet bits (bit 2) to gridFlags
        const gf = this._cachedGridFlags || new Uint32Array(this.totalCells);
        for (const idx of outlets) { gf[idx] |= 4; gf[idx] &= ~1; }
        this.device.queue.writeBuffer(this.gridFlagsBuffer, 0, gf);
    }

    _dispatch(encoder, pipeline, workgroupsArray, groups) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < groups.length; i++) pass.setBindGroup(i, groups[i]);
        pass.dispatchWorkgroups(workgroupsArray[0], workgroupsArray[1], workgroupsArray[2]);
        pass.end();
    }

    step(numSteps) {
        const encoder = this.device.createCommandEncoder();
        const pg = this.physicsGroups;
        
        const tg = this.gridWorkgroups;
        const gw = [Math.min(tg, 65535), Math.ceil(tg / 65535), 1];
        
        const inTg = Math.max(1, Math.ceil(this.inletCount / WG_SIZE));
        const inletWG = [Math.min(inTg, 65535), Math.ceil(inTg / 65535), 1];

        for (let s = 0; s < numSteps; s++) {
            // 1. Sources
            if (this.inletCount > 0) this._dispatch(encoder, this.pp.apply_sources, inletWG, pg);
            // 2. Gravity
            this._dispatch(encoder, this.pp.add_gravity, gw, pg);
            // 3. Pressure solve
            this._dispatch(encoder, this.pp.compute_divergence, gw, pg);
            for (let j = 0; j < this.solverIters; j++) {
                this._dispatch(encoder, j % 2 === 0 ? this.pp.jacobi_a_to_b : this.pp.jacobi_b_to_a, gw, pg);
            }
            if (this.solverIters % 2 === 1) {
                this._dispatch(encoder, this.pp.ensure_pressure, gw, pg);
            }
            this._dispatch(encoder, this.pp.pressure_correct, gw, pg);
            // 4. Extend velocity
            this._dispatch(encoder, this.pp.extend_velocity, gw, pg);
            // 5. Fill advection
            this._dispatch(encoder, this.pp.advect_fill, gw, pg);
            this._dispatch(encoder, this.pp.swap_fill, gw, pg);
            // 6. Velocity + temperature advection
            this._dispatch(encoder, this.pp.advect_velocity_temp, gw, pg);
            this._dispatch(encoder, this.pp.swap_velocity_temp, gw, pg);
            // 7. Re-enforce sources
            if (this.inletCount > 0) this._dispatch(encoder, this.pp.apply_sources, inletWG, pg);
            
            // 8. Heat transfer
            this._dispatch(encoder, this.pp.heat_transfer, gw, pg);
            this._dispatch(encoder, this.pp.swap_temp, gw, pg);
            // 9. Solidification
            this._dispatch(encoder, this.pp.solidification, gw, pg);
        }

        // 10. Extract visual data
        const eg = this.extractGroups;
        const extractWG = [Math.min(tg, 65535), Math.ceil(tg / 65535), 1];
        this._dispatch(encoder, this.resetCounterPipeline, [1, 1, 1], eg);
        this._dispatch(encoder, this.extractPipeline, extractWG, eg);

        // Copy to staging
        const outBytes = Math.min(this.maxOutput * 32, this.outputBuffer.size);
        encoder.copyBufferToBuffer(this.outputBuffer, 0, this.outputStaging, 0, outBytes);
        encoder.copyBufferToBuffer(this.counterBuffer, 0, this.counterStaging, 0, 4);

        this.device.queue.submit([encoder.finish()]);
        this.stepCount += numSteps;
    }

    async readVisualData() {
        // Read counter
        await this.counterStaging.mapAsync(GPUMapMode.READ, 0, 4);
        const count = new Uint32Array(this.counterStaging.getMappedRange(0, 4))[0];
        this.counterStaging.unmap();

        if (count === 0) {
            return { buffer: new Float32Array(0), time: this.stepCount * this.dt };
        }

        const byteSize = Math.min(count * 32, this.outputStaging.size);
        await this.outputStaging.mapAsync(GPUMapMode.READ, 0, byteSize);
        const mapped = new Float32Array(this.outputStaging.getMappedRange(0, byteSize));
        const result = new Float32Array(mapped); // copy before unmap
        this.outputStaging.unmap();

        return { buffer: result, time: this.stepCount * this.dt };
    }

    destroy() {
        const buffers = [
            this.uniformsBuffer, this.gridFlagsBuffer, this.velocityBuffer,
            this.velocityNewBuffer, this.scalarsBuffer, this.scalarsNewBuffer,
            this.inletDataBuffer, this.outputBuffer, this.counterBuffer,
            this.outputStaging, this.counterStaging
        ];
        for (const b of buffers) { if (b) b.destroy(); }
    }
}
