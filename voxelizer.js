import * as THREE from 'three';

/**
 * Voxelizer
 * Converts a Three.js Mesh into a 3D binary grid of mold (1) and empty cavity (0).
 */
export class Voxelizer {
    
    /**
     * Voxelize the mesh
     * @param {THREE.Mesh} mesh - The mesh to voxelize
     * @param {number} voxelSize - World units per voxel
     * @param {number} paddingVoxels - Extra bounding box extension
     * @param {Function} progressCallback - function(progress) 0 to 1
     * @returns {Object} Data containing dimensions, total voxels, and grid
     */
    static async voxelize(mesh, voxelSize, paddingVoxels, progressCallback) {
        return new Promise((resolve) => {
            // Guarantee geometry bounds
            mesh.geometry.computeBoundingBox();
            const bbox = mesh.geometry.boundingBox;
            
            // Apply parametric padding bounding extent
            const padding = voxelSize * paddingVoxels; 
            bbox.min.subScalar(padding);
            bbox.max.addScalar(padding);

            const size = new THREE.Vector3();
            bbox.getSize(size);

            const nx = Math.ceil(size.x / voxelSize);
            const ny = Math.ceil(size.y / voxelSize);
            const nz = Math.ceil(size.z / voxelSize);

            const gridLength = nx * ny * nz;
            
            // 0 = cavity, 1 = mold. Fill with 1 by default (solid mold).
            const grid = new Uint8Array(gridLength);
            grid.fill(1);

            // Setup raycaster
            const raycaster = new THREE.Raycaster();
            const rayDir = new THREE.Vector3(0, 0, 1);
            
            // Small offset to avoid edge cases where ray aligns perfectly with triangle edges
            const rayOffset = new THREE.Vector2(voxelSize * 0.1337, voxelSize * 0.2337);

            // Process asynchronously in batches so we don't freeze the main thread UI completely
            let currentX = 0;

            const processSlice = () => {
                const startTime = performance.now();

                // Process rays for column X
                while(currentX < nx) {
                    const xWordLocal = currentX * voxelSize;
                    const xWorld = bbox.min.x + xWordLocal + (voxelSize/2) + rayOffset.x;

                    for (let y = 0; y < ny; y++) {
                        const yWorldLocal = y * voxelSize;
                        const yWorld = bbox.min.y + yWorldLocal + (voxelSize/2) + rayOffset.y;

                        // Raycast from below the bounding box upwards along Z
                        const rayStart = new THREE.Vector3(xWorld, yWorld, bbox.min.z - voxelSize);
                        raycaster.set(rayStart, rayDir);

                        const intersects = raycaster.intersectObject(mesh, false);
                        
                        // Sort intersections by distance
                        intersects.sort((a, b) => a.distance - b.distance);

                        // Remove coincident triangle hits (common in non-manifold STLs)
                        let uniqueIntersects = [];
                        for(let k=0; k<intersects.length; k++) {
                            if(k === 0 || (intersects[k].distance - uniqueIntersects[uniqueIntersects.length-1].distance) > voxelSize * 0.1) {
                                uniqueIntersects.push(intersects[k]);
                            }
                        }

                        let inside = false;
                        let nextIntersectIndex = 0;

                        // Scan along Z
                        for (let z = 0; z < nz; z++) {
                            const zWorldLocal = z * voxelSize;
                            const zWorld = bbox.min.z + zWorldLocal + (voxelSize/2);

                            // Toggle "inside" state when we pass an intersection
                            while (nextIntersectIndex < uniqueIntersects.length && (bbox.min.z - voxelSize + uniqueIntersects[nextIntersectIndex].distance) < zWorld) {
                                inside = !inside;
                                nextIntersectIndex++;
                            }

                            // If we are inside the mesh (cavity), set to 0. Else (mold), keep 1.
                            if (inside) {
                                const idx = currentX + nx * (y + ny * z);
                                grid[idx] = 0; 
                            }
                        }
                    }

                    currentX++;

                    // Yield back to event loop if we spent more than 16ms to keep UI somewhat responsive
                    if (performance.now() - startTime > 16) {
                        break;
                    }
                }

                const progress = currentX / nx;
                if (progressCallback) progressCallback(progress);

                if (currentX < nx) {
                    requestAnimationFrame(processSlice);
                } else {
                    resolve({
                        nx, ny, nz,
                        voxelSize,
                        bboxMin: { x: bbox.min.x, y: bbox.min.y, z: bbox.min.z },
                        bboxMax: { x: bbox.max.x, y: bbox.max.y, z: bbox.max.z },
                        grid
                    });
                }
            };

            processSlice();
        });
    }
}
