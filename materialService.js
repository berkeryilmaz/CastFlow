/**
 * materialService.js
 * Handles fetching, normalization, and custom injection of material data.
 */

class MaterialService {
  constructor() {
    this.localJsonPath = './materials.json';
    this.externalApiUrl = 'https://api.example.com/materials'; // Placeholder for specific API
    this.customStorageKey = 'castflow_custom_materials';
    this.materials = [];
  }

  /**
   * Initializes and loads all materials from API/JSON + LocalStorage.
   */
  async loadMaterials() {
    let apiMaterials = [];
    try {
      // Attempt External API fetch
      const res = await fetch(this.externalApiUrl, { method: 'GET', mode: 'cors', signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const rawApiData = await res.json();
        apiMaterials = rawApiData.materials ? rawApiData.materials.map(m => this.normalizeMaterial(m)) : [];
      } else {
        throw new Error("External API responded with non-ok status");
      }
    } catch (e) {
      console.error("Material API Error", e);
      console.warn("External API fetch failed, falling back to local JSON materials:", e.message);
      // Fallback to local JSON
      try {
        const res = await fetch(this.localJsonPath);
        if (res.ok) {
          const rawLocalData = await res.json();
          apiMaterials = rawLocalData.materials.map(m => this.normalizeMaterial(m));
        } else {
          console.error("Local JSON fetch failed too.");
        }
      } catch (e2) {
        console.error("Could not load any predefined materials.", e2);
      }
    }

    // Load from Local Storage
    const customMaterials = this.loadCustomMaterials();
    
    // Merge
    this.materials = [...apiMaterials, ...customMaterials];
    return this.materials;
  }

  /**
   * Fetches a single material by name.
   */
  async fetchMaterial(name) {
    if (this.materials.length === 0) {
      await this.loadMaterials();
    }
    return this.materials.find(m => m.name.toLowerCase() === name.toLowerCase()) || null;
  }

  /**
   * Normalizes structural differences into our accepted schema
   */
  normalizeMaterial(rawData) {
    return {
      name: rawData.name || "Unknown Material",
      density: Number(rawData.density) || 1000,
      viscosity: Number(rawData.viscosity) || 0.001,
      liquidusTemp: Number(rawData.liquidusTemp) || 0,
      solidusTemp: Number(rawData.solidusTemp) || 0,
      thermalConductivity: Number(rawData.thermalConductivity) || 0,
      specificHeat: Number(rawData.specificHeat) || 0,
      latentHeat: Number(rawData.latentHeat) || 0,
      shrinkageFactor: Number(rawData.shrinkageFactor) || 0
    };
  }

  /**
   * Loads custom materials from localStorage
   */
  loadCustomMaterials() {
    try {
      const data = localStorage.getItem(this.customStorageKey);
      if (data) {
        const parsed = JSON.parse(data);
        if (Array.isArray(parsed)) {
          return parsed.map(m => this.normalizeMaterial(m));
        }
      }
    } catch (e) {
      console.warn("Could not parse custom materials from localStorage.", e);
    }
    return [];
  }

  /**
   * Saves a new custom material to localStorage
   */
  saveCustomMaterial(materialObj) {
    const normalized = this.normalizeMaterial(materialObj);
    const customMaterials = this.loadCustomMaterials();
    
    // Check for dupes
    const existingIndex = customMaterials.findIndex(m => m.name.toLowerCase() === normalized.name.toLowerCase());
    if (existingIndex >= 0) {
      customMaterials[existingIndex] = normalized; // overwrite definition
    } else {
      customMaterials.push(normalized);
    }

    localStorage.setItem(this.customStorageKey, JSON.stringify(customMaterials));
    
    // Update active memory
    const memIndex = this.materials.findIndex(m => m.name.toLowerCase() === normalized.name.toLowerCase());
    if(memIndex >= 0) {
       this.materials[memIndex] = normalized;
    } else {
       this.materials.push(normalized);
    }

    return normalized;
  }

  getAllMaterials() {
    return this.materials;
  }
}

export default new MaterialService();
