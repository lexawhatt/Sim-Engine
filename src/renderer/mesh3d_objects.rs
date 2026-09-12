//! Stable object addressing and bounded scene lifetime.

use super::*;

/// Stable scene handle for a retained 3D object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Object3dId {
    pub(super) scene_id: u64,
    pub(super) object_id: u64,
    slot: usize,
}

impl Object3dId {
    /// Returns the stable scene-local numeric value for diagnostics.
    ///
    /// Equality and hashing also include private scene provenance. Store the
    /// complete handle, rather than only this number, in host object maps.
    pub const fn get(self) -> u64 {
        self.object_id
    }
}

/// One retained mesh reference plus its independent model transform and style.
#[derive(Clone)]
pub struct Mesh3dInstance {
    pub(super) id: Object3dId,
    pub(super) mesh: RetainedMesh3d,
    pub(super) transform: Transform3d,
    pub(super) style: MeshStyle3d,
    pub(super) visible: bool,
}

impl Mesh3dInstance {
    fn new(
        id: Object3dId,
        mesh: &RetainedMesh3d,
        transform: Transform3d,
        style: MeshStyle3d,
    ) -> Self {
        Self {
            id,
            mesh: mesh.clone(),
            transform,
            style,
            visible: true,
        }
    }

    /// Returns the stable scene object identifier.
    pub const fn id(&self) -> Object3dId {
        self.id
    }

    /// Returns the retained GPU mesh reference.
    pub const fn mesh(&self) -> &RetainedMesh3d {
        &self.mesh
    }

    /// Returns this object's model-to-world transform.
    pub const fn transform(&self) -> Transform3d {
        self.transform
    }

    /// Returns its extensible surface/edge material bundle.
    pub const fn style(&self) -> MeshStyle3d {
        self.style
    }

    /// Returns whether this object participates in drawing and render counts.
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Returns explicit edge presentation when enabled for this object.
    pub const fn wireframe(&self) -> Option<WireframeStyle3d> {
        self.style.wireframe_style()
    }
}

/// Ready 3D visual state rendered with one camera and depth attachment.
pub struct Scene3d {
    pub(super) scene_id: u64,
    pub(super) background: Color,
    pub(super) instances: Vec<Mesh3dInstance>,
    pub(super) next_object_id: u64,
    pub(super) renderer_identity: Option<Arc<()>>,
    slots: Vec<ObjectSlot>,
    free_slot: Option<usize>,
    resources: Vec<ResourceUsage>,
    budget: Scene3dBudget,
}

#[derive(Clone)]
struct ObjectSlot {
    dense_index: Option<usize>,
    next_free: Option<usize>,
}

#[derive(Clone, Copy)]
struct ResourceUsage {
    key: (u8, usize),
    references: usize,
    bytes: usize,
}

pub(super) struct DynamicSceneMeshChange {
    index: usize,
    outgoing: [ResourceUsage; 4],
    resources: Option<Vec<ResourceUsage>>,
}

/// Bounds scene-owned slots and distinct retained mesh allocations.
///
/// Object storage includes dense instances, reusable lookup slots and resource
/// accounting. Mesh CPU/GPU bytes are deduplicated by allocation identity. Host
/// references outside this scene and in-flight driver allocations are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene3dBudget {
    max_objects: usize,
    max_storage_bytes: usize,
    max_mesh_cpu_bytes: usize,
    max_mesh_gpu_bytes: usize,
    max_texture_cpu_bytes: usize,
    max_texture_gpu_bytes: usize,
}

impl Scene3dBudget {
    /// Creates nonzero object, storage, recovery-topology and GPU-buffer limits.
    pub fn new(
        max_objects: usize,
        max_storage_bytes: usize,
        max_mesh_cpu_bytes: usize,
        max_mesh_gpu_bytes: usize,
    ) -> Result<Self, Scene3dError> {
        if [
            max_objects,
            max_storage_bytes,
            max_mesh_cpu_bytes,
            max_mesh_gpu_bytes,
        ]
        .contains(&0)
        {
            return Err(Scene3dError::InvalidBudget);
        }
        Ok(Self {
            max_objects,
            max_storage_bytes,
            max_mesh_cpu_bytes,
            max_mesh_gpu_bytes,
            max_texture_cpu_bytes: 64 * 1024 * 1024,
            max_texture_gpu_bytes: 64 * 1024 * 1024,
        })
    }
    /// Maximum simultaneous live objects; removed lookup slots can be reused.
    pub const fn max_objects(self) -> usize {
        self.max_objects
    }
    /// Maximum retained CPU bookkeeping capacity in bytes.
    pub const fn max_storage_bytes(self) -> usize {
        self.max_storage_bytes
    }
    /// Maximum distinct source topology capacity in bytes.
    pub const fn max_mesh_cpu_bytes(self) -> usize {
        self.max_mesh_cpu_bytes
    }
    /// Maximum distinct mesh buffer bytes referenced by the scene.
    pub const fn max_mesh_gpu_bytes(self) -> usize {
        self.max_mesh_gpu_bytes
    }
    /// Sets deduplicated texture recovery/GPU texel limits; zero prohibits
    /// textured resources. Mesh and bookkeeping limits remain independent.
    pub const fn with_texture_limits(mut self, cpu_bytes: usize, gpu_bytes: usize) -> Self {
        self.max_texture_cpu_bytes = cpu_bytes;
        self.max_texture_gpu_bytes = gpu_bytes;
        self
    }
    /// Maximum distinct retained texture-pixel capacity in bytes.
    pub const fn max_texture_cpu_bytes(self) -> usize {
        self.max_texture_cpu_bytes
    }
    /// Maximum distinct nominal GPU texture texel bytes.
    pub const fn max_texture_gpu_bytes(self) -> usize {
        self.max_texture_gpu_bytes
    }
}

impl Default for Scene3dBudget {
    fn default() -> Self {
        Self {
            max_objects: 65_536,
            max_storage_bytes: 64 * 1024 * 1024,
            max_mesh_cpu_bytes: 256 * 1024 * 1024,
            max_mesh_gpu_bytes: 256 * 1024 * 1024,
            max_texture_cpu_bytes: 64 * 1024 * 1024,
            max_texture_gpu_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Bounded scene resource whose limit rejected a mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene3dBudgetResource {
    /// Distinct retained CPU texture-pixel capacities.
    TextureCpuBytes,
    /// Distinct nominal GPU texture texel-storage bytes.
    TextureGpuBytes,
    /// Simultaneously live object count.
    Objects,
    /// Dense instances, lookup slots and allocation-accounting storage.
    StorageBytes,
    /// Distinct retained source topology bytes.
    MeshCpuBytes,
    /// Distinct retained GPU mesh buffer bytes.
    MeshGpuBytes,
}

/// Current scene-owned capacities and unique mesh allocations, excluding drivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene3dStatistics {
    object_count: usize,
    slot_capacity: usize,
    storage_bytes: usize,
    mesh_count: usize,
    mesh_cpu_bytes: usize,
    mesh_gpu_bytes: usize,
    texture_count: usize,
    texture_cpu_bytes: usize,
    texture_gpu_bytes: usize,
}

impl Scene3dStatistics {
    /// Distinct shared textures referenced by live objects, including hidden ones.
    pub const fn texture_count(self) -> usize {
        self.texture_count
    }
    /// Unique retained texture-pixel capacity, excluding fixed binding metadata.
    pub const fn texture_cpu_bytes(self) -> usize {
        self.texture_cpu_bytes
    }
    /// Unique nominal GPU texel storage, excluding driver overhead.
    pub const fn texture_gpu_bytes(self) -> usize {
        self.texture_gpu_bytes
    }
    /// Number of live objects, including hidden ones.
    pub const fn object_count(self) -> usize {
        self.object_count
    }
    /// Retained lookup-slot capacity, reused after removals.
    pub const fn slot_capacity(self) -> usize {
        self.slot_capacity
    }
    /// CPU bookkeeping capacity; does not include source topology.
    pub const fn storage_bytes(self) -> usize {
        self.storage_bytes
    }
    /// Distinct retained GPU meshes, not the number of references.
    pub const fn mesh_count(self) -> usize {
        self.mesh_count
    }
    /// Unique source topology bytes; two GPU uploads of one source count once.
    pub const fn mesh_cpu_bytes(self) -> usize {
        self.mesh_cpu_bytes
    }
    /// Unique GPU buffer bytes; excludes opaque driver/in-flight allocations.
    pub const fn mesh_gpu_bytes(self) -> usize {
        self.mesh_gpu_bytes
    }
}

/// Accounting for an immutable mesh rebinding, including old/new overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scene3dMeshUpdateReport {
    statistics: Scene3dStatistics,
    peak_mesh_cpu_bytes: usize,
    peak_mesh_gpu_bytes: usize,
    peak_texture_cpu_bytes: usize,
    peak_texture_gpu_bytes: usize,
}

impl Scene3dMeshUpdateReport {
    /// Unique old-plus-new texture-pixel capacity during rebinding.
    pub const fn peak_texture_cpu_bytes(self) -> usize {
        self.peak_texture_cpu_bytes
    }
    /// Unique old-plus-new GPU texel bytes during rebinding, not driver retirement.
    pub const fn peak_texture_gpu_bytes(self) -> usize {
        self.peak_texture_gpu_bytes
    }
    /// Accepted scene capacities and unique resource bytes.
    pub const fn statistics(self) -> Scene3dStatistics {
        self.statistics
    }
    /// Unique old-plus-new topology bytes during the rebind.
    pub const fn peak_mesh_cpu_bytes(self) -> usize {
        self.peak_mesh_cpu_bytes
    }
    /// Unique old-plus-new buffer bytes during the rebind, not driver retirement.
    pub const fn peak_mesh_gpu_bytes(self) -> usize {
        self.peak_mesh_gpu_bytes
    }
}

impl Scene3d {
    /// Creates an empty scene with a normalized opaque clear color.
    pub fn new(background: Color) -> Result<Self, Scene3dError> {
        Self::with_budget(background, Scene3dBudget::default())
    }

    /// Creates a lazily allocated scene with explicit live-resource limits.
    pub fn with_budget(background: Color, budget: Scene3dBudget) -> Result<Self, Scene3dError> {
        if !background.is_normalized() || background.alpha() != 1.0 {
            return Err(Scene3dError::InvalidBackground);
        }
        let scene_id = NEXT_SCENE3D_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| Scene3dError::SceneIdExhausted)?;
        Ok(Self {
            scene_id,
            background,
            instances: Vec::new(),
            next_object_id: 0,
            renderer_identity: None,
            slots: Vec::new(),
            free_slot: None,
            resources: Vec::new(),
            budget,
        })
    }

    /// Adds a retained object and returns a stable handle for later updates.
    pub fn try_push(
        &mut self,
        mesh: &RetainedMesh3d,
        transform: Transform3d,
        style: MeshStyle3d,
    ) -> Result<Object3dId, Scene3dError> {
        validate_mesh_style(mesh, style)?;
        validate_scene3d_transform(transform)?;
        self.validate_mesh_owner(mesh)?;
        check_scene_limit(
            Scene3dBudgetResource::Objects,
            self.budget.max_objects,
            self.instances.len().saturating_add(1),
        )?;
        let next_object_id = self
            .next_object_id
            .checked_add(1)
            .ok_or(Scene3dError::ObjectIdExhausted)?;
        let keys = mesh_resources(mesh);
        self.validate_resource_change(&keys, None)?;
        self.reserve_storage(
            1,
            usize::from(self.free_slot.is_none()),
            self.missing_resources(&keys),
        )?;
        let slot = self.free_slot.unwrap_or(self.slots.len());
        let id = Object3dId {
            scene_id: self.scene_id,
            object_id: self.next_object_id,
            slot,
        };
        if let Some(free) = self.free_slot {
            self.free_slot = self.slots[free].next_free;
            self.slots[free] = ObjectSlot {
                dense_index: Some(self.instances.len()),
                next_free: None,
            };
        } else {
            self.slots.push(ObjectSlot {
                dense_index: Some(self.instances.len()),
                next_free: None,
            });
        }
        for resource in keys {
            self.add_resource(resource);
        }
        self.renderer_identity = Some(Arc::clone(&mesh.renderer_identity));
        self.instances
            .push(Mesh3dInstance::new(id, mesh, transform, style));
        self.next_object_id = next_object_id;
        Ok(id)
    }

    /// Returns the normalized finite opaque target clear color.
    pub const fn background(&self) -> Color {
        self.background
    }

    /// Returns objects in host insertion order; depth determines visibility.
    pub fn instances(&self) -> &[Mesh3dInstance] {
        &self.instances
    }

    /// Returns the number of independently transformed objects.
    pub fn object_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns the configured simultaneous-live and retained-capacity limits.
    pub const fn budget(&self) -> Scene3dBudget {
        self.budget
    }

    /// Returns deduplicated allocation accounting without allocating scratch.
    pub fn statistics(&self) -> Scene3dStatistics {
        Scene3dStatistics {
            texture_count: self.resources.iter().filter(|r| r.key.0 == 3).count(),
            texture_cpu_bytes: self
                .resources
                .iter()
                .filter(|r| r.key.0 == 2)
                .map(|r| r.bytes)
                .sum(),
            texture_gpu_bytes: self
                .resources
                .iter()
                .filter(|r| r.key.0 == 3)
                .map(|r| r.bytes)
                .sum(),
            object_count: self.instances.len(),
            slot_capacity: self.slots.capacity(),
            storage_bytes: storage_bytes(
                self.instances.capacity(),
                self.slots.capacity(),
                self.resources.capacity(),
            ),
            mesh_count: self.resources.iter().filter(|r| r.key.0 == 1).count(),
            mesh_cpu_bytes: self
                .resources
                .iter()
                .filter(|r| r.key.0 == 0)
                .map(|r| r.bytes)
                .sum(),
            mesh_gpu_bytes: self
                .resources
                .iter()
                .filter(|r| r.key.0 == 1)
                .map(|r| r.bytes)
                .sum(),
        }
    }

    /// Resolves a live scene-provenance-checked handle in constant time.
    pub fn instance(&self, object_id: Object3dId) -> Result<&Mesh3dInstance, Scene3dError> {
        let index = self.instance_index(object_id)?;
        Ok(&self.instances[index])
    }

    /// Removes an object, preserving survivor IDs and insertion order.
    ///
    /// Removal shifts dense storage in O(N); transform/style/visibility lookups
    /// remain O(1). Lookup slots are reused with never-repeated numeric IDs.
    /// The returned instance owns its retired mesh reference. Dropping it releases
    /// that reference, not necessarily in-flight driver memory.
    pub fn remove(&mut self, object_id: Object3dId) -> Result<Mesh3dInstance, Scene3dError> {
        let index = self.instance_index(object_id)?;
        let removed = self.instances.remove(index);
        for instance in &self.instances[index..] {
            if let Some(dense) = &mut self.slots[instance.id.slot].dense_index {
                *dense -= 1;
            }
        }
        self.slots[object_id.slot] = ObjectSlot {
            dense_index: None,
            next_free: self.free_slot,
        };
        self.free_slot = Some(object_id.slot);
        for resource in mesh_resources(&removed.mesh) {
            self.remove_resource(resource.key);
        }
        Ok(removed)
    }

    /// Rebinds only this object to an immutable mesh after complete validation.
    ///
    /// Shared users of the old mesh are unchanged. Transform, style, visibility
    /// and ID survive. The accepted resource limits apply to final scene state;
    /// replacement overlap is at most old plus incoming mesh and is reported.
    /// Synchronous errors preserve all live visual state. In-flight submissions
    /// retain their old GPU references; no immediate driver retirement is promised.
    pub fn set_mesh(
        &mut self,
        object_id: Object3dId,
        mesh: &RetainedMesh3d,
    ) -> Result<Scene3dMeshUpdateReport, Scene3dError> {
        let index = self.instance_index(object_id)?;
        self.validate_mesh_owner(mesh)?;
        validate_mesh_style(mesh, self.instances[index].style)?;
        let incoming = mesh_resources(mesh);
        let outgoing = mesh_resources(&self.instances[index].mesh);
        let [
            peak_mesh_cpu_bytes,
            peak_mesh_gpu_bytes,
            peak_texture_cpu_bytes,
            peak_texture_gpu_bytes,
        ] = self.validate_resource_change(&incoming, Some(&outgoing))?;
        let removed_keys = outgoing
            .iter()
            .filter(|resource| {
                !incoming.iter().any(|r| r.key == resource.key)
                    && self
                        .resources
                        .binary_search_by_key(&resource.key, |r| r.key)
                        .ok()
                        .is_some_and(|i| self.resources[i].references == 1)
            })
            .count();
        self.reserve_storage(
            0,
            0,
            self.missing_resources(&incoming)
                .saturating_sub(removed_keys),
        )?;
        for resource in outgoing {
            self.remove_resource(resource.key);
        }
        for resource in incoming {
            self.add_resource(resource);
        }
        self.instances[index].mesh = mesh.clone();
        Ok(Scene3dMeshUpdateReport {
            statistics: self.statistics(),
            peak_mesh_cpu_bytes,
            peak_mesh_gpu_bytes,
            peak_texture_cpu_bytes,
            peak_texture_gpu_bytes,
        })
    }

    pub(super) fn prepare_dynamic_mesh_change(
        &self,
        object_id: Object3dId,
        source: &Mesh3d,
        gpu_bytes: usize,
        replace_buffers: bool,
    ) -> Result<DynamicSceneMeshChange, Scene3dError> {
        let index = self.instance_index(object_id)?;
        validate_mesh_source_style(source, self.instances[index].style)?;
        let outgoing = mesh_resources(&self.instances[index].mesh);
        let mut incoming = outgoing;
        incoming[0].key.1 = source.vertices().as_ptr() as usize;
        incoming[0].bytes = source.recovery_memory_bytes();
        if replace_buffers {
            // Real Arc allocation identities cannot be null. This temporary
            // key proves new-bundle accounting before any GPU allocation.
            incoming[1].key.1 = 0;
        }
        incoming[1].bytes = gpu_bytes;
        self.validate_resource_change(&incoming, Some(&outgoing))?;
        let retired = outgoing
            .iter()
            .filter(|resource| {
                !incoming
                    .iter()
                    .any(|candidate| candidate.key == resource.key)
                    && self
                        .resources
                        .binary_search_by_key(&resource.key, |entry| entry.key)
                        .ok()
                        .is_some_and(|index| self.resources[index].references == 1)
            })
            .count();
        let additional = self.missing_resources(&incoming).saturating_sub(retired);
        let mut capacity = planned_capacity(
            &self.resources,
            additional,
            self.budget.max_objects.saturating_mul(4),
        );
        if storage_bytes(self.instances.capacity(), self.slots.capacity(), capacity)
            > self.budget.max_storage_bytes
        {
            capacity = self
                .resources
                .capacity()
                .max(self.resources.len().saturating_add(additional));
        }
        check_scene_limit(
            Scene3dBudgetResource::StorageBytes,
            self.budget.max_storage_bytes,
            storage_bytes(self.instances.capacity(), self.slots.capacity(), capacity),
        )?;
        let resources = replacement_capacity(&self.resources, capacity)?;
        check_scene_limit(
            Scene3dBudgetResource::StorageBytes,
            self.budget.max_storage_bytes,
            storage_bytes(
                self.instances.capacity(),
                self.slots.capacity(),
                resources
                    .as_ref()
                    .map_or(self.resources.capacity(), Vec::capacity),
            ),
        )?;
        Ok(DynamicSceneMeshChange {
            index,
            outgoing,
            resources,
        })
    }

    pub(super) fn commit_dynamic_mesh_change(
        &mut self,
        change: DynamicSceneMeshChange,
        mesh: RetainedMesh3d,
    ) -> Scene3dStatistics {
        if let Some(resources) = change.resources {
            self.resources = resources;
        }
        for resource in change.outgoing {
            self.remove_resource(resource.key);
        }
        for resource in mesh_resources(&mesh) {
            self.add_resource(resource);
        }
        self.instances[change.index].mesh = mesh;
        self.statistics()
    }

    /// Returns objects currently participating in rendering.
    pub fn visible_object_count(&self) -> usize {
        self.instances
            .iter()
            .filter(|instance| instance.visible)
            .count()
    }

    /// Replaces one object's model transform without touching retained topology.
    pub fn set_transform(
        &mut self,
        object_id: Object3dId,
        transform: Transform3d,
    ) -> Result<(), Scene3dError> {
        validate_scene3d_transform(transform)?;
        let instance = self.instance_mut(object_id)?;
        instance.transform = transform;
        Ok(())
    }

    /// Replaces one object's complete extensible visual material bundle.
    pub fn set_style(
        &mut self,
        object_id: Object3dId,
        style: MeshStyle3d,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        validate_mesh_style(&instance.mesh, style)?;
        instance.style = style;
        Ok(())
    }

    /// Shows or hides one object without releasing retained topology.
    pub fn set_visible(
        &mut self,
        object_id: Object3dId,
        visible: bool,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        instance.visible = visible;
        Ok(())
    }

    /// Enables or disables explicit visible/hidden edges for one object.
    pub fn set_wireframe(
        &mut self,
        object_id: Object3dId,
        wireframe: Option<WireframeStyle3d>,
    ) -> Result<(), Scene3dError> {
        let instance = self.instance_mut(object_id)?;
        let surface = instance.style.surface_style();
        let style = match (surface, wireframe) {
            (Some(surface), Some(wireframe)) => {
                MeshStyle3d::surface(surface).with_wireframe(wireframe)
            }
            (Some(surface), None) => MeshStyle3d::surface(surface),
            (None, Some(wireframe)) => MeshStyle3d::wireframe(wireframe),
            (None, None) => return Err(Scene3dError::EmptyStyle),
        };
        validate_mesh_style(&instance.mesh, style)?;
        instance.style = style;
        Ok(())
    }

    fn instance_mut(&mut self, object_id: Object3dId) -> Result<&mut Mesh3dInstance, Scene3dError> {
        let index = self.instance_index(object_id)?;
        Ok(&mut self.instances[index])
    }

    fn instance_index(&self, object_id: Object3dId) -> Result<usize, Scene3dError> {
        let index = (object_id.scene_id == self.scene_id)
            .then_some(())
            .and_then(|()| self.slots.get(object_id.slot))
            .and_then(|slot| slot.dense_index)
            .filter(|&index| {
                self.instances
                    .get(index)
                    .is_some_and(|instance| instance.id == object_id)
            });
        index.ok_or(Scene3dError::ObjectNotFound { object_id })
    }

    fn validate_mesh_owner(&self, mesh: &RetainedMesh3d) -> Result<(), Scene3dError> {
        if mesh
            .material()
            .is_some_and(|material| !material.texture().belongs_to(&mesh.renderer_identity))
        {
            return Err(Scene3dError::RendererMismatch);
        }
        if self
            .renderer_identity
            .as_ref()
            .is_some_and(|identity| !Arc::ptr_eq(identity, &mesh.renderer_identity))
        {
            Err(Scene3dError::RendererMismatch)
        } else {
            Ok(())
        }
    }

    fn missing_resources(&self, incoming: &[ResourceUsage]) -> usize {
        incoming
            .iter()
            .filter(|r| {
                r.bytes > 0
                    && self
                        .resources
                        .binary_search_by_key(&r.key, |r| r.key)
                        .is_err()
            })
            .count()
    }

    fn add_resource(&mut self, resource: ResourceUsage) {
        if resource.bytes == 0 {
            return;
        }
        match self
            .resources
            .binary_search_by_key(&resource.key, |r| r.key)
        {
            Ok(index) => self.resources[index].references += 1,
            Err(index) => self.resources.insert(index, resource),
        }
    }

    fn remove_resource(&mut self, key: (u8, usize)) {
        if let Ok(index) = self.resources.binary_search_by_key(&key, |r| r.key) {
            self.resources[index].references -= 1;
            if self.resources[index].references == 0 {
                self.resources.remove(index);
            }
        }
    }

    fn validate_resource_change(
        &self,
        incoming: &[ResourceUsage],
        outgoing: Option<&[ResourceUsage]>,
    ) -> Result<[usize; 4], Scene3dError> {
        let stats = self.statistics();
        let mut bytes = [
            stats.mesh_cpu_bytes,
            stats.mesh_gpu_bytes,
            stats.texture_cpu_bytes,
            stats.texture_gpu_bytes,
        ];
        for resource in incoming {
            if self
                .resources
                .binary_search_by_key(&resource.key, |r| r.key)
                .is_err()
            {
                bytes[resource.key.0 as usize] =
                    bytes[resource.key.0 as usize].saturating_add(resource.bytes);
            }
        }
        let peak = bytes;
        for resource in outgoing.into_iter().flatten() {
            if !incoming.iter().any(|r| r.key == resource.key)
                && self
                    .resources
                    .binary_search_by_key(&resource.key, |r| r.key)
                    .ok()
                    .is_some_and(|i| self.resources[i].references == 1)
            {
                bytes[resource.key.0 as usize] -= resource.bytes;
            }
        }
        check_scene_limit(
            Scene3dBudgetResource::MeshCpuBytes,
            self.budget.max_mesh_cpu_bytes,
            bytes[0],
        )?;
        check_scene_limit(
            Scene3dBudgetResource::MeshGpuBytes,
            self.budget.max_mesh_gpu_bytes,
            bytes[1],
        )?;
        check_scene_limit(
            Scene3dBudgetResource::TextureCpuBytes,
            self.budget.max_texture_cpu_bytes,
            bytes[2],
        )?;
        check_scene_limit(
            Scene3dBudgetResource::TextureGpuBytes,
            self.budget.max_texture_gpu_bytes,
            bytes[3],
        )?;
        Ok(peak)
    }

    fn reserve_storage(
        &mut self,
        instances: usize,
        slots: usize,
        resources: usize,
    ) -> Result<(), Scene3dError> {
        let mut capacities = [
            planned_capacity(&self.instances, instances, self.budget.max_objects),
            planned_capacity(&self.slots, slots, self.budget.max_objects),
            planned_capacity(
                &self.resources,
                resources,
                self.budget.max_objects.saturating_mul(4),
            ),
        ];
        if storage_bytes(capacities[0], capacities[1], capacities[2])
            > self.budget.max_storage_bytes
        {
            capacities = [
                self.instances
                    .capacity()
                    .max(self.instances.len().saturating_add(instances)),
                self.slots
                    .capacity()
                    .max(self.slots.len().saturating_add(slots)),
                self.resources
                    .capacity()
                    .max(self.resources.len().saturating_add(resources)),
            ];
        }
        let planned_bytes = storage_bytes(capacities[0], capacities[1], capacities[2]);
        check_scene_limit(
            Scene3dBudgetResource::StorageBytes,
            self.budget.max_storage_bytes,
            planned_bytes,
        )?;
        let new_instances = replacement_capacity(&self.instances, capacities[0])?;
        let new_slots = replacement_capacity(&self.slots, capacities[1])?;
        let new_resources = replacement_capacity(&self.resources, capacities[2])?;
        let bytes = storage_bytes(
            new_instances
                .as_ref()
                .map_or(self.instances.capacity(), Vec::capacity),
            new_slots
                .as_ref()
                .map_or(self.slots.capacity(), Vec::capacity),
            new_resources
                .as_ref()
                .map_or(self.resources.capacity(), Vec::capacity),
        );
        check_scene_limit(
            Scene3dBudgetResource::StorageBytes,
            self.budget.max_storage_bytes,
            bytes,
        )?;
        if let Some(values) = new_instances {
            self.instances = values;
        }
        if let Some(values) = new_slots {
            self.slots = values;
        }
        if let Some(values) = new_resources {
            self.resources = values;
        }
        Ok(())
    }

    pub(super) fn rebuild_resource_keys(&mut self) {
        // Recovery preserves topology sharing and distinct mesh count, so the
        // existing table capacity suffices. New GPU allocation identities replace
        // old keys only after the complete recovery candidate has been accepted.
        self.resources.clear();
        for index in 0..self.instances.len() {
            for resource in mesh_resources(&self.instances[index].mesh) {
                self.add_resource(resource);
            }
        }
    }
}

fn mesh_resources(mesh: &RetainedMesh3d) -> [ResourceUsage; 4] {
    [
        ResourceUsage {
            key: (0, mesh.source.vertices().as_ptr() as usize),
            references: 1,
            bytes: mesh.recovery_memory_bytes(),
        },
        ResourceUsage {
            key: (1, Arc::as_ptr(&mesh.vertex_buffer) as usize),
            references: 1,
            bytes: mesh.gpu_allocation_bytes(),
        },
        ResourceUsage {
            key: (
                2,
                mesh.material()
                    .map_or(0, |material| material.texture().identity_key()),
            ),
            references: 1,
            bytes: mesh
                .material()
                .map_or(0, |material| material.texture().recovery_memory_bytes()),
        },
        ResourceUsage {
            key: (
                3,
                mesh.material()
                    .map_or(0, |material| material.texture().identity_key()),
            ),
            references: 1,
            bytes: mesh
                .material()
                .map_or(0, |material| material.texture().gpu_allocation_bytes()),
        },
    ]
}

fn check_scene_limit(
    resource: Scene3dBudgetResource,
    limit: usize,
    actual: usize,
) -> Result<(), Scene3dError> {
    if actual > limit {
        Err(Scene3dError::BudgetExceeded {
            resource,
            limit,
            actual,
        })
    } else {
        Ok(())
    }
}

fn storage_bytes(instances: usize, slots: usize, resources: usize) -> usize {
    instances
        .saturating_mul(std::mem::size_of::<Mesh3dInstance>())
        .saturating_add(slots.saturating_mul(std::mem::size_of::<ObjectSlot>()))
        .saturating_add(resources.saturating_mul(std::mem::size_of::<ResourceUsage>()))
}

fn replacement_capacity<T: Clone>(
    current: &Vec<T>,
    capacity: usize,
) -> Result<Option<Vec<T>>, Scene3dError> {
    if capacity <= current.capacity() {
        return Ok(None);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| Scene3dError::AllocationFailed {
            requested_bytes: capacity.saturating_mul(std::mem::size_of::<T>()),
        })?;
    values.extend_from_slice(current);
    Ok(Some(values))
}

fn planned_capacity<T>(current: &Vec<T>, additional: usize, maximum: usize) -> usize {
    let needed = current.len().saturating_add(additional);
    if needed <= current.capacity() {
        current.capacity()
    } else {
        needed
            .max(current.capacity().saturating_mul(2))
            .min(maximum)
    }
}

/// Rejection reason for 3D scene visual state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene3dError {
    /// Each explicit scene budget limit must be nonzero.
    InvalidBudget,
    /// The incoming mesh belongs to another logical renderer generation.
    RendererMismatch,
    /// A mutation would exceed a configured live-resource or storage limit.
    BudgetExceeded {
        /// Rejected resource category.
        resource: Scene3dBudgetResource,
        /// Configured maximum.
        limit: usize,
        /// Required count or bytes.
        actual: usize,
    },
    /// The target clear color must be normalized and opaque.
    InvalidBackground,
    /// Surface and wireframe were both disabled.
    EmptyStyle,
    /// The selected surface/edge modes have no corresponding mesh topology.
    StyleHasNoMatchingGeometry,
    /// A model transform cannot be represented portably by the GPU shader.
    InvalidTransform,
    /// No more process-unique scene identifiers can be allocated.
    SceneIdExhausted,
    /// No more stable object identifiers can be allocated.
    ObjectIdExhausted,
    /// CPU storage for another retained object could not be reserved.
    AllocationFailed {
        /// Additional bytes requested for the rejected object.
        requested_bytes: usize,
    },
    /// An object update referenced a missing stable handle.
    ObjectNotFound {
        /// Stable handle that is absent from this scene.
        object_id: Object3dId,
    },
}

impl fmt::Display for Scene3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget => write!(formatter, "3D scene budget limits must be nonzero"),
            Self::RendererMismatch => {
                write!(formatter, "3D mesh belongs to another renderer generation")
            }
            Self::BudgetExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "3D scene {resource:?} budget exceeded: {actual} > {limit}"
            ),
            Self::InvalidBackground => write!(formatter, "3D scene background must be opaque"),
            Self::EmptyStyle => write!(formatter, "3D object requires a surface or wireframe"),
            Self::StyleHasNoMatchingGeometry => write!(
                formatter,
                "3D object style has no matching triangles or display edges"
            ),
            Self::InvalidTransform => {
                write!(
                    formatter,
                    "3D model transform is outside the portable shader envelope"
                )
            }
            Self::SceneIdExhausted => write!(formatter, "3D scene identifiers exhausted"),
            Self::ObjectIdExhausted => write!(formatter, "3D scene object identifiers exhausted"),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for another 3D scene object"
            ),
            Self::ObjectNotFound { object_id } => write!(
                formatter,
                "3D scene object {} does not exist in this scene",
                object_id.get()
            ),
        }
    }
}

impl Error for Scene3dError {}
