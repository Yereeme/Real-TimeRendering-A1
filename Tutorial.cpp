#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>
#include <variant>

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <functional>
#include <unordered_map>


 
#include "external\tinyobjloader\tiny_obj_loader.h"

 
#include "external\tinyobjloader\stb_image.h"
 
static void collect_scene_cameras(S72 const& scene, std::vector<S72::Node const*>& out);


Tutorial::Tutorial(RTG& rtg_, std::string const& scene_file_) : rtg(rtg_), scene_file(scene_file_) {

	use_s72_scene = !scene_file_.empty();
	if (use_s72_scene) {
		scene = S72::load(scene_file_);

		std::cout << "[A1-load] scene: " << scene_file << "\n";
		std::cout << "  nodes:      " << scene.nodes.size() << "\n";
		std::cout << "  meshes:     " << scene.meshes.size() << "\n";
		std::cout << "  cameras:    " << scene.cameras.size() << "\n";
		std::cout << "  materials:  " << scene.materials.size() << "\n";
		std::cout << "  textures:   " << scene.textures.size() << "\n";
		std::cout << "  datafiles:  " << scene.data_files.size() << "\n";

		collect_scene_cameras(scene, scene_camera_nodes);
		std::cout << "[A1-show] scene cameras found: " << scene_camera_nodes.size() << "\n";

		// pick defaults:
		if (!scene_camera_nodes.empty()) active_scene_camera = 0;

		// give debug camera a sensible starting pose:
		debug_camera = free_camera;
	}else{
		// no scene; keep fallback mode
		scene_camera_nodes.clear();
		active_scene_camera = 0;
		debug_camera = free_camera;
	}


	

	 
	//select a depth format:
	// (at least one of these two must be supported, according to the spec; but neither are required)
	depth_format = rtg.helpers.find_image_format(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32 },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);

	{ //create render pass
		//attachments:
		std::array< VkAttachmentDescription, 2 > attachments{
			VkAttachmentDescription{ //0 - color attachment:
				.format = rtg.surface_format.format, //DEFINE FORMAT
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, //LOADOP LOAD THE DATA
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE, //how to write data back after rendering
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, //LAYOUT IMAGE TRANSITIONED TO BEFORE THE LOAD
				.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, //layout image is transitioned to after the store
			     
			},
			VkAttachmentDescription{ //1 - depth attachment:
				.format = depth_format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				},
		};

		// subpass ( parts of the rendering that can proceed (potentially) in parallel)
		VkAttachmentReference color_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_attachment_ref{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};

		//dependencies
		//this defers the image load actions for the attachments:
		std::array< VkSubpassDependency, 2> dependencies{
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				},
				VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				}
		};


		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};

		VK(vkCreateRenderPass(rtg.device, &create_info, nullptr, &render_pass));
	}

	
	{//create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK(vkCreateCommandPool(rtg.device, &create_info, nullptr, &command_pool));
	}

	//calling create function fron tutorial.hppp

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass, 0);

	{ //create descriptor pool:
		uint32_t per_workspace = uint32_t(rtg.workspaces.size()); //for easier-to-read counting

		std::array< VkDescriptorPoolSize, 2> pool_sizes{
			//we only need uniform buffer descriptors for the moment:
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2 * per_workspace, //one descriptor per set, one set per workspace
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1 * per_workspace, //one descriptor per set, one set per workspace
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_sET_BIT isn't included, **can't** free
			//individual descripttors allocated from this pool
			.maxSets = 3 * per_workspace, //two set per workspace
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool));
	}


	workspaces.resize(rtg.workspaces.size());
	for (Workspace& workspace : workspaces) {
		{//allocate command buffer:
			VkCommandBufferAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &workspace.command_buffer));
		}

		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible
			//memory, coherent (no special sync needed)
			Helpers::Mapped //get a pointer to the memory

		);
		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as a uniform
			//buffer, also going to have GPU copy into this memory
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
			Helpers::Unmapped //don't get a pointer to the memory
		);

		//descriptor set:
		{ //allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lines_pipeline.set0_Camera,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Camera_descriptors));
		}

		workspace.World_src = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors));
			//NOTE: will actually fill in this descriptor set just a bit lower
		}

		{ //allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};

			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors));
			//NOTE: will fill in this descriptor set in render when buffers are [re-allocated]
		}

		//descriptor write:
		{ //point descriptor to Camera buffer:
			VkDescriptorBufferInfo Camera_info{
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};

			std::array< VkWriteDescriptorSet, 2 > writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Camera_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &Camera_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.World_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &World_info,
				},
			};

			//std::array< VkWriteDescriptorSet, 1> writes{
				//VkWriteDescriptorSet{
					//.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					//.dstSet = workspace.Camera_descriptors,
					//.dstBinding = 0,
					//.dstArrayElement = 0,
					//.descriptorCount = 1,
					//.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					//.pBufferInfo = &Camera_info,

//},
			//};


			vkUpdateDescriptorSets(
				rtg.device, //device
				uint32_t(writes.size()), //descriptorWriteCount
				writes.data(), //pDescriptorWrites
				0, //descriptorCopyCount
				nullptr //pDescriptorCopies
			);
		}


	}

	{ //Pack S72 meshes into a single vertex buffer 

	//--- cache raw bytes for each DataFile so we only read each file once:
	std::unordered_map< S72::DataFile const*, std::vector<uint8_t> > data_cache;

	auto load_datafile = [&](S72::DataFile const& df) -> std::vector<uint8_t> const& {
		S72::DataFile const* key = &df;
		auto it = data_cache.find(key);
		if (it != data_cache.end()) return it->second;

		std::ifstream in(df.path, std::ios::binary);
		if (!in) {
			throw std::runtime_error("Failed to open data file: " + df.path);
		}

		in.seekg(0, std::ios::end);
		std::streamsize sz = in.tellg();
		in.seekg(0, std::ios::beg);

		std::vector<uint8_t> bytes;
		bytes.resize(size_t(sz));
		if (sz > 0) {
			in.read(reinterpret_cast<char*>(bytes.data()), sz);
		}

		auto [inserted_it, ok] = data_cache.emplace(key, std::move(bytes));
		return inserted_it->second;
		};

	auto find_attr = [&](S72::Mesh const& mesh, std::initializer_list<const char*> names) -> S72::Mesh::Attribute const* {
		for (auto n : names) {
			auto it = mesh.attributes.find(n);
			if (it != mesh.attributes.end()) return &it->second;
		}
		return nullptr;
		};

	auto read_vec3_f32 = [&](std::vector<uint8_t> const& bytes, size_t byte_offset) -> S72::vec3 {
		// assumes little-endian float32, which is what your class data will be
		if (byte_offset + 12 > bytes.size()) return S72::vec3{ 0,0,0 };
		float const* f = reinterpret_cast<float const*>(bytes.data() + byte_offset);
		return S72::vec3{ f[0], f[1], f[2] };
		};

	auto read_vec2_f32 = [&](std::vector<uint8_t> const& bytes, size_t byte_offset) -> std::pair<float, float> {
		if (byte_offset + 8 > bytes.size()) return { 0.0f, 0.0f };
		float const* f = reinterpret_cast<float const*>(bytes.data() + byte_offset);
		return { f[0], f[1] };
		};

	auto read_index = [&](std::vector<uint8_t> const& bytes, size_t byte_offset, VkIndexType type) -> uint32_t {
		if (type == VK_INDEX_TYPE_UINT16) {
			if (byte_offset + 2 > bytes.size()) return 0;
			uint16_t const* p = reinterpret_cast<uint16_t const*>(bytes.data() + byte_offset);
			return uint32_t(*p);
		}
		else if (type == VK_INDEX_TYPE_UINT32) {
			if (byte_offset + 4 > bytes.size()) return 0;
			uint32_t const* p = reinterpret_cast<uint32_t const*>(bytes.data() + byte_offset);
			return *p;
		}
		else {
			// unsupported index type
			return 0;
		}
		};

	//--- pack everything:
	std::vector< PosNorTexVertex > packed;
	packed.reserve(4096);

	s72_mesh_to_range.clear();


	// NOTE: scene.meshes is an unordered_map, so iteration order is arbitrary.
	// That's fine for now as long as we build instances using Mesh* later, not by index.
	for (auto const& kv : scene.meshes) {
		S72::Mesh const& mesh = kv.second;
		S72::Mesh const* mesh_ptr = &mesh;


		ObjectVertices range;
		range.first = uint32_t(packed.size());

		// We’ll support the common attribute names used in s72/glTF style exports.
		// If your exporter uses different keys, add them here.
		S72::Mesh::Attribute const* posA = find_attr(mesh, { "POSITION", "position", "pos" });
		S72::Mesh::Attribute const* norA = find_attr(mesh, { "NORMAL", "normal", "nor" });
		S72::Mesh::Attribute const* uvA = find_attr(mesh, { "TEXCOORD", "TEXCOORD_0", "texcoord", "uv", "UV" });

		if (!posA) {
			std::cout << "[A1-load] mesh '" << mesh.name << "' has no POSITION attribute; skipping.\n";
			range.count = 0;
			s72_mesh_to_range[mesh_ptr] = range;

			continue;
		}

		// Basic format sanity (you can relax later if needed):
		if (posA->format != VK_FORMAT_R32G32B32_SFLOAT) {
			std::cout << "[A1-load] mesh '" << mesh.name << "' POSITION format not vec3 f32; skipping.\n";
			range.count = 0;
			s72_mesh_to_range[mesh_ptr] = range;

			continue;
		}
		if (norA && norA->format != VK_FORMAT_R32G32B32_SFLOAT) {
			norA = nullptr; // ignore weird normals for now
		}
		if (uvA && uvA->format != VK_FORMAT_R32G32_SFLOAT) {
			uvA = nullptr; // ignore weird uvs for now
		}

		// Load the underlying data files:
		auto const& posBytes = load_datafile(posA->src);
		std::vector<uint8_t> const* norBytes = nullptr;
		std::vector<uint8_t> const* uvBytes = nullptr;

		if (norA) norBytes = &load_datafile(norA->src);
		if (uvA)  uvBytes = &load_datafile(uvA->src);

		auto emit_vertex = [&](uint32_t vtx_index) {
			size_t pos_off = size_t(posA->offset) + size_t(vtx_index) * size_t(posA->stride);
			S72::vec3 p = read_vec3_f32(posBytes, pos_off);

			S72::vec3 n{ 0.0f, 0.0f, 1.0f };
			if (norA && norBytes) {
				size_t nor_off = size_t(norA->offset) + size_t(vtx_index) * size_t(norA->stride);
				n = read_vec3_f32(*norBytes, nor_off);
			}

			float s = 0.0f, t = 0.0f;
			if (uvA && uvBytes) {
				size_t uv_off = size_t(uvA->offset) + size_t(vtx_index) * size_t(uvA->stride);
				auto uv = read_vec2_f32(*uvBytes, uv_off);
				s = uv.first;
				t = 1.0f - uv.second; // flip V like you already do
			}

			packed.emplace_back(PosNorTexVertex{
				.Position{.x = p.x, .y = p.y, .z = p.z },
				.Normal  {.x = n.x, .y = n.y, .z = n.z },
				.TexCoord{.s = s,   .t = t   },
				});
			};

		// Expand indices (or emit sequential vertices if non-indexed):
		if (mesh.indices.has_value()) {
			auto const& idx = mesh.indices.value();
			auto const& idxBytes = load_datafile(idx.src);

			// We assume triangles for now. If not triangles, we still just expand in given order.
			for (uint32_t i = 0; i < mesh.count; ++i) {
				size_t idx_off = size_t(idx.offset);
				if (idx.format == VK_INDEX_TYPE_UINT16) idx_off += size_t(i) * 2;
				else idx_off += size_t(i) * 4;

				uint32_t v = read_index(idxBytes, idx_off, idx.format);
				emit_vertex(v);
			}
		}
		else {
			for (uint32_t v = 0; v < mesh.count; ++v) emit_vertex(v);
		}

		range.count = uint32_t(packed.size()) - range.first;
		s72_mesh_to_range[mesh_ptr] = range;


		std::cout << "[A1-load] mesh '" << mesh.name << "' packed verts=" << range.count << "\n";
		std::cout << "[A1-load] mesh->range entries = " << s72_mesh_to_range.size() << "\n";

	}

	// Upload packed buffer:
	size_t bytes = packed.size() * sizeof(packed[0]);

	if (object_vertices.handle != VK_NULL_HANDLE) {
		rtg.helpers.destroy_buffer(std::move(object_vertices));
	}

	object_vertices = rtg.helpers.create_buffer(
		bytes,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);

	if (!packed.empty()) {
		rtg.helpers.transfer_to_buffer(packed.data(), bytes, object_vertices);
	}

	std::cout << "[A1-load] S72 packed total verts=" << packed.size() << " bytes=" << bytes << "\n";
}


	if (scene_file.empty()) {
		{// create object vertices

			std::vector< PosNorTexVertex > vertices;

			auto append_obj = [&](const std::string& obj_path) -> ObjectVertices {
				ObjectVertices out;
				out.first = uint32_t(vertices.size());

				tinyobj::attrib_t attrib;
				std::vector<tinyobj::shape_t> shapes;
				std::string warn, err;

				bool ok = tinyobj::LoadObj(
					&attrib,
					&shapes,
					nullptr,          // no mtl
					&warn,
					&err,
					obj_path.c_str(),
					"data/",          // base dir
					true              // triangulate
				);

				if (!warn.empty()) std::cout << warn << std::endl;
				if (!err.empty())  std::cout << err << std::endl;
				if (!ok) {
					std::cout << "Failed to load " << obj_path << std::endl;
					out.count = 0;
					return out;
				}

				for (const auto& shape : shapes) {
					size_t index_offset = 0;
					for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
						int fv = shape.mesh.num_face_vertices[f];
						if (fv < 3) { index_offset += fv; continue; }

						for (int v = 0; v < 3; ++v) {
							tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

							float px = attrib.vertices[3 * idx.vertex_index + 0];
							float py = attrib.vertices[3 * idx.vertex_index + 1];
							float pz = attrib.vertices[3 * idx.vertex_index + 2];

							float nx = 0, ny = 0, nz = 1;
							if (idx.normal_index >= 0 && !attrib.normals.empty()) {
								nx = attrib.normals[3 * idx.normal_index + 0];
								ny = attrib.normals[3 * idx.normal_index + 1];
								nz = attrib.normals[3 * idx.normal_index + 2];
							}

							float ts = 0, tt = 0;
							if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
								ts = attrib.texcoords[2 * idx.texcoord_index + 0];
								tt = 1.0f - attrib.texcoords[2 * idx.texcoord_index + 1];

							}

							vertices.emplace_back(PosNorTexVertex{
								.Position{.x = px, .y = py, .z = pz},
								.Normal{.x = nx, .y = ny, .z = nz},
								.TexCoord{.s = ts, .t = tt},
								});
						}
						index_offset += fv;
					}
				}

				out.count = uint32_t(vertices.size()) - out.first;
				std::cout << "Loaded " << obj_path << " verts=" << out.count << "\n";
				return out;
				};

			chen_body_vertices = append_obj("data/chen_body.obj");
			chen_clothes_vertices = append_obj("data/chen_clothes.obj");
			chen_hairs_vertices = append_obj("data/chen_hairs.obj");
			chen_face_vertices = append_obj("data/chen_face.obj");
			chen_iris_vertices = append_obj("data/chen_iris.obj");
			chen_sword_vertices = append_obj("data/chen_sword.obj");





			{//A [-1,1]x[-1,1]x{0} quadrilateral:
				plane_vertices.first = uint32_t(vertices.size());

				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 0.0f, .t = 0.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 1.0f, .t = 0.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 0.0f, .t = 1.0f },

					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f },
					.TexCoord{.s = 1.0f, .t = 1.0f },
					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = -1.0f, .y = 1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f},
					.TexCoord{.s = 0.0f, .t = 1.0f },
					});
				vertices.emplace_back(PosNorTexVertex{
					.Position{.x = 1.0f, .y = -1.0f, .z = 0.0f },
					.Normal{.x = 0.0f, .y = 0.0f, .z = 1.0f},
					.TexCoord{.s = 1.0f, .t = 0.0f },
					});

				plane_vertices.count = uint32_t(vertices.size()) - plane_vertices.first;
			}

			{ // A torus:
				torus_vertices.first = uint32_t(vertices.size());

				//torus!
				//will parameterize with (u,v) where:
				// - u is angle around main axis (+z)
				// - v is angle around the tube

				constexpr float R1 = 0.75f; //main radius
				constexpr float R2 = 0.15F; //tube radius

				constexpr uint32_t U_STEPS = 20;
				constexpr uint32_t V_STEPS = 16;

				//texture repeats around the torus:
				constexpr float V_REPEATS = 2.0f;
				constexpr float U_REPEATS = int(V_REPEATS / R2 * R1 + 0.999f); //approximately square, 
				//rounded up

				auto emplace_vertex = [&](uint32_t ui, uint32_t vi) {
					//convert steps to angles:
					// (doing the mod since trig on 2 M_PI nay not exactly match 0)
					float ua = (ui % U_STEPS) / float(U_STEPS) * 2.0F * float(M_PI);
					float va = (vi % V_STEPS) / float(V_STEPS) * 2.0F * float(M_PI);

					vertices.emplace_back(PosNorTexVertex{
						.Position{
							.x = (R1 + R2 * std::cos(va)) * std::cos(ua),
							.y = (R1 + R2 * std::cos(va)) * std::sin(ua),
							.z = R2 * std::sin(va),
						},
						.Normal{
							.x = std::cos(va) * std::cos(ua),
							.y = std::cos(va) * std::sin(ua),
							.z = std::sin(ua),
						},
						.TexCoord{
							.s = ui / float(U_STEPS) * U_REPEATS,
							.t = vi / float(V_STEPS) * V_REPEATS,
						},
						});
					};

				for (uint32_t ui = 0; ui < U_STEPS; ++ui) {
					for (uint32_t vi = 0; vi < V_STEPS; ++vi) {
						emplace_vertex(ui, vi);
						emplace_vertex(ui + 1, vi);
						emplace_vertex(ui, vi + 1);

						emplace_vertex(ui, vi + 1);
						emplace_vertex(ui + 1, vi);
						emplace_vertex(ui + 1, vi + 1);


					}
				}

				torus_vertices.count = uint32_t(vertices.size()) - torus_vertices.first;
			}

			{ //A low-poly crystal (stylized gem) - simple + looks nice:
				crystal_vertices.first = uint32_t(vertices.size());

				//local small vec type since PosNorTexVertex uses anonymous structs:
				struct P3 { float x, y, z; };

				constexpr uint32_t STEPS = 10; //8-12 looks good
				constexpr float R = 0.55f;   //ring radius
				constexpr float TOP = 1.0f;    //top point height
				constexpr float MID = 0.15f;   //ring height
				constexpr float BOT = -0.9f;   //bottom point height

				P3 top{ 0.0f, 0.0f, TOP };
				P3 bot{ 0.0f, 0.0f, BOT };

				auto add_tri = [&](P3 p0, P3 p1, P3 p2, P3 n,
					float s0, float t0,
					float s1, float t1,
					float s2, float t2) {
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p0.x, .y = p0.y, .z = p0.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s0, .t = t0 },
							});
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p1.x, .y = p1.y, .z = p1.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s1, .t = t1 },
							});
						vertices.emplace_back(PosNorTexVertex{
							.Position{.x = p2.x, .y = p2.y, .z = p2.z },
							.Normal{.x = n.x,  .y = n.y,  .z = n.z  },
							.TexCoord{.s = s2, .t = t2 },
							});
					};

				auto ring_point = [&](uint32_t i) -> P3 {
					float a = (i % STEPS) / float(STEPS) * 2.0f * float(M_PI);
					return P3{ R * std::cos(a), R * std::sin(a), MID };
					};

				for (uint32_t i = 0; i < STEPS; ++i) {
					P3 p0 = ring_point(i);
					P3 p1 = ring_point(i + 1);

					//outward-ish normals, good enough for now:
					float mx = (p0.x + p1.x) * 0.5f;
					float my = (p0.y + p1.y) * 0.5f;

					P3 nTop{ mx, my, 1.0f };
					P3 nBot{ mx, my, -1.0f };

					//top faces (CCW from outside)
					add_tri(p0, p1, top, nTop,
						i / float(STEPS), 0.0f,
						(i + 1) / float(STEPS), 0.0f,
						(i + 0.5f) / float(STEPS), 1.0f);

					//bottom faces (CCW from outside)
					add_tri(p1, p0, bot, nBot,
						(i + 1) / float(STEPS), 0.0f,
						i / float(STEPS), 0.0f,
						(i + 0.5f) / float(STEPS), 1.0f);
				}

				crystal_vertices.count = uint32_t(vertices.size()) - crystal_vertices.first;
			}







			size_t bytes = vertices.size() * sizeof(vertices[0]);

			object_vertices = rtg.helpers.create_buffer(
				bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			);

			//copy data to buffer:
			rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
		}
	}

	{ //make some textures
		textures.reserve(3);
		//textures.reserve(2);

		auto load_png_texture_srgb = [&](std::string const& tex_path) -> uint32_t {
			auto it = s72_texture_path_to_index.find(tex_path);
			if (it != s72_texture_path_to_index.end()) return it->second;

			int w = 0, h = 0, n = 0;
			stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
			if (!pixels) {
				std::cout << "[A1-show] failed to load texture: " << tex_path << "\n";
				return 0; // checker fallback
			}

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
			stbi_image_free(pixels);

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[tex_path] = idx;
			return idx;
			};

		auto make_solid_srgb_texture = [&](S72::color const& c) -> uint32_t {
			// quantize to 8-bit and use as a cache key:
			auto to_u8 = [](float v) -> uint8_t {
				v = std::max(0.0f, std::min(1.0f, v));
				return uint8_t(std::round(v * 255.0f));
				};
			uint8_t r = to_u8(c.r), g = to_u8(c.g), b = to_u8(c.b);

			char key[64];
			std::snprintf(key, sizeof(key), "solid_%u_%u_%u", r, g, b);
			auto it = s72_texture_path_to_index.find(key);
			if (it != s72_texture_path_to_index.end()) return it->second;

			uint32_t pixel = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (0xffu << 24);

			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1, .height = 1 },
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped
			));

			rtg.helpers.transfer_to_image(&pixel, sizeof(pixel), textures.back());

			uint32_t idx = uint32_t(textures.size() - 1);
			s72_texture_path_to_index[key] = idx;
			return idx;
			};

		// --- Build material->texture mapping for S72:
		if (use_s72_scene) {
			material_to_texture.clear();

			for (auto const& mkv : scene.materials) {
				S72::Material const& mat = mkv.second;
				uint32_t tex_idx = 0;

				// handle Lambertian + PBR albedo only (enough for A1-show)
				if (auto const* lam = std::get_if<S72::Material::Lambertian>(&mat.brdf)) {
					if (std::holds_alternative<S72::color>(lam->albedo)) {
						tex_idx = make_solid_srgb_texture(std::get<S72::color>(lam->albedo));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(lam->albedo);
						if (t && t->type == S72::Texture::Type::flat) {
							tex_idx = load_png_texture_srgb(t->path);
						}
					}
				}
				else if (auto const* pbr = std::get_if<S72::Material::PBR>(&mat.brdf)) {
					if (std::holds_alternative<S72::color>(pbr->albedo)) {
						tex_idx = make_solid_srgb_texture(std::get<S72::color>(pbr->albedo));
					}
					else {
						S72::Texture* t = std::get<S72::Texture*>(pbr->albedo);
						if (t && t->type == S72::Texture::Type::flat) {
							tex_idx = load_png_texture_srgb(t->path);
						}
					}
				}

				material_to_texture[&mat] = tex_idx;
			}

			std::cout << "[A1-show] mapped materials -> textures: " << material_to_texture.size() << "\n";
		}

		{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
			//actually make the texture:
			uint32_t size = 128;
			std::vector< uint32_t > data;
			data.reserve(size * size);
			for (uint32_t y = 0; y < size; ++y) {
				float fy = (y + 0.5f) / float(size);
				for (uint32_t x = 0; x < size; ++x) {
					float fx = (x + 0.5f) / float(size);
					//highlight the origin:
					if (fx < 0.05f && fy < 0.05f) data.emplace_back(0xff0000ff); //red
					else if ((fx < 0.5f) == (fy < 0.5f)) data.emplace_back(0xff444444); //dark grey
					else data.emplace_back(0xffbbbbbb); //light grey
				}
			}
			assert(data.size() == size * size);

			//make a place for the texture to live on the GPU
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = size , .height = size }, //size of image
				VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
				Helpers::Unmapped
			));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		}

			{ //texture 1 will be a classic 'xor' texture
				//actually make the texture:
				uint32_t size = 256;
				std::vector< uint32_t > data;
				data.reserve(size* size);
				for (uint32_t y = 0; y < size; ++y) {
					for (uint32_t x = 0; x < size; ++x) {
						uint8_t r = uint8_t(x) ^ uint8_t(y);
						uint8_t g = uint8_t(x + 128) ^ uint8_t(y);
						uint8_t b = uint8_t(x) ^ uint8_t(y + 27);
						uint8_t a = 0xff;
						data.emplace_back(uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24));
					}
				}
				assert(data.size() == size * size);

				//make a place for the texture to live on the GPU:
				textures.emplace_back(rtg.helpers.create_image(
					VkExtent2D{ .width = size , .height = size }, //size of image
					VK_FORMAT_R8G8B8A8_SRGB, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
					Helpers::Unmapped
				));

			//transfer data:
			rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
			}


			if (scene_file.empty()) {
				auto load_png_texture = [&](std::string const& tex_path) -> uint32_t {
					int w = 0, h = 0, n = 0;
					stbi_uc* pixels = stbi_load(tex_path.c_str(), &w, &h, &n, 4);
					if (!pixels) {
						std::cout << "stb_image failed to load: " << tex_path << std::endl;
						return 0; //fallback to texture 0
					}

					textures.emplace_back(rtg.helpers.create_image(
						VkExtent2D{ .width = uint32_t(w), .height = uint32_t(h) },
						VK_FORMAT_R8G8B8A8_SRGB,
						VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						Helpers::Unmapped
					));

					rtg.helpers.transfer_to_image(pixels, size_t(w) * size_t(h) * 4, textures.back());
					stbi_image_free(pixels);

					uint32_t idx = uint32_t(textures.size() - 1);
					std::cout << "Loaded texture[" << idx << "]: " << tex_path << " (" << w << "x" << h << ")\n";
					return idx;
					};




				// load textures and store the actual indices:
				tex_body = load_png_texture("data/chen_body.png");
				tex_clothes = load_png_texture("data/chen_clothes.png");
				tex_hair = load_png_texture("data/chen_hair.png");
				tex_sword = load_png_texture("data/chen_sword.png");
				tex_iris = load_png_texture("data/chen_iris.png");
				tex_face = load_png_texture("data/chen_face.png");
			}

		
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const& image : textures) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				// .components sets swizzling and is fine when zero-initialized
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK(vkCreateImageView(rtg.device, &create_info, nullptr, &image_view));

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ //make a sampler for the textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK(vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler));
	}

	{ //create the texture descriptor pool
		uint32_t per_texture = uint32_t(textures.size()); //for easier-to-read counting

		std::array< VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1 * 1 * per_texture, //one descriptor per set, one set per texture
			},
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = 1 * per_texture, //one set per texture
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool));
	}

	{ //allocate and write the texture descriptor sets
		//allocate the descriptors (using the same alloc_info):
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_TEXTURE,
		};
		texture_descriptors.assign(textures.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet& descriptor_set : texture_descriptors) {
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set));
		}

		//write descriptors for textures
		std::vector< VkDescriptorImageInfo > infos(textures.size());
		std::vector< VkWriteDescriptorSet > writes(textures.size());

		for (Helpers::AllocatedImage const& image : textures) {
			size_t i = &image - &textures[0];

			infos[i] = VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[i],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			writes[i] = VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture_descriptors[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &infos[i],
			};
		}

		vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
	}
}


Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	if (texture_descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = nullptr;

		//this also frees the descriptor sets allocated from the pool:
		texture_descriptors.clear();
	}

	if (texture_sampler) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView& view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto& texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();


	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces) {
		 
		if (workspace.command_buffer != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(rtg.device,  command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}

		//cleaning up per-workspace lines buffers in Tutorial::~Tutorial
		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}

		if (workspace.Camera_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}

		//Camera_descriptors freed when pool is destroyed.

		if (workspace.World_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		//World_descriptors freed when pool is destroyed.

		

		if (workspace.Transforms_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		//Transforms_descriptor freed when pool is destroyed

	}

	
	workspaces.clear();

	if (descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
		//(this also frees the descriptor sets allocated from the pool)
	}

	//destroy pipeline (sequenced in opposite order of construction)

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);

	//destroy command pool
	if (command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE) { //cleanup
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}
}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain) {
	 //TODO: clean up existing framebuffers
	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}
	//Allocate depth image for framebuffers to share:
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent,
		depth_format,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);

	{//create an image view of the depth image:
		VkImageViewCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_depth_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
		};

		VK(vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view));
	}

	//create framebuffers pointing to each swapchain image view and the shared depth image view
	//framebuffers for each swapchain image:
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i) {
		std::array< VkImageView, 2 > attachments{
			swapchain.image_views[i],
			swapchain_depth_image_view,
		};
		VkFramebufferCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = render_pass,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.width = swapchain.extent.width,
			.height = swapchain.extent.height,
			.layers = 1,
		};

		VK(vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]));
	}
}

void Tutorial::destroy_framebuffers() {
	 
	for (VkFramebuffer& framebuffer : swapchain_framebuffers) {
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}


void Tutorial::render(RTG& rtg_, RTG::RenderParams const& render_params) {
	//assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	//get more convenient names for the current workspace and target framebuffer:
	Workspace& workspace = workspaces[render_params.workspace_index];
	VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	//reset the command buffer (clear old commands):
	VK(vkResetCommandBuffer(workspace.command_buffer, 0));
	{//begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, //set to the proper for this structure
			//.pNext set to null by zero-initialization!
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, //will record again every submit
		};
		VK(vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}

	if (!lines_vertices.empty()) {//upload lines vertices:
		//[re-]allocate lines buffers if needed:
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE ||
			workspace.lines_vertices_src.size < needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.lines_vertices_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}
			//actual memory allocation
			workspace.lines_vertices_src = rtg.helpers.create_buffer( //use staging buffer
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory,
				//coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(//
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as vertex buffer
				//also going to have GPU into this memory 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			std::cout << "Re-allocated lines buffers to " << new_bytes << "bytes." << std::endl;

		}

		assert(workspace.lines_vertices_src.size == workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);

		//host-side copy into lines_vertices_src;
		assert(workspace.lines_vertices_src.allocation.mapped);
		//helper allocatin data member function is to account for any offset in the allocation when getting a
		//pointer to start the allocation's mapped memory
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		//device-side copy from lines_vertices_src -> lines_vertices;
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.
			handle, 1, &copy_region);
	}

	{//upload camera info:
		LinesPipeline::Camera camera{
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD
		};
		assert(workspace.Camera_src.size == sizeof(camera));

		//host-side copy into Camera_src:
		memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

		//add device-side copy from Camera_src -> Camera:
		assert(workspace.Camera_src.size == workspace.Camera.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Camera_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle,
			workspace.Camera.handle, 1, &copy_region);
	}

	{ //upload world info:
		assert(workspace.World_src.size == sizeof(world));

		//host-side copy into World_src:
		memcpy(workspace.World_src.allocation.data(), &world, sizeof(world));

		//add device-side copy from World_src -> World:
		assert(workspace.World_src.size == workspace.World.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.World_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.World_src.handle, workspace.World.handle, 1, &copy_region);
	}



	if (!object_instances.empty()) { //upload object transforms:
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size <
			needed_bytes) {
			//round to next multiple of 4k to avoid re-allocating continuously 
			// if vertex count grows slowly:
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			if (workspace.Transforms_src.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle) {
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}
			//actual memory allocation
			workspace.Transforms_src = rtg.helpers.create_buffer( //use staging buffer
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT, //going to have GPU copy from this memory
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, //host-visible memory,
				//coherent (no special sync needed)
				Helpers::Mapped //get a pointer to the memory
			);
			workspace.Transforms = rtg.helpers.create_buffer(//
				new_bytes,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //going to use as vertex buffer
				//also going to have GPU into this memory 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //GPU-local memory
				Helpers::Unmapped //don't get a pointer to the memory
			);

			//update the descriptor set:
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};

			std::array< VkWriteDescriptorSet, 1 > writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Transforms_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Transforms_info,
					},
			};

			vkUpdateDescriptorSets(
				rtg.device,
				uint32_t(writes.size()), writes.data(),  //descriptorWrites count, data
				0, nullptr //descriptorCopies count, data
			);

			std::cout << "Re-allocated object transforms buffers to " << new_bytes << "bytes."
				<< std::endl;
		}

		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);

		{ //copy transforms into Transforms_src:
			assert(workspace.Transforms_src.allocation.mapped);
			ObjectsPipeline::Transform* out = reinterpret_cast<ObjectsPipeline::Transform*>
				(workspace.Transforms_src.allocation.data()); // Strict aliasing violation, but it doesn't matter
			for (ObjectInstance const& inst : object_instances) {
				*out = inst.transform;
				++out;
			}
		}

		//device-side copy from Transform_src -> Transform;
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle,
			workspace.Transforms.handle, 1, &copy_region);

	}


	{//Memory barrier to make sure copies complete before rendering happens;
		VkMemoryBarrier memory_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
			| VK_ACCESS_UNIFORM_READ_BIT
			| VK_ACCESS_SHADER_READ_BIT,
		};

		vkCmdPipelineBarrier(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, //srcStageMask
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, //dependencyFlags
			1, &memory_barrier, //memoryBarriers (count, data)
			0, nullptr, //bufferMemoryBarriers (count, data)
			0, nullptr //imageMemoryBarriers (count, data)
		);
	}

	//GPU commands here:
	{//render pass
		std::array< VkClearValue, 2 > clear_values{
			VkClearValue{.color{.float32{1.0f, 0.85f, 0.90f, 1.0f}
}},
			VkClearValue{.depthStencil{.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			 },

			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};

		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		//TODO: run pipelines here
		{//set scissor rectangle
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &draw_scissor);
		}
		{//configure viewport transform
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &draw_viewport);
		}

		{//draw with background pipeline:
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				background_pipeline.handle);

			{//push time:
				BackgroundPipeline::Push push{
					.time = time,
				};
				vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout,
					VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
			}
			vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		}

		{ //draw with the lines pipeline:
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);
			{ //use lines_vertices (offset 0) as vertex buffer binding 0:

				std::array< VkBuffer, 1> vertex_buffers{ workspace.lines_vertices.handle };
				std::array< VkDeviceSize, 1> offsets{ 0 };
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.
					data(), offsets.data());

			}

			{//bind Camera descriptor set:
				std::array< VkDescriptorSet, 1 > descriptor_sets{
					workspace.Camera_descriptors, //0: Camera
				};
				vkCmdBindDescriptorSets(
					workspace.command_buffer, //command buffer
					VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
					lines_pipeline.layout, //pipeline layout
					0, //first set
					//uint32_t(descriptor_sets.size())
					1, descriptor_sets.data(), //descriptor sets
					//count, ptr
					0, nullptr //dynamic offsets count, ptr
				);
			}

			 

			//draw lines vertices:
			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);

		}

		{//draw with object pipeline
			if (!object_instances.empty()) { //draw with the objects pipeline:
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
					objects_pipeline.handle);

				{ //use object vertices (offset 0) as vertex buffer binding 0:
					std::array< VkBuffer, 1 > vertex_buffers{ object_vertices.handle };
					std::array< VkDeviceSize, 1 > offsets{ 0 };
					vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()),
						vertex_buffers.data(), offsets.data());
				}

				{ //bind Transforms descriptor set:
					std::array< VkDescriptorSet, 2 > descriptor_sets{
					workspace.World_descriptors, //set 0 world
					workspace.Transforms_descriptors, //1: Transforms
					};

					vkCmdBindDescriptorSets(
						workspace.command_buffer, //command buffer
						VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
						objects_pipeline.layout, //pipeline layout
						0, //first set
						uint32_t(descriptor_sets.size()), descriptor_sets.data(), //descriptor sets count, ptr
						0, nullptr //dynamic offsets count, ptr
					);

				}

				//Camera descriptor set is still bound, but unused(!)

				//draw all instances:
				for (ObjectInstance const& inst : object_instances) {
					uint32_t index = uint32_t(&inst - &object_instances[0]);
					//vkCmdDraw(workspace.command_buffer, uint32_t(object_vertices.size / sizeof(ObjectsPipeline::Vertex)), 1, 0, 0);
					//draw torus vertices:
					//vkCmdDraw(workspace.command_buffer, torus_vertices.count, 1, torus_vertices.first, 0);

					//bind texture descriptor set:
					vkCmdBindDescriptorSets(
						workspace.command_buffer, //command buffer
						VK_PIPELINE_BIND_POINT_GRAPHICS, //pipeline bind point
						objects_pipeline.layout, //pipeline layout
						2, //second set
						1, &texture_descriptors[inst.texture], //descriptor sets count, ptr
						0, nullptr //dynamic offsets count, ptr
					);

					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first,
						index);
				}
			}
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}

	//end recording:
	VK(vkEndCommandBuffer(workspace.command_buffer));


	{ //submit `workspace.command buffer` for the GPU to run:

		std::array< VkSemaphore, 1 > wait_semaphores{
			render_params.image_available
		};
		std::array< VkPipelineStageFlags, 1 > wait_stages{
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		};
		static_assert(wait_semaphores.size() == wait_stages.size(), "every semaphore needs a stage");

		std::array< VkSemaphore, 1 > signal_semaphores{
			render_params.image_done
		};
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = uint32_t(wait_semaphores.size()),
			.pWaitSemaphores = wait_semaphores.data(),
			.pWaitDstStageMask = wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &workspace.command_buffer,
			.signalSemaphoreCount = uint32_t(signal_semaphores.size()),
			.pSignalSemaphores = signal_semaphores.data(),
		};

		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, render_params.workspace_available));
	}
}

static mat4 mat4_identity() {
	return mat4{
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
}

static mat4 mat4_translate(float x, float y, float z) {
	return mat4{
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		x,y,z,1
	};
}

static mat4 mat4_scale(float x, float y, float z) {
	return mat4{
		x,0,0,0,
		0,y,0,0,
		0,0,z,0,
		0,0,0,1
	};
}

// quaternion (x,y,z,w) to rotation matrix:
static mat4 mat4_from_quat(float x, float y, float z, float w) {
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;

	// column-major mat4 matching your existing usage (translation in last row)
	return mat4{
		1.0f - 2.0f * (yy + zz),  2.0f * (xy + wz),        2.0f * (xz - wy),        0.0f,
		2.0f * (xy - wz),        1.0f - 2.0f * (xx + zz),  2.0f * (yz + wx),        0.0f,
		2.0f * (xz + wy),        2.0f * (yz - wx),        1.0f - 2.0f * (xx + yy),  0.0f,
		0.0f,                  0.0f,                  0.0f,                  1.0f
	};
}

static mat4 mat4_mul(mat4 const& A, mat4 const& B) {
	mat4 R{};
	for (int c = 0; c < 4; ++c) {
		for (int r = 0; r < 4; ++r) {
			R[c * 4 + r] =
				A[0 * 4 + r] * B[c * 4 + 0] +
				A[1 * 4 + r] * B[c * 4 + 1] +
				A[2 * 4 + r] * B[c * 4 + 2] +
				A[3 * 4 + r] * B[c * 4 + 3];
		}
	}
	return R;
}

static mat4 mat4_inverse_rigid(mat4 const& M) {
	// Assumes M is rotation + translation only (no scale/shear).
	// Your mat4 is column-major; translation is in the last row (indices 12,13,14).

	// Extract rotation (upper-left 3x3):
	float r00 = M[0], r01 = M[4], r02 = M[8];
	float r10 = M[1], r11 = M[5], r12 = M[9];
	float r20 = M[2], r21 = M[6], r22 = M[10];

	// Transpose rotation:
	float t00 = r00, t01 = r10, t02 = r20;
	float t10 = r01, t11 = r11, t12 = r21;
	float t20 = r02, t21 = r12, t22 = r22;

	// Translation (your convention):
	float tx = M[12], ty = M[13], tz = M[14];

	// New translation = -R^T * t
	float ntx = -(t00 * tx + t01 * ty + t02 * tz);
	float nty = -(t10 * tx + t11 * ty + t12 * tz);
	float ntz = -(t20 * tx + t21 * ty + t22 * tz);

	return mat4{
		t00, t10, t20, 0.0f,
		t01, t11, t21, 0.0f,
		t02, t12, t22, 0.0f,
		ntx, nty, ntz, 1.0f
	};
}

static void collect_scene_cameras(S72 const& scene, std::vector<S72::Node const*>& out) {
	out.clear();

	std::function<void(S72::Node const&)> walk;
	walk = [&](S72::Node const& n) {
		if (n.camera) out.emplace_back(&n);
		for (S72::Node* child : n.children) {
			if (child) walk(*child);
		}
		};

	for (S72::Node* root : scene.scene.roots) {
		if (root) walk(*root);
	}
}

void Tutorial::compute_letterbox(float target_aspect) {
	//target_aspect == 0 => full screen
	float W = float(rtg.swapchain_extent.width);
	float H = float(rtg.swapchain_extent.height);

	//default full screen:
	draw_viewport = VkViewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = W,
		.height = H,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	draw_scissor = VkRect2D{
		.offset = {0, 0},
		.extent = rtg.swapchain_extent,
	};

	if (target_aspect <= 0.0f) return;

	float win_aspect = W / H;

	//If camera is wider than window => letterbox (bars top/bottom)
	//If camera is taller than window => pillarbox (bars left/right)
	if (target_aspect > win_aspect) {
		//fit width
		float newH = W / target_aspect;
		float y0 = (H - newH) * 0.5f;

		draw_viewport.y = y0;
		draw_viewport.height = newH;

		draw_scissor.offset.y = int32_t(std::round(y0));
		draw_scissor.extent.height = uint32_t(std::round(newH));
	}
	else {
		//fit height
		float newW = H * target_aspect;
		float x0 = (W - newW) * 0.5f;

		draw_viewport.x = x0;
		draw_viewport.width = newW;

		draw_scissor.offset.x = int32_t(std::round(x0));
		draw_scissor.extent.width = uint32_t(std::round(newW));
	}
}


void Tutorial::update(float dt) {
	//time += dt;
	time = std::fmod(time + dt, 60.0f);

	auto local_from_node = [&](S72::Node const& n) -> mat4 {
		mat4 T = mat4_translate(n.translation.x, n.translation.y, n.translation.z);
		mat4 R = mat4_from_quat(n.rotation.x, n.rotation.y, n.rotation.z, n.rotation.w);
		mat4 S = mat4_scale(n.scale.x, n.scale.y, n.scale.z);
		return mat4_mul(T, mat4_mul(R, S)); // TRS
		};

	auto world_from_node = [&](S72::Node const& n) -> mat4 {
		// compute by walking parents using the node's .parent pointer if it exists.
		// If Node doesn't have parent, we compute via recursion from roots (fallback below).
		mat4 W = mat4_identity();

		// Fallback path: recompute from roots every time (fine for A1-show).
		// We’ll do a quick DFS until we hit this node.
		bool found = false;
		std::function<void(S72::Node const&, mat4 const&)> find;
		find = [&](S72::Node const& cur, mat4 const& parentW) {
			if (found) return;
			mat4 Wcur = mat4_mul(parentW, local_from_node(cur));
			if (&cur == &n) {
				W = Wcur;
				found = true;
				return;
			}
			for (S72::Node* ch : cur.children) if (ch) find(*ch, Wcur);
			};

		for (S72::Node* root : scene.scene.roots) if (root) find(*root, mat4_identity());
		return W;
		};

	 

	auto clip_from_orbit = [&](OrbitCamera const& cam) -> mat4 {
		return perspective(
			cam.fov,
			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height),
			cam.near, cam.far
		) * orbit(
			cam.target_x, cam.target_y, cam.target_z,
			cam.azimuth, cam.elevation, cam.radius
		);
		};

	if (camera_mode == CameraMode::Scene) {
		if (scene_camera_nodes.empty()) {
			//no cameras in scene -> fall back to user camera, full screen
			CLIP_FROM_WORLD = clip_from_orbit(free_camera);
			CLIP_FROM_CULL = CLIP_FROM_WORLD;

			scene_cam_aspect = 0.0f;
			compute_letterbox(0.0f);
		}
		else {
			active_scene_camera = std::min(active_scene_camera, uint32_t(scene_camera_nodes.size() - 1));
			S72::Node const& cam_node = *scene_camera_nodes[active_scene_camera];

			// Camera transform in world space:
			mat4 WORLD_FROM_CAMERA = world_from_node(cam_node);

			// View matrix:
			mat4 CAMERA_FROM_WORLD = mat4_inverse_rigid(WORLD_FROM_CAMERA);

			//--- pull camera params (fallback if missing):
			float vfov = 60.0f * float(M_PI) / 180.0f;
			float near_ = 0.1f;
			float far_ = 1000.0f;
			scene_cam_aspect = rtg.swapchain_extent.width / float(rtg.swapchain_extent.height);

			if (cam_node.camera) {
				//S72.hpp: camera->projection is std::variant< Perspective >
				if (auto const* persp = std::get_if<S72::Camera::Perspective>(&cam_node.camera->projection)) {
					vfov = persp->vfov;
					near_ = persp->near;
					far_ = persp->far;
					if (persp->aspect > 0.0f) scene_cam_aspect = persp->aspect;
				}
			}

			//optional safety clamp for far=infinity or weird values:
			if (!std::isfinite(far_) || far_ <= near_) far_ = 1000.0f;

			//letterbox based on camera aspect:
			compute_letterbox(scene_cam_aspect);

			CLIP_FROM_WORLD = perspective(vfov, scene_cam_aspect, near_, far_) * CAMERA_FROM_WORLD;
			CLIP_FROM_CULL = CLIP_FROM_WORLD;
		}
	}
	else if (camera_mode == CameraMode::User) {
		CLIP_FROM_WORLD = clip_from_orbit(free_camera);
		CLIP_FROM_CULL = CLIP_FROM_WORLD;

		//full-screen viewport/scissor:
		scene_cam_aspect = 0.0f;
		compute_letterbox(0.0f);
	}
	else if (camera_mode == CameraMode::Debug) {
		CLIP_FROM_WORLD = clip_from_orbit(debug_camera);

		if (debug_cull_locked) CLIP_FROM_CULL = debug_locked_CLIP_FROM_CULL;
		else CLIP_FROM_CULL = CLIP_FROM_WORLD;

		//full-screen viewport/scissor:
		scene_cam_aspect = 0.0f;
		compute_letterbox(0.0f);
	}



	{ //static sun and sky:
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = 1.0f;

		world.SKY_ENERGY.r = 0.1f;
		world.SKY_ENERGY.g = 0.1f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = 6.0f / 23.0f;
		world.SUN_DIRECTION.y = 13.0f / 23.0f;
		world.SUN_DIRECTION.z = 18.0f / 23.0f;

		world.SUN_ENERGY.r = 1.0f;
		world.SUN_ENERGY.g = 1.0f;
		world.SUN_ENERGY.b = 0.9f;
	}


	//
	//lines_vertices.reserve(4);
	/* { //make some crossing lines at different depths:
		lines_vertices.clear();


		const int N = 60;
		const float size = 1.0f;
		const float step = (2.0f * size) / float(N);
		size_t count = 4 * (N + 1) * N;
		lines_vertices.reserve(count);


		auto rippleY = [&](float x, float z) -> float {
			float d = std::sqrt(x * x + z * z);
			float y = std::sin(float(M_PI) * (4.0f * d - time));
			y /= (1.0f + 10.0f * d);
			return y;
			};

		// rows (z fixed, x changes)
		for (int zi = 0; zi <= N; ++zi) {
			float z = -size + zi * step;
			for (int xi = 0; xi < N; ++xi) {
				float x0 = -size + xi * step;
				float x1 = x0 + step;

				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x0, .y = rippleY(x0, z), .z = z },
					.Color{.r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff }
					});
				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x1, .y = rippleY(x1, z), .z = z },
					.Color{.r = 0xff, .g = 0xff, .b = 0x00, .a = 0xff }
					});
			}
		}

		// columns (x fixed, z changes)
		for (int xi = 0; xi <= N; ++xi) {
			float x = -size + xi * step;
			for (int zi = 0; zi < N; ++zi) {
				float z0 = -size + zi * step;
				float z1 = z0 + step;

				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x, .y = rippleY(x, z0), .z = z0 },
					.Color{.r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff }
					});
				lines_vertices.emplace_back(PosColVertex{
					.Position{.x = x, .y = rippleY(x, z1), .z = z1 },
					.Color{.r = 0x44, .g = 0x00, .b = 0xff, .a = 0xff }
					});
			}
		}

		assert(lines_vertices.size() == count);
	}*/

	{ //make some objects:
		object_instances.clear();

		if (!scene_file.empty()) {
			// build instances from S72 scene nodes/meshes
			auto local_from_node2 = [&](S72::Node const& n) -> mat4 {
				mat4 T = mat4_translate(n.translation.x, n.translation.y, n.translation.z);
				mat4 R = mat4_from_quat(n.rotation.x, n.rotation.y, n.rotation.z, n.rotation.w);
				mat4 S = mat4_scale(n.scale.x, n.scale.y, n.scale.z);
				return mat4_mul(T, mat4_mul(R, S));
				};


			std::function<void(S72::Node const&, mat4 const&)> emit_node;
			emit_node = [&](S72::Node const& n, mat4 const& parent_world) {
				mat4 WORLD_FROM_LOCAL = mat4_mul(parent_world, local_from_node2(n));


				if (n.mesh) {
					auto it = s72_mesh_to_range.find(n.mesh);
					if (it != s72_mesh_to_range.end() && it->second.count > 0) {

						// pick texture from mesh material (fallback to checker=0)
						uint32_t tex = 0;
						if (n.mesh->material) {
							auto itM = material_to_texture.find(n.mesh->material);
							if (itM != material_to_texture.end()) tex = itM->second;
						}

						object_instances.emplace_back(ObjectInstance{
							.vertices = it->second,
							.transform{
								.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
								.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
								.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL, // good enough for now
							},
							.texture = tex,
							});
					}
				}


				for (S72::Node* child : n.children) {
					if (child) emit_node(*child, WORLD_FROM_LOCAL);
				}
				};

			mat4 I = mat4_identity();
			for (S72::Node* root : scene.scene.roots) {
				if (root) emit_node(*root, I);
			}
			

		}
		else {
			//fallback: old hardcoded objects (optional)
			// (you can keep this branch if you want a non-scene demo mode)
			{ //plane translated +x by one unit:
				mat4 WORLD_FROM_LOCAL{
					1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					1.0f, 0.0f, 0.0f, 1.0f,
				};

				object_instances.emplace_back(ObjectInstance{
					.vertices = plane_vertices,
					.transform{
						.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
					},
					.texture = 1,
					});
			}
			{ //torus translated -x by one unit and rotated CCW around +y:
				float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
				float ca = std::cos(ang);
				float sa = std::sin(ang);
				mat4 WORLD_FROM_LOCAL{
					  ca, 0.0f,  -sa, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					  sa, 0.0f,   ca, 0.0f,
					-1.0f,0.0f, 0.0f, 1.0f,
				};

				object_instances.emplace_back(ObjectInstance{
					.vertices = torus_vertices,
					.transform{
						.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
						.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
					},
					});
			}

			{ //chen parts near origin
// if she’s huge/small, scale here later
				float s = 0.05f;

				mat4 WORLD_FROM_LOCAL{
					s,    0.0f, 0.0f, 0.0f,
					0.0f, s,    0.0f, 0.0f,
					0.0f, 0.0f, s,    0.0f,
					0.0f, -0.5f, 0.0f, 1.0f,
				};



				auto add_part = [&](ObjectVertices vr, uint32_t tex) {
					object_instances.emplace_back(ObjectInstance{
						.vertices = vr,
						.transform{
							.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
							.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
							.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
						},
						.texture = tex,
						});
					};

				add_part(chen_body_vertices, tex_body);
				add_part(chen_clothes_vertices, tex_clothes);
				add_part(chen_hairs_vertices, tex_hair);
				add_part(chen_face_vertices, tex_face);
				add_part(chen_iris_vertices, tex_iris);

				// optional:
				add_part(chen_sword_vertices, tex_sword);

			}


		
		}
	}
}
void Tutorial::on_input(InputEvent const &evt) {
	//if there is a current action, it gets input priority:
	if (action) {
		action(evt);
		return;
	}

	//general controls:
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB) {
		CameraMode prev = camera_mode;
		camera_mode = CameraMode((int(camera_mode) + 1) % 3);

		// if we are ENTERING debug, lock cull camera to whatever it was
		if (camera_mode == CameraMode::Debug && prev != CameraMode::Debug) {
			debug_cull_locked = true;
			debug_locked_CLIP_FROM_CULL = CLIP_FROM_CULL; // from last update()
		}

		// if we are LEAVING debug, unlock
		if (prev == CameraMode::Debug && camera_mode != CameraMode::Debug) {
			debug_cull_locked = false;
		}
		if (camera_mode == CameraMode::Scene) std::cout << "[A1-show] camera mode: Scene\n";
		if (camera_mode == CameraMode::User)  std::cout << "[A1-show] camera mode: User\n";
		if (camera_mode == CameraMode::Debug) std::cout << "[A1-show] camera mode: Debug\n";

		return;
	}


	// cycle scene cameras (only when in scene mode):
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_C) {
		if (camera_mode == CameraMode::Scene && !scene_camera_nodes.empty()) {
			active_scene_camera = (active_scene_camera + 1) % uint32_t(scene_camera_nodes.size());
			std::cout << "[A1-show] active scene camera: " << active_scene_camera << " / " << scene_camera_nodes.size() << "\n";
		}
		return;
	}

	
	// user/debug orbit controls
	if (camera_mode == CameraMode::User || camera_mode == CameraMode::Debug) {
		//free camera controls
		OrbitCamera& cam = (camera_mode == CameraMode::Debug ? debug_camera : free_camera);

		//This camera move is a "dolly" not a "zoom" because 
		//we're moving the camera's position, not changing its field of view.

		if (evt.type == InputEvent::MouseWheel) {
			//change distance by 10% every scroll click:
			cam.radius *= std::exp(std::log(1.1f) * -evt.wheel.y);
			//make sure camera isn't too close or too far from target:
			cam.radius = std::max(cam.radius, 0.5f * cam.near);
			cam.radius = std::min(cam.radius, 2.0f * cam.far);
			return;
		}

		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT
			&& (evt.button.mods & GLFW_MOD_SHIFT)) {
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = cam;

			action = [this, init_x, init_y, init_camera, &cam](InputEvent const& evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//image height at plane of target point:
					float height = 2.0f * std::tan(init_camera.fov * 0.5f) * init_camera.radius;

					//motion, therefore, at target point:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy = (evt.motion.y - init_y) / rtg.swapchain_extent.height * height; //note: negated because glfw uses y-down

					//compute camera transform to extract right (first row) and up (second row):
					mat4 camera_from_world = orbit(
						init_camera.target_x, init_camera.target_y, init_camera.target_z,
						init_camera.azimuth, init_camera.elevation, init_camera.radius
					);

					//move the desired distance:
					cam.target_x = init_camera.target_x - dx * camera_from_world[0] - dy * camera_from_world[1];
					cam.target_y = init_camera.target_y - dx * camera_from_world[4] - dy * camera_from_world[5];
					cam.target_z = init_camera.target_z - dx * camera_from_world[8] - dy * camera_from_world[9];

					return;
				}
			};
			return;
		}

		 
	


		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
			//start tumbling

			//std::cout << "Tumble started." << std::endl;

			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this, init_x,init_y,init_camera](InputEvent const& evt) {
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT) {
					//cancel upon button lifted:
					action = nullptr;

					//std::cout << "Tumble ended." << std::endl;
					return;
				}
				if (evt.type == InputEvent::MouseMotion) {
					//motion, normalized so 1.0 is window height:
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height; //note: negated because
					//glfw uses y-down coordinate system

					//rotate camera based on motion:
					float speed = float(M_PI); //how much rotation happens at one full window height
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f);
					//switch azimuth rotation when camera is upside-down
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					//reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					free_camera.azimuth -= std::round(free_camera.azimuth / twopi) * twopi;
					free_camera.elevation -= std::round(free_camera.elevation / twopi) * twopi;
					return;
				}
				};
			return;
		}
	}

}
	

