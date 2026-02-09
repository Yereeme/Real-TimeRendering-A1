#pragma once

#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "mat4.hpp"
#include <string>

#include "RTG.hpp"
#include <unordered_map>

struct Tutorial : RTG::Application {

	//Tutorial(RTG &);
	Tutorial(RTG& rtg_, std::string const& scene_file_);
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	//kept for use in destructor:
	RTG &rtg;
	std::string scene_file;
	//--------------------------------------------------------------------
	//Resources that last the lifetime of the application:

	//chosen format for depth buffer:
	VkFormat depth_format{};
	//Render passes describe how pipelines write to images:
	VkRenderPass render_pass = VK_NULL_HANDLE;

	//Pipelines:
	//...
	//adding BackgroundPipeline member structure to Tutorial.cppp

	struct BackgroundPipeline {
		//no descriptor set layouts
		// 
		// push constants
		struct Push { //cpu side description of what we are pushing to the shader
			float time;
		};

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//no vertex bindings

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	} background_pipeline;
	//...
	//LinesPipeline for grid
	struct LinesPipeline {
		 
		//descriptor set layouts:
		VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;

		//types for descriptors:
		struct Camera {
			mat4 CLIP_FROM_WORLD;
		};
		static_assert(sizeof(Camera) == 16 * 4, "Camera buffer structure is packed");
		// push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//we have vertex bindings now!
		using Vertex = PosColVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG&, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG&);
	} lines_pipeline;

	//ObjectsPipeline  
	struct ObjectsPipeline {

		//descriptor set layouts:
 
		VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;


		//types for descriptors:
		struct World {
			struct { float x, y, z, padding_; } SKY_DIRECTION;
			struct { float r, g, b, padding_; } SKY_ENERGY;
			struct { float x, y, z, padding_; } SUN_DIRECTION;
			struct { float r, g, b, padding_; } SUN_ENERGY;
		};
		static_assert(sizeof(World) == 4 * 4 + 4 * 4 + 4 * 4 + 4 * 4, "World is the expected size.");

		//using Camera = LinesPipeline::Camera;
		struct Transform { //storage buffer descriptor set layout
			mat4 CLIP_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL;
			mat4 WORLD_FROM_LOCAL_NORMAL;
		};
		static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform is the expected size.");
	 
 
		// no push constants

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//we have vertex bindings 

		using Vertex = PosNorTexVertex;

		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG&, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG&);
	} objects_pipeline;
	//pools from which per-workspace things are allocated:
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

	//workspaces hold per-render resources:
	struct Workspace {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; //from the command pool above; reset at the start of every render.

		//location for lines data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer lines_vertices_src; //host coherent; mapped
		Helpers::AllocatedBuffer lines_vertices; //device-local

		//location for LinesPipeline::Camera data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Camera_src; //host coherent; mapped
		Helpers::AllocatedBuffer Camera; //device-local
		VkDescriptorSet Camera_descriptors; //references Camera

		//location for ObjectsPipeline::World data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer World_src; //host coherent; mapped
		Helpers::AllocatedBuffer World; //device-local
		VkDescriptorSet World_descriptors; //references World

		//location for ObjectsPipeline::Transforms data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Transforms_src; //host coherent; mapped
		Helpers::AllocatedBuffer Transforms; //device-local
		VkDescriptorSet Transforms_descriptors; //references Tranforms


	};
	std::vector< Workspace > workspaces;

	//-------------------------------------------------------------------
	//static scene resources:

	Helpers::AllocatedBuffer object_vertices;
	//store the index of the first vertex and the count of vertices (parameters used by vkCmdDraw) for each 
	//mesh stored in obj vertices arraay
	struct ObjectVertices {
		uint32_t first = 0;
		uint32_t count = 0;
	};
	ObjectVertices plane_vertices;
	ObjectVertices torus_vertices;
	ObjectVertices crystal_vertices;
	ObjectVertices chen_sword_vertices;
	ObjectVertices chen_body_vertices;
	ObjectVertices chen_clothes_vertices;
	ObjectVertices chen_face_vertices;
	ObjectVertices chen_hairs_vertices;
	ObjectVertices chen_iris_vertices;

	//uint32_t character_texture = 0;

	std::vector< Helpers::AllocatedImage > textures;
	std::vector< VkImageView > texture_views;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
	std::vector< VkDescriptorSet > texture_descriptors; //allocated from texture_descriptor_pool
	//maps a loaded material texture to our textures[] index:
	std::unordered_map<std::string, uint32_t> texture_lookup;//cpu side metadata
	uint32_t tex_body = 0;
	uint32_t tex_clothes = 0;
	uint32_t tex_hair = 0;
	uint32_t tex_face = 0;
	uint32_t tex_iris = 0;
	uint32_t tex_sword = 0;

	//--------------------------------------------------------------------
	//Resources that change when the swapchain is resized:

	virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

	Helpers::AllocatedImage swapchain_depth_image;
	VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
	std::vector< VkFramebuffer > swapchain_framebuffers;
	//used from on_swapchain and the destructor: (framebuffers are created in on_swapchain)
	void destroy_framebuffers();

	//--------------------------------------------------------------------
	//Resources that change when time passes or the user interacts:

	virtual void update(float dt) override;
	virtual void on_input(InputEvent const &) override;

	//modal action, interceps inputs:
	std::function< void(InputEvent const&) > action;

		//global variable
	float time = 0.0f;

	struct OrbitCamera {
		float target_x = 20.5f, target_y = 0.0f, target_z = 0.0f; //where the camera is 
		//looking + orbiting
		float radius = 2.0f; //distance from camera to target
		float azimuth = 0.0f; //counterclockwise angle around z axis between x axis and camera direction
		//(radians)
		float elevation = 0.25f * float(M_PI); //angle up from xy plane to camera direction (radians)

		float fov = 60.0f / 180.0f * float(M_PI); //vertical field of view (radians)
		float near = 0.1f; //near clipping plane
		float far = 1000.0f; //far clipping plane
	} free_camera;

	//for selecting between cameras:
	enum class CameraMode {
		Scene = 0,
		Free = 1,
	} camera_mode = CameraMode::Free;

	//computed from the current camera (as set by camera_mode) during update():
	mat4 CLIP_FROM_WORLD; //matrix through which to view grid line

	std::vector< LinesPipeline::Vertex > lines_vertices;

	ObjectsPipeline::World world;

	struct ObjectInstance {
		ObjectVertices vertices;
		ObjectsPipeline::Transform transform;
		uint32_t texture = 0;
		

	};
	std::vector< ObjectInstance > object_instances;
	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;
};
