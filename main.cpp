
#include "RTG.hpp"

#include "Tutorial.hpp"

#include <iostream>

int main(int argc, char **argv) {
	//main wrapped in a try-catch so we can print some debug info about uncaught exceptions:
	try {
		//parse -scene and forward rest to rtg
		std::string scene_file;

		//build argv for RTG :
		std::vector<char*> rtg_argv;
		rtg_argv.reserve(argc);
		rtg_argv.push_back(argv[0]);

		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];

			if (arg == "--scene") {
				if (i + 1 >= argc) {
					throw std::runtime_error("--scene requires a path to a .s72 file.");
				}
				scene_file = argv[i + 1];
				++i; //skip the filename
			}
			else {
				//not ours pass through to RTG
				rtg_argv.push_back(argv[i]);
			}
		}

		bool print_usage = false;

		if (scene_file.empty()) {
			throw std::runtime_error("Missing required argument: --scene <file.s72>");
			print_usage = true;
		}
		//configure application:
		RTG::Configuration configuration;

		configuration.application_info = VkApplicationInfo{
			.pApplicationName = "A1 Scene Loader",
			.applicationVersion = VK_MAKE_VERSION(0,0,0),
			.pEngineName = "Unknown",
			.engineVersion = VK_MAKE_VERSION(0,0,0),
			.apiVersion = VK_API_VERSION_1_3
		};


		try {
			configuration.parse((int)rtg_argv.size(), rtg_argv.data());
		} catch (std::runtime_error &e) {
			std::cerr << "Failed to parse arguments:\n" << e.what() << std::endl;
			print_usage = true;
		}

		if (print_usage) {
			std::cerr << "Usage:" << std::endl;
			std::cerr << "    bin/viewer.exe --scene <file.s72> [RTG options]\n" << std::endl;
			std::cerr << "RTG options:" << std::endl;
			RTG::Configuration::usage( [](const char *arg, const char *desc){ 
				std::cerr << "    " << arg << "\n        " << desc << std::endl;
			});
			return 1;
		}

		//loads vulkan library, creates surface, initializes helpers:
		RTG rtg(configuration);

		//initializes global (whole-life-of-application) resources:
		Tutorial application(rtg, scene_file); //scene file gets passed into Tutorial

		//main loop -- handles events, renders frames, etc:
		rtg.run(application);

	} catch (std::exception &e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return 1;
	}
}
