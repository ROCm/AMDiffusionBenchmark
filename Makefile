# Check if .env exists, if not, use .env.example
ifneq ("$(wildcard .env)","")
	include .env
else
	include .env.example
endif

HF_HOME ?= ~/.cache/huggingface

-include local.mk # include custom makefile

.PHONY: download_assets build_docker launch_docker build_and_launch_docker exec_docker stop_docker submit_launcher_job run_tests

# Target to download the assets
download_assets:
	# login into HF if needed
	@if [ ! -f "$(HF_HOME)/token" ]; then \
		echo "You are not logged in. Please log in to Hugging Face."; \
		if [ -n "$(HF_TOKEN)" ]; then \
			echo "Logging in with the provided token"; \
			huggingface-cli login --token $(HF_TOKEN); \
		else \
			huggingface-cli login; \
		fi \
	fi
	# download pseudo-camera-10k dataset
	@echo "\033[1;31mDownloading frank-chieng/chinese_architecture_siheyuan dataset\033[0m"
	huggingface-cli download --repo-type=dataset frank-chieng/chinese_architecture_siheyuan

	# download FLUX.1-dev model and checkpoints
	@echo "\033[1;31mDownloading black-forest-labs/FLUX.1-dev\033[0m"
	huggingface-cli download black-forest-labs/FLUX.1-dev

	# download stable-diffusion-xl model and checkpoints
	@echo "\033[1;31mDownloading stabilityai/stable-diffusion-xl-base-1.0\033[0m"
	huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0

	# download Disney-VideoGeneration-Dataset
	@echo "\033[1;31mDownloading Wild-Heart/Disney-VideoGeneration-Dataset\033[0m"
	huggingface-cli download --repo-type=dataset Wild-Heart/Disney-VideoGeneration-Dataset

	# download HunyuanVideo model and checkpoints
	@echo "\033[1;31mDownloading hunyuanvideo-community/HunyuanVideo\033[0m"
	huggingface-cli download hunyuanvideo-community/HunyuanVideo

	# download Mochi-1 model and checkpoints
	@echo "\033[1;31mDownloading genmo/mochi-1-preview\033[0m"
	huggingface-cli download genmo/mochi-1-preview

	# download Wan2.1 model and checkpoints
	@echo "\033[1;31mDownloading Wan-AI/Wan2.1-I2V-14B-480P-Diffusers\033[0m"
	huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers

	@echo "\033[1;31mDownloading completed.\033[0m"

# Target to build the Docker image
build_docker:
	@echo "\033[1;31mBuilding Docker image with name: $(IMAGE_NAME)\033[0m"
	@if [ "$$(docker ps -a -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "\033[1;31mContainer $(CONTAINER_NAME) is already running.\033[0m"; \
		echo -n "Do you want to stop the container? [y/N]: "; \
		read answer; \
		case $$answer in \
			[Yy]*) \
				$(MAKE) stop_docker; \
				;; \
			*) \
				echo "Cannot proceed with building a new container while the existing one is running."; \
				exit 1; \
				;; \
		esac; \
	fi
	docker build -f Dockerfile.rocm -t $(IMAGE_NAME) .

# Target to launch the Docker container
launch_docker:
	@echo "\033[1;31mLaunching Docker container\033[0m"
	CONTAINER_NAME=$(CONTAINER_NAME) \
	IMAGE_NAME=$(IMAGE_NAME) \
	VOLUME_MOUNT=$(VOLUME_MOUNT) \
	bash utility_scripts/launch_docker_container.sh

# Target to build and then launch the Docker container
build_and_launch_docker:
	@$(MAKE) build_docker
	@$(MAKE) launch_docker
	@echo "\033[1;31mBuild and launch process completed.\033[0m"

# Target to execute a shell inside the running Docker container
exec_docker:
	docker exec -it $(CONTAINER_NAME) bash -c '\
		echo -e "\n\n\033[1;31mEntered Docker container: $(CONTAINER_NAME)\033[0m"; \
		exec bash \
	'

# Target to stop the Docker container
stop_docker:
	@if [ "$$(docker ps -a -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "\033[1;31mStopping existing container: $(CONTAINER_NAME)\033[0m"; \
		docker stop $(CONTAINER_NAME) 2>/dev/null || true; \
	else \
		echo "\033[1;31mNo existing container found: $(CONTAINER_NAME)\033[0m"; \
	fi

run_tests:
	# Run tests cross-platform (Windows/Linux)
	if [ "$$(uname)" = "Windows_NT" ]; then \
		set PYTHONPATH=. && pytest ./tests; \
	else \
		PYTHONPATH=. pytest ./tests; \
	fi
