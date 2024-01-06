
## General Instructions
Unless otherwise specified, all commands listed in this documentation are intended to be run from the root of the project directory.
<br>
 
 ---------------------------------------------------- **DOCKER** ----------------------------------------------------

### 1) Build Docker images:

	# Use docker-compose to deal with multi-container Docker applications:
	🟢 docker-compose build

	# Can also be built separately: 
	🟢 docker build -f dockerfiles/train_model.dockerfile . -t trainer:latest
	🟢 docker build -f dockerfiles/predict_model.dockerfile . -t predict:latest

### 2) Create and run container instances from Docker images:
🟢 docker run --name train_instance trainer:latest  
🟢 docker run --name predict_instance predict:latest
<br>

### 3) Override entry point and run in interactive mode:
(usefull for debugging)
🟢 docker run -it --entrypoint sh trainer:latest
<br>

### 4) Run containers using docker-compose:
(so the volumes are mounted as specified in docker-compose.yaml)

🟢 docker-compose run --name train_instance trainer:latest  
🟢 docker-compose run --name predict_instance predict:latest  
