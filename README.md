##LeoVegas test task

- docs - directory with files for API endpoints documentation (Swagger)
- Dockerfile - build instructions for Docker Image which will be uploaded to the Cloud Registry (Google Cloud Platforom)
- LeoVegas.ipynb - the main document describing the process of working with data, training the model, development and deployment of the service
- requirements.txt - list of used libraries (needed to build Docker Image correctly)
- service_.py - Flask service that runs in Cloud Run and is available at https://my-flask-app3-bga4ajkzpa-uc.a.run.app/apidocs/ (it can also run locally if you have all the libraries installed in requirements)
- test_endpoints.py - autotests to check if the API works correctly
