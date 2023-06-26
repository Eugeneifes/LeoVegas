**LeoVegas test task**

- docs - directory with files for API endpoints documentation (Swagger)
- Dockerfile - build instructions for Docker Image which will be uploaded to the Cloud Registry (Google Cloud Platforom)
- LeoVegas.ipynb - the main document describing the process of working with data, training the model, development and deployment of the service
- deposit_next_NN.pt - trained Torch model instance
- requirements.txt - list of used libraries (needed to build Docker Image correctly)
- test_endpoints.py - autotests to check if the API works correctly
- service_.py - Flask service that runs in Cloud Run and is available at https://my-flask-app3-bga4ajkzpa-uc.a.run.app/apidocs/ (it can also be run locally if you have all the libraries installed in requirements)
  

**Endpoints**

 - **/add_data** - to add data on which the model has not been trained to a separate file (**tenure**, **deposit**, **turnover**, **withdrawal** and **deposit_next** values should be passed)
- **/evaluate_model** - to evaluate the performance of the model on unseen data (new data can be added to a separate file using **/add_data** endpoint as mentioned above)
- **/health_check** - to verify if the service is running and accessible, as well as to know the current version of the deployed model. In addition I display information about the Loss metric on the test sample, so that it is possible to compare it with the metric value on the new data and identify the moment when the model begins to degrade (at this point it is worth retraining)
- **/predict** - the basic method that allows you to predict the value of the **deposit_next** based on **tenure**, **deposit**, **turnover** and **withdrawal**
- **/retrain** - allows users to trigger the retraining of the model (update the model with new data, that could be added by **/add_data**). As mentioned above we can determine when to retrain the model based on the Loss metric on the test sample and compare it to the Loss metric on the new data
  

**Ideas for the future**

- set up a trigger alert in Google Cloud which will indicate the degradation moment of the model (e.g., the Loss metric on new data is below the threshold for some time). Notification can be sent to a convenient channel: email, google chat, etc.
- implementation of a mechanism for switching between model versions on demand (for example, traffic management scenario as part of an A/B test)
