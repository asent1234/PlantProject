# PlantProject: Flask Plant Disease Detection & Chatbot

## Description
A Flask web app to detect plant diseases from images and provide plant care recommendations. Includes user authentication, upload history, and a plant Q&A chatbot.

## Local Development
1. Create a virtual environment and activate it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file (see below).
4. Run locally:
   ```
   python application.py
   ```

## Deployment to AWS Lightsail Container Service

### Prerequisites
1. AWS Account with Lightsail access
2. Docker installed on your development machine
3. Docker Hub account (or other container registry)

### Local Testing
1. Build the Docker image locally:
   ```
   docker build -t plant-app-test .
   ```
2. Run the container locally to test:
   ```
   docker run -p 8080:8080 --env-file .env plant-app-test
   ```
3. Visit http://localhost:8080 to verify the application works properly
4. Test the health check endpoint: http://localhost:8080/health

### Deployment Process
1. Run the provided deployment script to build and push your image:
   ```powershell
   ./deploy.ps1
   ```
   This will build and push the image to Docker Hub at `aakashsenthilnathan/plant-app:latest`

2. In AWS Lightsail Console:
   - Create a new container service (recommended: Micro or Small tier)
   - Choose the "Specify a custom deployment" option
   - Container name: `plant-app`
   - Image: `aakashsenthilnathan/plant-app:latest`
   - Port: `8080`
   - Set your environment variables in the "Variables" section

3. Monitor deployment in the Lightsail Console
   - For troubleshooting view container logs
   - Use the "Health checks" tab to verify your container is responding

### Troubleshooting Deployment
- If deployment times out, check the container logs in Lightsail
- Verify the application is binding to 0.0.0.0:8080
- Ensure your health check endpoint is responding within timeout limits
- For memory issues, consider adding swap to your container or using a larger container size

## Environment Variables
- SECRET_KEY
- SQLALCHEMY_DATABASE_URI
- Any API keys needed for external services

## Notes
- For production, set `debug=False` in `application.py` and use a persistent database.
- Ensure model weights (`ConvNextAIPredictionModel/plant_disease_convnext_best.pth`) are included.
- Static and template folders must be present.
- If using AWS Lambda, consider Zappa or AWS Lambda Powertools for Flask.
