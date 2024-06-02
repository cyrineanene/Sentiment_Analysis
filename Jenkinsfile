pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'cyrine326/ysentiment_analysis_python_scripts:latest'
        KUBECONFIG = credentials('kubeconfig-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the source code from the GitHub repository
               checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/cyrineanene/sentiment_analysis.git']])
            }
        }

        stage('Run Tests') {
            steps {
                // Run unit tests using pytest
                sh 'pip install -r requirements.txt'
                sh 'pytest'
            }
        }

        stage('Model Training and Evaluation') {
            steps {
                // Trigger model training and evaluation
                sh 'python train_model.py'
            }
        }

        stage('Model Versioning') {
            steps {
                // Version control trained models and associated artifacts using MLflow
                sh 'mlflow experiments create --experiment-name my-experiment'
                sh 'mlflow run . --experiment-name my-experiment'
            }
        }

        stage('Build Docker Image') {
            steps {
                // Build the Docker image for the model serving application
                script {
                    docker.build("${DOCKER_IMAGE}")
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                // Deploy the Docker image to Kubernetes
                withKubeConfig(credentialsId: 'kubeconfig-credentials', serverUrl: 'https://your-kubernetes-cluster-url') {
                    sh 'kubectl apply -f kubernetes/deployment.yaml'
                    sh 'kubectl apply -f kubernetes/service.yaml'
                }
            }
        }

        stage('Monitoring with Prometheus and Grafana') {
            steps {
                // Set up Prometheus and Grafana
                // Configuration steps for Prometheus and Grafana go here
            }
        }
    }

    post {
        always {
            // Cleanup steps (if any) go here
        }
    }
}



// pipeline {
//     agent any
//     environment {
//         IMAGE_NAME = 'sentiment_analysis_python_scripts'
//         DOCKER_REGISTRY = 'cyrine326/sentiment_analysis'
//     }
//     stages {
//         stage('Checkout') {
//             steps {
//                 // Pull the code from the repository
//                 checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/cyrineanene/sentiment_analysis.git']]) 
//             }
//         }
//         stage('Build') {
//             steps {
//                 script {
//                    script {
//                     // Change directory to where docker-compose.yml is located
//                     dir('./') {
//                         // Run docker-compose build command
//                         sh 'docker-compose build'
//                     }
//                    }
//                 }
//             }
//         }
//         // stage('Test') {
//         //     steps {
//         //         script {
//         //             // Run tests inside the Docker container
//         //             docker.image("${DOCKER_REGISTRY}/${IMAGE_NAME}:latest").inside {
//         //                 sh 'python -m unittest discover -s tests'
//         //             }
//         //         }
//         //     }
//         // }
//         stage('Push to DockerHub') {
//             steps {
//                 script {
//                     // Push Docker image to the registry
//                    withCredentials([string(credentialsId: 'dockerhubpass', variable: 'dockerhubpass')]) {
//                     sh 'docker login -u cyrine326 -p ${dockerhubpass}'
// }
//                     sh 'docker push cyrine326/sentiment_analysis:latest'
//                     }
//                 }
//             }
        
//         stage('Deploy to Kubernetes') {
//             steps {
//                 script {
//                     // Deploy to Kubernetes
//                     kubernetesDeploy(
//                         configs: 'k8s-deployment.yml',
//                         kubeConfig: [path: './deployment.yml']
//                     )
//                 }
//             }
//         }
//     }
// }
