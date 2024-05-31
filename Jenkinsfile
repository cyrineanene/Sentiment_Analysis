// pipeline {
//     agent any

//     stages {
//         stage('Checkout') {
//             steps {
//                 // Checkout code from GitHub
//                 checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/cyrineanene/sentiment_analysis.git']])
//             }
//         }
//         stage('Execute Docker Compose') {
//             steps {
//                 // Execute Docker Compose
//                 sh 'docker-compose build'
//             }
//         }
//         stage('Push to DockerHub') {
//             steps {
//                 // Execute Docker Compose
//                 sh 'docker-compose up -d'
//             }
//         }
//         stage('Deploy') {
//             steps {
//                 // Execute Docker Compose
//                 sh 'docker-compose up -d'
//             }
//         }
//     }
// }


pipeline {
    agent any
    environment {
        IMAGE_NAME = 'sentiment_analysis_python_scripts'
        DOCKER_REGISTRY = 'cyrine326/sentiment_analysis'
    }
    stages {
        stage('Checkout') {
            steps {
                // Pull the code from the repository
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/cyrineanene/sentiment_analysis.git']]) 
            }
        }
        stage('Build') {
            steps {
                script {
                   script {
                    // Change directory to where docker-compose.yml is located
                    dir('./') {
                        // Run docker-compose build command
                        sh 'docker-compose build'
                    }
                   }
                }
            }
        }
        // stage('Test') {
        //     steps {
        //         script {
        //             // Run tests inside the Docker container
        //             docker.image("${DOCKER_REGISTRY}/${IMAGE_NAME}:latest").inside {
        //                 sh 'python -m unittest discover -s tests'
        //             }
        //         }
        //     }
        // }
        stage('Push to DockerHub') {
            steps {
                script {
                    // Push Docker image to the registry
                   withCredentials([string(credentialsId: 'dockerhubpass', variable: 'dockerhubpass')]) {
                    sh 'docker login -u cyrine326 -p ${dockerhubpass}'
}
                    sh 'docker push sentiment_analysis_python_scripts'
                    }
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Deploy to Kubernetes
                    kubernetesDeploy(
                        configs: 'k8s-deployment.yml',
                        kubeConfig: [path: './deployment.yml']
                    )
                }
            }
        }
    }
}
