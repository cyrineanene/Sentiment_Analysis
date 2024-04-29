pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'sentiment_analysis-python_scripts'
    }
    
    stages {
        stage('Checkout Code') {
        steps {
                // Checkout from version control
            checkout scm
            }
        }
        stage('Build') {
            steps {
                script {
                    docker.build(env.DOCKER_IMAGE)
                }
            }
        }
        
        stage('Test') {
            steps {
                // Add any testing steps here
            }
        }
        
        stage('Push to Docker Registry') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'dockerhub-pwd', usernameVariable: 'cyrine326', passwordVariable: 'dockerhubpwd')]) {
                        docker.withRegistry('https://hub.docker.com/repository/docker/cyrine326/sentiment_analysis/general', DOCKER_USERNAME, DOCKER_PASSWORD) {
                            docker.image(env.DOCKER_IMAGE).push('latest')
                        }
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    // Run the Docker image on your deployment environment
                    sh 'docker run -d -p <HOST_PORT>:<CONTAINER_PORT> --name sentiment-analysis sentiment-analysis:latest'
                }
            }
        }
    }
}











// pipeline {
//     agent any

//     stages {
//         stage('Checkout Code') {
//             steps {
//                 // Checkout from version control
//                 checkout scm
//             }
//         }

//         stage('Build Docker Image') {
//             steps {
//                 script{
//                     sh 'docker build -t star_generator .'
//                 }
//             }
//         }

//         stage('Push Image to Dockerhub') {
//             steps {
//                script{
//                 withCredentials([string(credentialsId: 'dockerhub-pwd', variable: 'dockerhubpwd')]) {
//                 sh 'docker login -u cyrine326 -p ${dockerhubpwd}'
// }
//                 sh 'docker push cyrine236/star_generator'
//                }
//             }
//         }
//     }
// }

       
//         stage('Run Docker Compose') {
//             steps {
//                 script {
//                     sh 'docker-compose up -d'
//                 }
//             }
//         }
 
//         stage('Cleanup') {
//             steps {
//                 script {
//                     sh 'docker-compose down'
//                 }
//             }
//         }
//     }
    
//     post {
//         always {
//             // Actions to perform after pipeline completion, successful or not
//             echo 'Pipeline execution complete!'
//         }
//     }
// }
