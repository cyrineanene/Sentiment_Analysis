pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                checkout scmGit(branches: [[name: '*/test']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/cyrineanene/sentiment_analysis']])
            }
        }

        stage('Build Docker Image') {
            steps {
                script{
                    sh 'docker build -t sentiment_models '
                }
            }
        }
    }
}        
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
