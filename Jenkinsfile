pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                // Checkout from version control
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script{
                    sh 'sudo docker build -t star_generator .'
                }
            }
        }

        stage('Push Image to Dockerhub') {
            steps {
               script{
                withCredentials([string(credentialsId: 'dockerhub-pwd', variable: 'dockerhubpwd')]) {
                sh 'sudo docker login -u cyrine326 -p ${dockerhubpwd}'
}
                sh 'sudo docker push cyrine236/star_generator'
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
