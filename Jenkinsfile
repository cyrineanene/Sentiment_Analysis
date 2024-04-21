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
                sh 'docker build -t star_generator .'

            }
        }

        stage('Running Docker Image') {
            steps {
               sh 'docker run star_generator'
            }
        }

        // stage('Stopping Docker Image') {
        //     steps {
        //        sh 'docker stop star_generator'
        //     }
        // }
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
