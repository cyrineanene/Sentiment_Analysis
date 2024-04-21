pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                // Checkout from version control
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies from requirements.txt...'
                sh 'pip install -r requirement.txt'

            }
        }

        stage('Run Kafka Producer') {
            steps {
                echo 'Running Kafka Producer...'
                sh 'python kafka_producer.py'
            }
        }
        
        stage('Run Consumer Star Generator') {
            steps {
                echo 'Running Consumer Star Generator...'
                sh 'python consumer_star_generator.py'
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            // Add any post-build clean up here if necessary
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
