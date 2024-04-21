pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git 'https://github.com/cyrineanene/sentiment_analysis'
            }
        }
        
        stage('Run Docker Compose') {
            steps {
                script {
                    sh 'docker-compose up -d'
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                // Implement your testing scripts here
                // This can be a health check or API response test
            }
        }
        
        stage('Deploy Model') {
            steps {
                // Steps to deploy the model, could be on a remote server or a cloud environment
                script {
                    // Example of scp command or could use any deployment script
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                script {
                    sh 'docker-compose down'
                }
            }
        }
    }
    
    post {
        always {
            // Actions to perform after pipeline completion, successful or not
            echo 'Pipeline execution complete!'
        }
    }
}
