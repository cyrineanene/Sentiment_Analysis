pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout code from GitHub
                git 'https://github.com/cyrineanene/sentiment_analysis.git'
            }
        }
        stage('Execute Docker Compose') {
            steps {
                // Execute Docker Compose
                sh 'docker-compose up -d'
            }
        }
    }

    post {
        success {
            // Send success notification
            echo 'Model deployment successful!'
        }
        failure {
            // Send failure notification
            echo 'Model deployment failed!'
        }
    }
}
