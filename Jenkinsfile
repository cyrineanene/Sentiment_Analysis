pipeline {
    agent any

    stages {
        stage('Build Docker Images') {
            steps {
                script {
                    // Build Docker images for Python scripts
                    docker.build('python_scripts')
                }
            }
        }
        stage('Deploy') {
            steps {
                script {
                    // Run Docker Compose to deploy services
                    sh 'docker-compose -f docker-compose.yml up -d'
                }
            }
        }
    }

    post {
        always {
            // Clean up Docker containers after pipeline execution
            cleanDocker()
        }
    }
}

def cleanDocker() {
    // Stop and remove Docker containers
    sh 'docker-compose -f docker-compose.yml down'
}
