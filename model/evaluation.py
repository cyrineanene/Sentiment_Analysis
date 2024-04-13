from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, mean_squared_error, r2_score

class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        self.f1 = f1_score(self.y_true, self.y_pred, average='binary')  # F1 score (binary)
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

    def print_metrics(self):
        print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.confusion_matrix)
        print("Accuracy:", self.accuracy)