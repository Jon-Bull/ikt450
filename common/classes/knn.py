import numpy as np

class KNN:
    """
    A class to represent the K-Nearest Neighbors (KNN) algorithm.
    
    Attributes:
    -----------
    k : int
        The number of nearest neighbors to consider for voting.
    X_train : np.ndarray
        The training input samples.
    Y_train : np.ndarray
        The training target values.

    Methods:
    --------
    fit(X_train, Y_train):
        Stores the training data.
    
    distance(one, two):
        Calculates the Euclidean distance between two points.
    
    get_k_nearest_neighbors(x):
        Finds the k-nearest neighbors for a given input sample.
    
    predict_class(x):
        Predicts the class for a single instance.
    
    predict(X_val):
        Predicts the class labels for the validation set.
    
    calculate_metrics(Y_true, Y_pred):
        Calculates and returns accuracy, precision, recall, and F1 score.
    
    calculate_mse(Y_true, Y_pred):
        Calculates the Mean Squared Error (MSE) between true and predicted labels.
    
    calculate_mae(Y_true, Y_pred):
        Calculates the Mean Absolute Error (MAE) between true and predicted labels.
    
    evaluate(X_val, Y_val, return_metrics=["accuracy", "precision", "recall", "f1", "mse", "rmse", "mae"]):
        Evaluates the KNN model on the validation set, returning the specified metrics.
    """

    def __init__(self, k=3):
        """
        Initializes the KNN classifier with a specified number of neighbors.
        
        Parameters:
        -----------
        k : int
            The number of nearest neighbors to consider for voting (default is 3).
        """
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Stores the training data.

        Parameters:
        -----------
        X_train : np.ndarray
            The training input samples.
        Y_train : np.ndarray
            The training target values.
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def distance(self, one, two):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
        -----------
        one : np.ndarray
            The first point.
        two : np.ndarray
            The second point.

        Returns:
        --------
        float
            The Euclidean distance between the two points.
        """
        return np.linalg.norm(one - two)

    def get_k_nearest_neighbors(self, x):
        """
        Finds the k-nearest neighbors for a given input sample.

        Parameters:
        -----------
        x : np.ndarray
            The input sample to classify.

        Returns:
        --------
        np.ndarray
            The labels of the k-nearest neighbors.
        """
        distances = [self.distance(x, self.X_train[i]) for i in range(len(self.X_train))]
        sorted_indices = np.argsort(distances)
        top_k_indices = sorted_indices[:self.k]
        top_k_labels = self.Y_train[top_k_indices]
        return top_k_labels

    def predict_class(self, x):
        """
        Predicts the class for a single instance.

        Parameters:
        -----------
        x : np.ndarray
            The input sample to classify.

        Returns:
        --------
        int
            The predicted class label.
        """
        neighbors = self.get_k_nearest_neighbors(x)
        return np.argmax(np.bincount(neighbors.astype(int)))

    def predict(self, X_val):
        """
        Predicts the class labels for the validation set.

        Parameters:
        -----------
        X_val : np.ndarray
            The validation input samples.

        Returns:
        --------
        np.ndarray
            The predicted class labels for the validation set.
        """
        return np.array([self.predict_class(x) for x in X_val])

    def calculate_precision(self, TP, FP):
        """
        Calculates precision.

        Parameters:
        -----------
        TP : int
            True Positives.
        FP : int
            False Positives.

        Returns:
        --------
        float
            The precision score.
        """
        return TP / (TP + FP)

    def calculate_recall(self, TP, FN):
        """
        Calculates recall.

        Parameters:
        -----------
        TP : int
            True Positives.
        FN : int
            False Negatives.

        Returns:
        --------
        float
            The recall score.
        """
        return TP / (TP + FN)

    def calculate_accuracy(self, TP, TN, FP, FN):
        """
        Calculates accuracy.

        Parameters:
        -----------
        TP : int
            True Positives.
        TN : int
            True Negatives.
        FP : int
            False Positives.
        FN : int
            False Negatives.

        Returns:
        --------
        float
            The accuracy score.
        """
        return (TP + TN) / (TP + TN + FP + FN)

    def calculate_f1(self, precision, recall):
        """
        Calculates F1 score.

        Parameters:
        -----------
        precision : float
            The precision score.
        recall : float
            The recall score.

        Returns:
        --------
        float
            The F1 score.
        """
        # Ref. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ16
        return 2 * (precision * recall) / (precision + recall)

    def calculate_mse(self, Y_true, Y_pred):
        """
        Calculates Mean Squared Error (MSE).

        Parameters:
        -----------
        Y_true : np.ndarray
            The true labels.
        Y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The Mean Squared Error.
        """
        return np.mean((Y_true - Y_pred) ** 2)

    def calculate_mae(self, Y_true, Y_pred):
        """
        Calculates Mean Absolute Error (MAE).

        Parameters:
        -----------
        Y_true : np.ndarray
            The true labels.
        Y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The Mean Absolute Error.
        """
        return np.mean(np.abs(Y_true - Y_pred))

    def calculate_metrics(self, Y_true, Y_pred):
        """
        Calculates accuracy, precision, recall, and F1 score.

        Parameters:
        -----------
        Y_true : np.ndarray
            The true labels.
        Y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        tuple
            A tuple containing accuracy, precision, recall, and F1 score.
        """
        TP = np.sum((Y_true == 1) & (Y_pred == 1))
        TN = np.sum((Y_true == 0) & (Y_pred == 0))
        FP = np.sum((Y_true == 0) & (Y_pred == 1))
        FN = np.sum((Y_true == 1) & (Y_pred == 0))
        
        accuracy = self.calculate_accuracy(TP, TN, FP, FN)
        precision = self.calculate_precision(TP, FP)
        recall = self.calculate_recall(TP, FN)
        f1 = self.calculate_f1(precision, recall)
        
        return accuracy, precision, recall, f1

    def evaluate(self, X_val, Y_val, return_metrics=["accuracy", "precision", "recall", "f1", "mse", "rmse", "mae"]):
        """
        Evaluates the KNN model on the validation set.

        Parameters:
        -----------
        X_val : np.ndarray
            The validation input samples.
        Y_val : np.ndarray
            The true labels for the validation set.
        return_metrics : list of str, optional
            List of metrics to return. Defaults to all metrics.

        Returns:
        --------
        dict
            A dictionary containing the requested metrics.
        """
        Y_pred = self.predict(X_val)
        results = {}

        if "accuracy" in return_metrics:
            accuracy, precision, recall, f1 = self.calculate_metrics(Y_val, Y_pred)
            results["accuracy"] = accuracy
            results["precision"] = precision
            results["recall"] = recall
            results["f1"] = f1
        
        if "mse" in return_metrics or "rmse" in return_metrics:
            mse = self.calculate_mse(Y_val, Y_pred)
            results["mse"] = mse
            if "rmse" in return_metrics:
                results["rmse"] = np.sqrt(mse)
        
        if "mae" in return_metrics:
            mae = self.calculate_mae(Y_val, Y_pred)
            results["mae"] = mae

        return results
