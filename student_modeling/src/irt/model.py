"""IRT (Item Response Theory) model - 1PL Rasch model."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class IRTModel:
    """1PL (Rasch) Item Response Theory model.

    P(correct | theta, b) = sigmoid(theta - b)

    where:
    - theta: student ability
    - b: item difficulty
    """

    def __init__(
        self,
        n_students: int,
        n_items: int,
        learning_rate: float = 0.1,
        regularization: float = 0.01,
    ):
        """Initialize IRT model.

        Args:
            n_students: Number of students.
            n_items: Number of items.
            learning_rate: Learning rate for gradient descent.
            regularization: L2 regularization strength.
        """
        self.n_students = n_students
        self.n_items = n_items
        self.learning_rate = learning_rate
        self.regularization = regularization

        # Initialize parameters
        self.theta = np.zeros(n_students)  # Student abilities
        self.b = np.zeros(n_items)  # Item difficulties

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def predict_proba(
        self,
        student_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> np.ndarray:
        """Predict probability of correct response.

        Args:
            student_indices: Array of student indices.
            item_indices: Array of item indices.

        Returns:
            Array of probabilities.
        """
        theta = self.theta[student_indices]
        b = self.b[item_indices]
        return self.sigmoid(theta - b)

    def fit(
        self,
        student_indices: np.ndarray,
        item_indices: np.ndarray,
        correct: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> dict:
        """Fit model using gradient descent.

        Args:
            student_indices: Array of student indices.
            item_indices: Array of item indices.
            correct: Array of correct/incorrect (0/1).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Training history dict.
        """
        history = {"loss": []}

        for iteration in range(max_iter):
            # Forward pass
            probs = self.predict_proba(student_indices, item_indices)

            # Compute loss (binary cross-entropy + L2 regularization)
            eps = 1e-10
            bce = -np.mean(
                correct * np.log(probs + eps) +
                (1 - correct) * np.log(1 - probs + eps)
            )
            reg = self.regularization * (np.sum(self.theta ** 2) + np.sum(self.b ** 2))
            loss = bce + reg
            history["loss"].append(loss)

            # Check convergence
            if iteration > 0 and abs(history["loss"][-2] - loss) < tol:
                break

            # Gradients
            error = probs - correct  # (P - y)

            # Gradient for theta (student abilities)
            theta_grad = np.zeros(self.n_students)
            np.add.at(theta_grad, student_indices, error)

            # Count observations per student for averaging
            theta_counts = np.zeros(self.n_students)
            np.add.at(theta_counts, student_indices, 1)
            theta_counts = np.maximum(theta_counts, 1)  # Avoid division by zero
            theta_grad = theta_grad / theta_counts + 2 * self.regularization * self.theta

            # Gradient for b (item difficulties)
            b_grad = np.zeros(self.n_items)
            np.add.at(b_grad, item_indices, -error)  # Negative because P = sigmoid(theta - b)

            # Count observations per item for averaging
            b_counts = np.zeros(self.n_items)
            np.add.at(b_counts, item_indices, 1)
            b_counts = np.maximum(b_counts, 1)
            b_grad = b_grad / b_counts + 2 * self.regularization * self.b

            # Update parameters
            self.theta -= self.learning_rate * theta_grad
            self.b -= self.learning_rate * b_grad

        return history

    def get_student_abilities(self) -> np.ndarray:
        """Get student ability parameters."""
        return self.theta.copy()

    def get_item_difficulties(self) -> np.ndarray:
        """Get item difficulty parameters."""
        return self.b.copy()
