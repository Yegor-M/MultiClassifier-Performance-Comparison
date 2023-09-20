# Multi-Class Classification with scikit-learn

# Project Goal

  The primary aim is to assess and contrast the effectiveness of diverse classifiers and ensemble techniques when applied to a synthetic dataset created using the make_moons function.

## Project Overview

  • Dataset Creation: Generate a synthetic dataset with 10,000 samples and controlled noise (0.4) using make_moons.

  • Data Splitting: Split the dataset into training and test subsets for model evaluation.

  • Decision Tree Analysis: Investigate Decision Trees as classifiers, exploring entropy and Gini criteria and varying tree depths.

  • Random Forests: Evaluate Random Forests as classifiers, testing performance with different tree counts.

  • Logistic Regression and SVM: Train Logistic Regression and Support Vector Machine (SVM) classifiers.

  • Ensemble Classifier: Create an ensemble classifier (VotingClassifier) by combining SVM, Logistic Regression, and Random     Forests to enhance classification.

  • Results Evaluation: Assess classifier performance using accuracy scores to understand their effectiveness.

## Classifier Accuracy Summary:

  • Voting Classifier (Ensemble)
    The Voting Classifier, which combines the strengths of multiple classifiers, emerged as the top performer with an accuracy of 84.65%. This reinforces the concept that ensemble methods can often enhance classification results.
  
  • Decision Trees
    Both Decision Tree classifiers, one using Entropy and the other Gini criterion, showcased solid performance, with accuracies of approximately 84.60% and 84.55%, respectively. This indicates their competence in capturing complex decision boundaries.

  • Random Forest
    The Random Forest Classifier achieved an accuracy of 83.85%, providing robust results. While slightly below individual Decision Trees, it demonstrated consistent performance.

  • Logistic Regression and SVM 
    Logistic Regression and Support Vector Machine (SVM) classifiers both delivered competitive results, with accuracies of approximately 82.45% and 82.50%, respectively. They proved to be reliable options for multi-class classification tasks.
