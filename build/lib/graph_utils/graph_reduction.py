# define graph reduction helper functions

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize, label_binarize, MinMaxScaler
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import time

RANDOM_STATE = 1

def preprocess(dataset, scaler=MinMaxScaler()):
    # Renaming target columns to 'class' if exists
    if "Class" in dataset.columns:
        dataset = dataset.rename(columns={"Class": "class"})
    elif " Class" in dataset.columns:
        dataset = dataset.rename(columns={" Class": "class"})
    elif " Customer" in dataset.columns:
        dataset = dataset.rename(columns={" Customer": "class"})
    
    # Filtering classes with more than 5 instances
    dataset = dataset.groupby('class').filter(lambda x: len(x) > 5)
    X = dataset.drop('class', axis=1)  # Features
    y = dataset['class']  # Target variable

    # Splitting the dataset
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=RANDOM_STATE, stratify=y_temp)

    # Resetting index
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Scaling the data
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    X_test = pd.DataFrame(X_test)

    # Creating a mapping from original classes to 0-based consecutive integers
    unique_classes = sorted(y.unique())
    class_mapping = {original: new for new, original in enumerate(unique_classes)}
    
    # Applying the mapping to target variables
    y_train = y_train.map(class_mapping)
    y_val = y_val.map(class_mapping)
    y_test = y_test.map(class_mapping)

    return X_train, X_val, X_test, y_train, y_val, y_test

def test_classification_models(X_train, y_train, X_val, y_val, X_test, y_test, reduction_percentage, original_training_instances, similarity_method, simple=False):
    if simple:
        models = {
            'RF': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        }
    else:
        models = {
            'LR': LogisticRegression(random_state=RANDOM_STATE, n_jobs=1),
            'RF': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            'KNN': KNeighborsClassifier(n_jobs=1, n_neighbors=1),
            'XGB': XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
            'NB': GaussianNB(),
            # 'MLP': MLPClassifier(random_state=RANDOM_STATE),
        }
    
    results = []

    # Calculate class distributions
    total_class_distribution = Counter(y_train.tolist() + y_val.tolist() + y_test.tolist())
    train_class_distribution = Counter(y_train)
    val_class_distribution = Counter(y_val)
    test_class_distribution = Counter(y_test)

    # Unique classes across datasets
    classes = np.unique(y_train.tolist() + y_val.tolist() + y_test.tolist())

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predicting test data
        start_time = time.time()
        y_pred_test = model.predict(X_test)
        test_time = time.time() - start_time

        # Predicting validation data
        y_pred_val = model.predict(X_val)

        # Predicting training data
        y_pred_train = model.predict(X_train)

        # Test data metrics
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test) if len(np.unique(y_test)) == 2 else precision_score(y_test, y_pred_test, average='macro')
        recall_test = recall_score(y_test, y_pred_test) if len(np.unique(y_test)) == 2 else recall_score(y_test, y_pred_test, average='macro')
        f1_test = f1_score(y_test, y_pred_test) if len(np.unique(y_test)) == 2 else f1_score(y_test, y_pred_test, average='macro')

        # Validation data metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        precision_val = precision_score(y_val, y_pred_val) if len(np.unique(y_val)) == 2 else precision_score(y_val, y_pred_val, average='macro')
        recall_val = recall_score(y_val, y_pred_val) if len(np.unique(y_val)) == 2 else recall_score(y_val, y_pred_val, average='macro')
        f1_val = f1_score(y_val, y_pred_val) if len(np.unique(y_val)) == 2 else f1_score(y_val, y_pred_val, average='macro')

        # Training data metrics
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train) if len(np.unique(y_train)) == 2 else precision_score(y_train, y_pred_train, average='macro')
        recall_train = recall_score(y_train, y_pred_train) if len(np.unique(y_train)) == 2 else recall_score(y_train, y_pred_train, average='macro')
        f1_train = f1_score(y_train, y_pred_train) if len(np.unique(y_train)) == 2 else f1_score(y_train, y_pred_train, average='macro')

        effectiveness_val = reduction_percentage * accuracy_val
        effectiveness_test = reduction_percentage * accuracy_test

        # ROC AUC for multi-class
        if len(classes) > 2:
            y_test_binarized = label_binarize(y_test, classes=classes)
            y_val_binarized = label_binarize(y_val, classes=classes)
            y_score_test = model.predict_proba(X_test)
            y_score_val = model.predict_proba(X_val)
            roc_auc_test = roc_auc_score(y_test_binarized, y_score_test, average='macro', multi_class='ovr')
            roc_auc_val = roc_auc_score(y_val_binarized, y_score_val, average='macro', multi_class='ovr')

            # retrieve TP, FP, TN, FN
            conf_matrix = confusion_matrix(y_test, y_pred_test)
            class_wise_metrics = {}

            for class_index in range(len(conf_matrix)):
                tp = conf_matrix[class_index, class_index]
                fp = conf_matrix[:, class_index].sum() - tp
                fn = conf_matrix[class_index, :].sum() - tp
                tn = conf_matrix.sum().sum() - (tp + fp + fn)

                class_wise_metrics[class_index] = {
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'TN': tn
                }

            tp = sum(metrics['TP'] for metrics in class_wise_metrics.values())
            fp = sum(metrics['FP'] for metrics in class_wise_metrics.values())
            fn = sum(metrics['FN'] for metrics in class_wise_metrics.values())
            tn = sum(metrics['TN'] for metrics in class_wise_metrics.values())
        else:
            roc_auc_test = roc_auc_score(y_test, y_pred_test)
            roc_auc_val = roc_auc_score(y_val, y_pred_val)
            
            # retrieve TP, FP, TN, FN
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

        # Get model parameters
        model_params = model.get_params()

        results.append({
            'Model': name,
            'Test Accuracy': accuracy_test,
            'Test Precision': precision_test,
            'Test Recall': recall_test,
            'Test F1 Score': f1_test,
            'Test ROC AUC': roc_auc_test,
            'Test Effectiveness': effectiveness_test,
            'Reduction Percentage': reduction_percentage,
            'Original Training Instances': original_training_instances,
            # 'Total Instances': len(X_train) + len(X_val) + len(X_test),
            'Reduced Training Instances': len(X_train),
            'Validation Instances': len(X_val),
            'Testing Instances': len(X_test),
            'Training Time': train_time,
            'Testing Time': test_time,
            'Train Accuracy': accuracy_train,
            'Train Precision': precision_train,
            'Train Recall': recall_train,
            'Train F1 Score': f1_train,
            'Validation Accuracy': accuracy_val,
            'Validation Precision': precision_val,
            'Validation Recall': recall_val,
            'Validation F1 Score': f1_val,
            'Validation ROC AUC': roc_auc_val,
            'Validation Effectiveness': effectiveness_val,
            'Test TP': tp,
            'Test FP': fp,
            'Test TN': tn,
            'Test FN': fn,
            'Random State': RANDOM_STATE,
            'Total Class Distribution': total_class_distribution,
            'Train Class Distribution': train_class_distribution,
            'Validation Class Distribution': val_class_distribution,
            'Test Class Distribution': test_class_distribution,
            'Model Parameters': model_params,
            'Similarity Method': similarity_method,
        })

    return pd.DataFrame(results)

def calculate_reduction_percentage(reduced_nodes, total_nodes):
    return (1 - (reduced_nodes / total_nodes)) * 100

def build_graph(X, method='similarity', **kwargs):
    """
    General function to build a graph based on the specified method.

    Parameters:
    - X: Input data.
    - method: Method to use for graph construction ('similarity' or 'knn').
    - **kwargs: Additional arguments specific to the method chosen.
    """
    if method == 'similarity':
        return build_graph_from_similarity(X, **kwargs)
    elif method == 'knn':
        return build_graph_from_knn(X, **kwargs)
    else:
        raise ValueError("Invalid method specified. Choose either 'similarity' or 'knn'.")

def build_graph_from_similarity(X, metric='cosine', threshold=0.5, use_distances_as_weights=False, adjacency_matrix=False, normalize_data=False):
    """
    Build a graph from the similarity of X, with support for multiple similarity/distance metrics.

    Parameters:
    - X: Input data.
    - metric: Similarity/distance metric to use ('cosine', 'euclidean', 'manhattan', 'pearson').
    - threshold: Threshold for determining edge existence.
    - use_distances_as_weights: Whether to use distances or inverse similarity as edge weights.
    - adjacency_matrix: Whether to convert the matrix to an adjacency matrix based on average similarity.
    """
    if normalize_data:
        X = normalize(X, norm='l2')

    if metric in ['cosine', 'euclidean', 'manhattan']:
        # Calculate pairwise distances for supported metrics
        distances = pairwise_distances(X, metric=metric)
        
        # Convert distances to similarities for non-distance metrics
        if metric != 'euclidean' and metric != 'manhattan':
            # For cosine, the distance is 1 - similarity
            similarities = 1 - distances
        else:
            # Convert distances to similarities by inverting the scale
            similarities = 1 / (1 + distances)
    elif metric == 'pearson':
        # Calculate Pearson correlation coefficient for each pair
        similarities = np.corrcoef(X.T)
    else:
        raise ValueError("Unsupported metric. Choose from 'cosine', 'euclidean', 'manhattan', 'pearson'.")

    # Apply threshold to determine edges
    if adjacency_matrix:
        avg_similarity = np.mean(similarities)
        similarities[similarities < avg_similarity] = 0
        similarities[similarities >= avg_similarity] = 1
    else:
        if use_distances_as_weights:
            if metric in ['euclidean', 'manhattan']:
                weights = distances
            else:
                weights = 1 - similarities
        else:
            weights = similarities
            weights[weights < threshold] = 0
    
    edges = np.transpose(np.where(weights > threshold))
    
    G = nx.Graph()
    for (i, j) in edges:
        weight = weights[i, j]
        G.add_edge(i, j, weight=weight)

    return G

def build_graph_from_knn(X, k, metric='minkowski', use_distances_as_weights=False):
    """
    Build a graph based on k-nearest neighbors (KNN) of X.

    Parameters:
    - X: Input data.
    - k: Number of neighbors for KNN graph.
    - use_distances_as_weights: Whether to use distances as edge weights.
    """
    G = nx.Graph()

    knn_matrix = kneighbors_graph(X, n_neighbors=k, metric=metric, mode='distance', include_self=False)

    if not use_distances_as_weights:
        knn_matrix.data = 1 - knn_matrix.data

    knn_matrix = knn_matrix.toarray()

    for i in range(knn_matrix.shape[0]):
        for j in range(knn_matrix.shape[1]):
            if knn_matrix[i, j] > 0:
                G.add_edge(i, j, weight=knn_matrix[i, j])

    return G

def convert_networkx_to_igraph(G, transfer_attributes=True):
    """
    Convert a NetworkX graph to an igraph graph, ensuring continuous vertex IDs.
    """
    # Create a mapping from NetworkX node IDs to continuous integer IDs
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    # Create the igraph graph
    g = ig.Graph(directed=G.is_directed())
    g.add_vertices(len(G.nodes()))
    g.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    
    if transfer_attributes:
        # Transfer node attributes
        for nx_node, ig_node in mapping.items():
            for attribute, value in G.nodes[nx_node].items():
                g.vs[ig_node][attribute] = value
        
        # Transfer edge attributes
        for nx_edge in G.edges(data=True):
            ig_edge = g.es[g.get_eid(mapping[nx_edge[0]], mapping[nx_edge[1]])]
            for attribute, value in nx_edge[2].items():
                ig_edge[attribute] = value

    return g, reverse_mapping