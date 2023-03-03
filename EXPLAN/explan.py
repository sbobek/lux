import sys
sys.path.append('./EXPLAN')
from utils import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
from imblearn.over_sampling import SMOTE,SMOTENC
from random_sampling import RandomSampling
from quartile_discretizer import QuartileDiscretization
from sturges_discretizer import SturgesDiscretization
from sample_manipulation import SampleManipulation

def Explainer(instance2explain, blackbox, dataset, N_samples=3000, tau=250, depth=2):
    """
    This is the main function of EXPLAN method. It includes the four main steps
    of the algorithm. The first three steps define a representative locality for
    the instance2explain, and the forth step creates a decision tree as
    interpretable model for explaining the instance2explain.
    """

    # Dense data generation step
    dense_samples = DataGeneration(instance2explain, blackbox, dataset, N_samples)

    # Representative data selection step
    representative_samples = DataSelection(instance2explain, blackbox, dense_samples, tau)

    # Data balancing step
    neighborhood_data = DataBalancing(blackbox, representative_samples, dataset)

    # Rule-based interpretable Model step
    exp_rule, exp_info = InterpretabelModel(instance2explain, blackbox, neighborhood_data, dataset, depth=depth)
    return exp_rule, exp_info

# Dense data generation step
def DataGeneration(instance2explain, blackbox, dataset, N_samples):
    """
    This function performs dense data generation for the instance2explain.
    It starts by randomly generating data points using the distribution of
    training data, and then making them closer to the instance2explain
    by considering similarities between feature values and feature importance.
    """

    # Generating random data using the distribution of training data
    # Discretizing random data for comparison of feature values
    training_data = dataset['X']
    random_samples = RandomSampling(instance2explain, training_data, N_samples)
    random_samples_dc = QuartileDiscretization(random_samples)

    # Constructing a random forest classifier as surrogate model
    surrogate_model = RandomForestClassifier(n_estimators=10)
    surrogate_model.fit(random_samples, blackbox.predict(random_samples))

    # Extracting feature contributions using TreeIntepreter
    # Discretizing contributions for comparison of feature importance
    prediction, bias, contributions = ti.predict(surrogate_model, random_samples)
    contributions_dc = SturgesDiscretization(contributions)

    # Making a dense neighborhood w.r.t instance2explain
    dense_samples = SampleManipulation(prediction, random_samples, random_samples_dc, contributions_dc)

    return dense_samples

# Representative data selection step
def DataSelection(instance2explain, blackbox, dense_samples, tau):
    """
    This function accept generated compact data and select representative samples
    as candidate set for the instance2explain. In this way, created data points in
    the previous phase that are outlier are removed from sample set. This helps the
    interpretable model to rely on the samples in the close locality for explanation.
    """

    n_clusters = 2     # Number of clusters
    groups = list()    # Groups of data per class label
    preds = blackbox.predict(dense_samples)
    labels = np.unique(preds)
    for l in labels:
        # Appending instance2explain to each class of data
        groups.append(np.r_[instance2explain.reshape(1,-1), np.squeeze(dense_samples[np.where(preds == l),:], axis=0)])
        # Iterative data selection
        while True:
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(groups[l])
            # Collecting data points belong to the cluster of instance2explain
            indices = np.where(clustering.labels_ == clustering.labels_[0])
            c_instance2explain = np.squeeze(groups[l][indices, :], axis=0)
            # Checking the termination condition
            if c_instance2explain.shape[0] <= tau:
                break
            else:
                groups[l] = c_instance2explain
    # Merging the representative samples of every class
    representative_samples = np.concatenate([np.array(groups[l]) for l in labels])
    return representative_samples

# Data balancing step
def DataBalancing(blacbox, representative_samples, dataset):
    """
    The aim of this function is to handle potential class imbalance problem
    in the representative sample set. Having a balanced data set is necessary
    for creating a fair interpretable model. The output of this step is the
    final training data for the interpretable model.
    """

    # Applying SMOTE oversampling
    discrete_indices = dataset['discrete_indices']
    if len(discrete_indices) > 0:
        oversampler = SMOTENC(random_state=42,categorical_features=discrete_indices)
    else:
        oversampler = SMOTE(random_state=42)
    balanced_samples, _ = oversampler.fit_resample(representative_samples, blacbox.predict(representative_samples))
    balanced_samples[:, discrete_indices] = np.around(balanced_samples[:, discrete_indices]).astype(int)
    return balanced_samples

# Rule-based interpretable model step
def InterpretabelModel(instance2explain, blackbox, neighborhood_data, dataset,depth=2):
    """
    This function creates a rule based interpretable classifier for explaining the
    instance2explain. Here, YaDT implementation of the C4.5 decision tree is used.
    The output is the explanation rule, interpretable classifier, and several useful
    information about the explanation.
    """

    # Reading data set information
    dataset_name = dataset['name']
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']

    # Creating a data frame of the neighborhood data
    dfX = build_df2explain(blackbox, neighborhood_data, dataset)
    
    # Using YaDT as interpretable model
    dt, dt_dot = pyyadt.fit(dfX, class_name, columns, features_type, discrete, continuous,
                            filename=dataset_name, path='./EXPLAN/yadt/', sep=';', log=True, depth=depth)

    # Applying black-Box and decision tree on the instance2explain
    y_x_bb = blackbox.predict(instance2explain.reshape(1,-1))[0]
    dfx = build_df2explain(blackbox, instance2explain.reshape(1,-1), dataset).to_dict('records')[0]
    y_x_dt, exp_rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Applying black-Box and decision tree on the neighborhood data
    y_X_bb = blackbox.predict(neighborhood_data)
    y_X_dt,leaf_nodes = pyyadt.predict(dt, dfX.to_dict('records'), class_name, features_type , discrete, continuous)

    # YaDT prediction function
    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Updating labels
    if class_name in label_encoder:
        y_x_dt = label_encoder[class_name].transform(np.array([y_x_dt]))[0]
    if class_name in label_encoder:
        y_X_dt = label_encoder[class_name].transform(y_X_dt)

    # Returning explanation information
    exp_info = {
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'predict': predict,
        'y_x_bb': y_x_bb,
        'y_x_dt': y_x_dt,
        'y_X_bb': y_X_bb,
        'y_X_dt': y_X_dt,
        'X': neighborhood_data,
        'dfX': dfX,
        'C': dt
    }
    return  exp_rule, exp_info
