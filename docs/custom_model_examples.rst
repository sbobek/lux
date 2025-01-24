Custom models and datasets
=============

In the following sections we provide examples of using LUX with custom models and datasets.

The example shows how to handle categorical variables in LUX and how to add PyTorch model with custom input transformation.
The full working example can be found here: `Google Colab example <https://colab.research.google.com/drive/1Yb-VGzsJupTYyyuwA9dEVLYkftuyk4C8?usp=sharing>`_

You can add categorical variables to lux, by passing `categorical` parameter to `fit` function.
The `categorical` should be alist of boolean values, having `True` in places that are considered categorical.

.. code-block:: python

    from lux.lux import LUX
    ## NOTE: RESULTS WILL BE DIFFERENT DEPENDING ON THE SAMPLE THAT IS SAMPLE
    explain_instance = X_train.sample(1).values
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    features = df.columns[:-1]
    categorical_indicator = [col in categorical_columns for col in features]

    #train lux on neighbourhood equal 20 instances
    lux = LUX(predict_proba = model.predict_proba,
        neighborhood_size=50,max_depth=2,
        node_size_limit = 1,
        grow_confidence_threshold = 0 )
    lux.fit(X_train, y_train,
    instance_to_explain=explain_instance,
    # categorical indicator
    categorical=categorical_indicator)


Below, the example of how to use custom, more complex model is provided (e.g. Deep Neural Network).
The only requirement is that the custom model implements `predict` and `predict_proba` functions.

.. code-block:: python

    from lux.lux import LUX
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    import numpy as np
    import pandas as pd


    class SimpleNN(nn.Module):
        def __init__(self,input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

        def fit(self,X,y):
          # Convert data to tensors
          X_train_tensor = torch.tensor(X.todense(), dtype=torch.float32)
          y_train_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

          # Loss and optimizer
          criterion = nn.BCELoss()
          optimizer = optim.Adam(self.parameters(), lr=0.001)

          # Training loop
          for epoch in range(100):
              optimizer.zero_grad()
              outputs = self(X_train_tensor)
              loss = criterion(outputs, y_train_tensor)
              loss.backward()
              optimizer.step()

              if (epoch + 1) % 10 == 0:
                  print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

        def predict_proba(self, X):
            # Ensure input is dense (convert if sparse)
            if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
                X = X.toarray()  # Convert sparse matrix to dense

            # Convert to tensor if necessary
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Perform the forward pass to get predictions
            with torch.no_grad():
                outputs = self(X_tensor)

            # Convert to probabilities (binary classification)
            probabilities = outputs.numpy()  # Convert to numpy array
            return np.column_stack([1 - probabilities, probabilities])  # For binary classification

        def predict(self, X):
            # Ensure input is dense (convert if sparse)
            if hasattr(X, "toarray"):  # This checks if the input is a sparse matrix (like from OneHotEncoder)
                X = X.toarray()  # Convert sparse matrix to dense

            # Convert to tensor if necessary
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Perform the forward pass to get predictions
            with torch.no_grad():
                outputs = self(X_tensor)

            # Classify based on the output probability (threshold of 0.5)
            predictions = (outputs >= 0.5).float()  # Binary classification: 0 or 1
            return predictions.numpy()  # Convert to numpy array


Once done, the custom model can be wrapped with custom data transformer, to make the whole classiferi one single blackbox to LUX:

.. code-block:: python

    class CategoricalWrapper:
        def __init__(self, model_creator,  model_params=None, ohe_encoder=None, categorical_indicator=None, features=None, categories='auto', normalize=False):
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder,StandardScaler

            # OneHotEncoder for categorical columns
            if ohe_encoder is None:
                self.ohe_encoder = OneHotEncoder(categories=categories)
            else:
                self.ohe_encoder = ohe_encoder

            # Store parameters
            self.features = features
            self.categories = categories
            self.categorical_indicator = categorical_indicator
            self.model_params = model_params
            self.model_creator = model_creator

            # Add StandardScaler for non-categorical features if normalize=True
            transformers = [
                ("categorical", self.ohe_encoder, [f for f, c in zip(features, categorical_indicator) if c])
            ]

            # If normalize is True, add StandardScaler for non-categorical columns
            if normalize:
                non_categorical_columns = [f for f, c in zip(features, categorical_indicator) if not c]
                transformers.append(("scaler", StandardScaler(), non_categorical_columns))

            # Create the ColumnTransformer
            self.ct = ColumnTransformer(
                transformers,
                remainder='passthrough'
            )

            self.model_params = model_params
            self.model_creator = model_creator


        def fit(self, X, y):
            X_tr = self.ct.fit_transform(X)

            if self.model_params is None:
                model_params = {}
            elif self.model_params=='input_size':
                model_params = {'input_size':X_tr.shape[1]}

            # Create the model by passing parameters to the model_creator lambda
            self.model = self.model_creator(**model_params)

            self.model.fit(X_tr, y)
            return self


        def predict(self, X):
            if type(X) is np.ndarray and self.features is not None:
                X = pd.DataFrame(X, columns=self.features)
            return self.model.predict(self.ct.transform(X))

        def predict_proba(self, X):
            if type(X) is np.ndarray and self.features is not None:
                X = pd.DataFrame(X, columns=self.features)

            X = self.ct.transform(X)
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            elif hasattr(self.model, 'decision_function'):
                # Sigmoid transformation for decision_function output
                decision_scores = self.model.decision_function(X)
                probabilities = 1 / (1 + np.exp(-decision_scores))
                return np.column_stack([1 - probabilities, probabilities])
            else:
                return np.array([self.model.predict(X)==c for c in self.model.classes_]).T

        def score(self,X,y):
            return self.model.score(self.ct.transform(X),y)


Finally, the whole can be run in a unified way:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    import xgboost as xgb


    # Define the URL of the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    # Define the column names (based on the dataset documentation)
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]

    # Download the dataset into a Pandas DataFrame
    df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

    # Display basic information about the dataset
    print("Dataset Shape:", df.shape)
    print("\nSample Data:")
    print(df.head())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Preprocess the dataset (e.g., encoding categorical variables, handling missing values)
    df = df.dropna()
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns[categorical_columns != 'income']
    features = df.columns[:-1]
    categorical_indicator = [col in categorical_columns for col in features]

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    df['income'] = le.fit_transform(df['income'])

    # Split the data into features and target
    target = 'income'
    X = df.drop(columns=[target])
    y = df[target]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose and train a model
    model_type = 'deep_learning'

    model_creators = {
        'random_forest': lambda: RandomForestClassifier(),
        'svm': lambda: SVC(probability=True),
        'logistic_regression': lambda: LogisticRegression(),
        'mlp': lambda: MLPClassifier(),
        'deep_learning': lambda input_size: SimpleNN(input_size=input_size)  # Lambda with parameter
    }

    if model_type == 'xgb':
        # Use XGBoost with categorical support enabled
        model = xgb.XGBClassifier(enable_categorical=True)
        model.fit(X_train, y_train)
    elif model_type == 'random_forest':
        model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
        model.fit(X_train, y_train)
    elif model_type == 'svm':
        model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
        model.fit(X_train, y_train)
    elif model_type == 'logistic_regression':
        model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
        model.fit(X_train, y_train)
    elif model_type == 'mlp':
        model = CategoricalWrapper(model_creators[model_type], categorical_indicator=categorical_indicator, features=features)
        model.fit(X_train, y_train)
    elif model_type == 'deep_learning':
        # Define a simple neural network with PyTorch

        # Wrap the trained model in CategoricalWrapper
        model = CategoricalWrapper(
            model_creator=model_creators[model_type],
            model_params='input_size',
            features=X_train.columns,
            normalize=True,
            categorical_indicator=categorical_indicator
        )
        model.fit(X_train, y_train)
    else:
        print("Invalid model type selected.")

    # If not deep learning, evaluate the model
    if model_type != 'deep_learning':
        accuracy = model.score(X_test, y_test)
        print(f"\nModel Accuracy: {accuracy:.2f}")
