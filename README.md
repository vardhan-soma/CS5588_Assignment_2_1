**Diabetes Prediction using Artificial Neural Netwok(ANN) and Deep Belief Network(DBN)**


**Dataset Overview:**

The PIMA Indian Diabetes Dataset is a collection of medical data for 768 female patients of Pima Indian heritage, aged 21 years and older, who were either diagnosed with diabetes or not. This dataset is widely used for machine learning and statistical analysis to predict diabetes based on several health factors.
Features:
	•	Pregnancies: Number of times pregnant.
	•	Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test).
	•	BloodPressure: Diastolic blood pressure (mm Hg).
	•	SkinThickness: Triceps skinfold thickness (mm).
	•	Insulin: 2-Hour serum insulin (mu U/ml).
	•	BMI: Body mass index (weight in kg/(height in m)^2).
	•	DiabetesPedigreeFunction: A function that scores likelihood of diabetes based on family history.
	•	Age: Age of the patient (years).
	•	Outcome: Class variable (0 if non-diabetic, 1 if diabetic).

The goal is to use the dataset to build a predictive model for diagnosing diabetes based on patient health metrics.

The dataset was initially collected by the National Institute of Diabetes and Digestive and Kidney Diseases and is available in the public domain through the UCI Machine Learning Repository.
Dataset Link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Models:**

**Model 1: Feedforward artificial neural network (ANN) using TensorFlow and Keras to predict diabetes**
Model Architecture:
	•	Input Layer: The model takes 8 input features corresponding to the medical data from the dataset (e.g., pregnancies, glucose levels, BMI, etc.).
	•	Hidden Layers: The model includes two hidden layers:
	•	The first hidden layer has a variable number of units (3, 4, 5, or 6), using the ReLU activation function.
	•	The second hidden layer also has a variable number of units (3, 4, 5, or 6), using the ReLU activation function.
	•	Output Layer: The output layer consists of a single neuron with a sigmoid activation function, designed to predict the binary classification outcome (diabetes or not).

Also updated model to enhance by adding Batch Normalization and Dropout layers.
	•	Batch Normalization helps speed up training and improve generalization by normalizing activations, while Dropout (with a rate of 0.5) is used to prevent overfitting by randomly ignoring some neurons during training.
	•	This model also uses the Adam optimizer and binary cross entropy loss function, with similar hidden layer configurations.
 
**Model 2: Deep Belief Network (DBN) with Restricted Boltzmann Machine (RBM)**

Deep Belief Network (DBN) is implemented using a combination of Restricted Boltzmann Machines (RBM) and a fully connected neural network to predict diabetes based on the PIMA Indian Diabetes Dataset. This approach adds an unsupervised pre-training step with RBM before training a deep neural network.
Model Workflow:
	1.	Data Preprocessing:
	•	Input features are scaled using MinMaxScaler to normalize the data between 0 and 1, making it suitable for RBM and neural network processing.
	2.	RBM Pre-training:
	•	The first layer is a Bernoulli RBM with 128 hidden units. It serves as an unsupervised feature extractor, transforming the original dataset into a new feature space.
	•	The RBM is trained with a learning rate of 0.01 for 30 iterations to capture the complex structure of the input data.
	3.	Deep Neural Network:
	•	The transformed features from the RBM are fed into a deep neural network consisting of:
	•	A fully connected layer with 128 units and ReLU activation.
	•	A second fully connected layer with 64 units.
	•	Dropout layers (0.3) are applied after each layer to prevent overfitting.
	•	The output layer is a single neuron with sigmoid activation for binary classification.
	4.	Training:
	•	The model is compiled with the Adam optimizer and binary cross entropy loss function.
	•	The neural network is trained on the RBM-transformed data for 150 epochs with a batch size of 10.
	5.	Evaluation:
	•	Predictions are made on the test set transformed by the RBM, and the model’s performance is evaluated using a confusion matrix and accuracy.

**Results:**

1. The basic ANN without additional regularization techniques achieved accuracies ranging from 77.92% to 80.46%.
2. Model incorporated with Batch Normalization to speed up training and Dropout layers to prevent overfitting, achieving accuracies ranging from 77.92% to 81.81%, with the highest accuracy being 81.81%.
3. The DBN model, which combined a Restricted Boltzmann Machine (RBM) for feature extraction and a deep neural network for classification, achieved an accuracy of 69.48% on the test set.

**To run the code:**

1. Download the dataset
2. Install the dependencies like Python, Tensorflow, Numpy, Pandas, Scikit-Learn, Seaborn, Matplotlib if not downloaded earlier
3. Run the notebook

**Applicability to ongoing project:**

The implemented models, including the Deep Belief Network (DBN) and Artificial Neural Networks (ANN) with Dropout and Batch Normalization, are critical in achieving the project’s objectives of early diabetes detection. These models enhance predictive accuracy by exploring different architectures, ensuring reliable predictions of pre-diabetes, Type-2 diabetes, or gestational diabetes. Techniques like Dropout and Batch Normalization prevent overfitting and speed up training, making the models more robust when applied to diverse real-world data, such as demographic, lifestyle, and biometric information. The RBM in the DBN helps capture complex patterns in the data, improving feature representation and overall prediction performance. Moreover, these models are scalable and can handle large datasets, which aligns with the goal of developing a user-friendly application for early intervention, ultimately contributing to improved patient outcomes and better healthcare management.
