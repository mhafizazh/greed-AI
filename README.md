---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
---

# Model Card for greedAI 1.0

<!-- Provide a quick summary of what the model is/does. -->

Our AI Hacked (hackathon) project is a natural language processing model that classifies input messages as either emergency (1) or non-emergency (0). It uses Python libraries like NumPy for data manipulation, Pandas for dataset handling, and Matplotlib for plotting. The text data is transformed into a numerical format using CountVectorizer, and a Gaussian Naive Bayes classifier is employed for the prediction task. The model's performance is evaluated using a confusion matrix and accuracy score. Finally, the trained model can be saved and reused with the help of the pickle module.


- **Developed by:** <a href="https://github.com/mhafizazh">Muhammad Hafiz Azhari</a>


## Key Uses

### Understanding Patient Explanations
- Processes and interprets textual descriptions provided by patients about their symptoms or health concerns.

### Emergency Classification
- Classifies patient situations into varying levels of emergency, from non-urgent to critical, based on the input text.

### Assisting Medical Professionals
- Acts as a decision-support tool in triage and initial assessments, helping healthcare providers prioritize patients based on the severity of their conditions.

### Enhancing Healthcare Response
- Aids in optimizing the allocation of medical resources and ensuring timely attention to critical cases.

## Bias in the Dataset
- **Source Dependency:** The training dataset is generated from ChatGPT, which may reflect the biases inherent in its training data.
- **Language Variability:** The dataset might not fully represent the diversity of human expression in emergency situations, leading to potential biases.

## Risks
- **Misclassification:** There is a significant risk of misclassifying emergencies with a 77% accuracy rate.
- **Overreliance:** Users might overly depend on the AI, which can be risky in critical scenarios requiring human judgement.

## Limitations
- **Clinical Diagnosis:** The system's accuracy is not suitable for direct medical diagnosis.
- **Contextual Understanding:** The AI lacks the ability to understand context and severity as a human doctor would.
- **Professional Oversight Required:** The system should not replace professional medical advice and requires human oversight.

## Usage Guidelines
- Always consult with a qualified healthcare professional for emergency situations.
- Use the AI system as a supplementary tool rather than a standalone solution.



## Training Details

### Training Data

The training dataset used for this project can be found at [this link](https://docs.google.com/spreadsheets/d/1uMbRIAH-JqNr_ubZjRLGRUxodSqVjHKdYsrecNenQ5M/edit?usp=sharing). This dataset is integral to the model's ability to accurately classify text messages as emergency or non-emergency. For more detailed information about the composition and characteristics of the training data, including any preprocessing or filtering steps applied, please refer to the accompanying Dataset Card.


### Training Procedure

The training of the model is a crucial step in ensuring its effectiveness and accuracy. Our approach utilizes the Gaussian Naive Bayes algorithm from the Scikit-learn library for the classification task.

#### Preprocessing 

Before training, the dataset undergoes specific preprocessing steps to convert the raw text into a format suitable for the Gaussian Naive Bayes classifier. This may include tokenization, removal of stop words, and vectorization of the text data.


#### Model Training

The core of our training procedure is succinctly encapsulated in the following Python code:

```python
from sklearn.naive_bayes import GaussianNB
import joblib

# Initialize the Gaussian Naive Bayes classifier
classifier = GaussianNB()

# Training the classifier with the training data
classifier.fit(X_train, y_train)
```




## Model Card Authors [optional]
<a href="https://github.com/mhafizazh">Muhammad Hafiz Azhari</a>

## Model Card Contact
<p>email: mhafizazh.dev@gmail.com</p>
<p>linkedin: www.linkedin.com/in/muhammad-hafiz-azhari</p>


