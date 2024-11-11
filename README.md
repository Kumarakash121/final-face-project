Face Recognition using K-Nearest Neighbors (KNN)


Project Overview

This project implements face recognition using the K-Nearest Neighbors (KNN) algorithm, which is a simple and effective method for classification tasks. The goal of this project is to recognize and classify human faces from images or video frames by comparing them to a set of known faces stored in a dataset.


The KNN algorithm works by finding the 'K' closest samples to the input data (in this case, the face to be recognized) and assigning the most frequent class among those neighbors.


Features

Face Detection: Uses OpenCV's Haar Cascade Classifier to detect faces in an image.
Face Recognition: KNN algorithm is applied to classify the detected faces by comparing them to a known dataset of faces.
Dataset Support: You can train the model on your own dataset by adding labeled face images.
Real-time Recognition: Option to use webcam for live face recognition.
Prerequisites
To run this project, you'll need the following:

Python 3.x
OpenCV for Python (cv2)
Numpy
scikit-learn (for KNN implementation)
dlib (optional, for better performance in face detection)
Installation
Step 1: Clone the Repository
Clone this repository to your local machine using the following command:

bash
Copy code
git clone https://github.com/yourusername/face-recognition-knn.git
cd face-recognition-knn
Step 2: Create a Virtual Environment (Optional but Recommended)
It is a good practice to use a virtual environment to manage dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate   # For Linux/MacOS
venv\Scripts\activate      # For Windows
Step 3: Install Required Libraries
Install the required Python libraries by running:

bash
Copy code
pip install -r requirements.txt
requirements.txt should include the following dependencies:

Copy code
opencv-python
numpy
scikit-learn
dlib
You can generate this file using pip freeze > requirements.txt if needed.

Usage
Step 1: Prepare the Dataset
Create a folder named dataset in the project directory.
In the dataset folder, create subfolders where each subfolder's name is the person's label (e.g., person1, person2).
Inside each person's folder, add images of that person (preferably frontal face images).
The folder structure should look something like this:

markdown
Copy code
dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── person2/
│   ├── img1.jpg
│   └── img2.jpg
└── person3/
    └── img1.jpg
Step 2: Train the Model
Run the script to train the KNN classifier with the dataset:

bash
Copy code
python train_knn.py
This will process the images, extract facial features, and store the KNN model in a file named knn_model.yml (or another preferred format).

Step 3: Run Face Recognition
After training the model, you can run the face recognition on an image or video.

For image recognition:
bash
Copy code
python recognize_face.py --image "test_image.jpg"
For live webcam recognition:
bash
Copy code
python recognize_face.py --webcam
This will open your webcam and attempt to recognize any faces that appear in real-time.

Example Output
When the recognition is successful, you should see an output similar to this:

csharp
Copy code
[INFO] Recognized person: person1 with confidence 95%
If the face is not recognized, the output will be something like:

csharp
Copy code
[INFO] Face not recognized.
File Structure
bash
Copy code
.
├── dataset/                # Folder to store labeled images of faces
├── knn_model.yml           # Trained KNN model
├── train_knn.py            # Script to train the KNN model
├── recognize_face.py       # Script to recognize faces from an image or video stream
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
Training and Tuning
You can experiment with different values for the number of neighbors K in the KNN algorithm. You can adjust this by modifying the K parameter in the training script (train_knn.py).

For better accuracy, consider increasing the dataset size and using different preprocessing techniques like histogram equalization or face alignment.

Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome, and you can improve this project by adding:

Better face detection using deep learning-based models (e.g., dlib, MTCNN).
Improved KNN parameter tuning for better accuracy.
Adding an option for recognizing faces from video files.
License
This project is licensed under the MIT License - see the LICENSE file for details.
