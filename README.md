# CNN Handwritten Marks Recognition from Exam Sheets

Ever wondered how marks written on physical exam answer sheets can be digitized?

In this project, I built a handwritten number recognition system designed to automatically extract marks from the front page of exam answer books, where scores are manually written in structured boxes.

---

## How It Works

1. A clean reference image of the blank answer sheet is used to define Region of Interest (ROI) coordinates for all the mark entry boxes.
2. When new filled answer sheets are provided:
   - The system automatically detects and crops each predefined ROI.
   - Each cropped image is passed to a pre-trained CNN model that predicts the handwritten number.
3. Multi-digit numbers are recognized and results are saved in a structured CSV file for analysis or digital storage.

---

## Features

-  Recognizes handwritten digits using a CNN model
-  Fast processing and suitable for batch image handling
-  Outputs results in a structured CSV format
-  Works using reference-based ROI detection
-  No manual cropping or image pre-processing needed

---

## Project Structure

CNN_Handwritten_Number_Recognition/
│
├── input_images/ # Folder containing scanned answer sheet images
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── reference_image.jpg # Blank answer sheet used to define ROIs
├── roi_values.txt # Contains manually defined ROI coordinates
├── model.h5 # Trained CNN model for digit recognition
├── cnn_number_recognition.py # Main script to extract and predict marks
├── output.csv # CSV file storing prediction results
├── requirements.txt # Required Python packages
└── README.md # Project documentation

---

## How to run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   Note: python3.10 is recommended.

2. Run the script:
    python cnn_number_recognition.py
   Note: Before running the script change the paths for input_images, reference_image, model.h5

3. Check output.csv for predicted marks.

---

### Model Info

The digit recognition model is a lightweight CNN trained to recognize single-digit and multi-digit handwritten numbers.

---

## License
This project is licensed under the MIT License.
