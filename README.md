# Advanced Market Segmentation using Deep Clustering
An advanced application that segments customers based on their spending behaviour in various categories. In the output, these details of a customer is displayed using their Customer ID.

## Features
=> **User friendly GUI**: with PyQt5

=> **Machine Learning based segmentation**: uses Autoencoders & K-Means for clustering

=> **Popup Display**: Customer details are shown in a popup window

## Project Structure

📦 Market-Segmentation-App

│-- 📜 README.md     ------------       *Project Documentation*

│-- 📂 Datasets      ----------- *Customer & Transactions data*

│   │-- Customers.csv

│   │-- Transactions_New.csv

|-- InputWindow.py  -----------  *Python script of PyQT file* 

|-- InputWindow.ui ------------  *PyQt script file* 

|-- Src.ipynb -----------  *Jupyter Notebook (original logic)*

|-- Src.py ------------  *Python Script (convereted from notebook)* 


## Installation & Setup

#### 1. Download the Zip file of this Git reopsitory
#### 2. Extract in your desired folder
#### 3. Use the folder as the host of a new Jupyter environment
#### 4. Start a new terminal for the notebook
#### 5. Run the application using the following command "` python InputWindow.py `"

## Usage

#### 1. Enter the customer ID in the popup window. The CustomerID row consists of entries like "C0001, C0002,... C0100, etc. 

#### 2. Click the "Get Customer Information" button.

#### 3. A popup window will show the details like: Customer Name, Spending Group, Total spending and transactions.

#### 4. If an invalid Customer ID is entered, a warning popup appears.


## Technologies Used

Frontend: PyQt5

Backend: Pandas, NumPy, Scikiy-Learn

Machine Learning : Autoencoders, K-Means

Visualisation: Seaborn, Plotly



