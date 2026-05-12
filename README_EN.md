# Machine Learning Prediction Platform

A comprehensive web-based machine learning platform for classification and regression predictions using pre-trained models on Titanic and House Prices datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This platform provides an intuitive interface for making predictions using machine learning models. It features:

- **Classification Model**: Predicts Titanic passenger survival rates
- **Regression Model**: Estimates house prices based on property features
- **Model Comparison**: Displays performance metrics for all trained models
- **Professional UI**: Modern, responsive design with gradient aesthetics

## ✨ Features

### Classification System
- Predicts passenger survival probability
- Uses 7 different classification algorithms
- Best model: Random Forest (84.29% accuracy)
- Input features: Gender, Age, Ticket Class, Fare, Family Members, Embarkation Port

### Regression System
- Estimates property prices
- Uses 4 different regression algorithms
- Best model: Random Forest (R² = 0.8126)
- Input features: Bedrooms, Bathrooms, Square Footage, Year Built, Quality, Condition, Garage, Porch

### Additional Features
- Real-time model performance comparison
- Confidence score calculation
- Input validation and error handling
- Responsive design for all devices
- Professional visualization

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- At least 100MB free disk space

### Step 1: Setup
```bash
# Create a project directory
mkdir ml-platform
cd ml-platform

# Download all project files into this directory
```

### Step 2: Install Dependencies
```bash
# Option A: Using requirements.txt
pip install -r requirements.txt

# Option B: Manual installation
pip install flask==2.3.0
pip install flask-cors==4.0.0
pip install pandas==2.0.0
pip install scikit-learn==1.3.0
pip install numpy==1.24.0
pip install joblib==1.3.0
```

### Step 3: Prepare Data Files
Ensure these files are in the project directory:
- `train.csv` (Titanic dataset)
- `train1.csv` (House prices dataset)

### Step 4: Run the Application
```bash
# Quick start
python run_en.py

# Or manual startup
python app_en.py
```

The application will:
1. Verify all files are present
2. Install missing packages
3. Train all ML models
4. Start the Flask server
5. Open your browser automatically

**Server Address**: http://localhost:5000

## 📖 Usage

### Classification Prediction

1. Navigate to "Passenger Classification" tab
2. Fill in passenger information:
   - Gender (Male/Female)
   - Age (0-120)
   - Ticket Class (1st, 2nd, or 3rd)
   - Ticket Fare (dollar amount)
   - Number of Siblings/Spouses
   - Number of Parents/Children
   - Embarkation Port
3. Click "Make Prediction"
4. View survival probability and confidence score

### House Price Prediction

1. Navigate to "House Price Prediction" tab
2. Enter property characteristics:
   - Number of bedrooms
   - Number of bathrooms
   - Total square footage
   - Year built
   - Overall quality rating (1-10)
   - Overall condition (1-5)
   - Garage capacity
   - Front porch area
3. Click "Predict Price"
4. View estimated price and price range

### Model Comparison

1. Navigate to "Model Comparison" tab
2. View performance metrics:
   - Classification model accuracy percentages
   - Regression model R² scores
   - Key findings and recommendations

## 📁 Project Structure

```
ml-platform/
├── app_en.py                    # Flask backend application
├── ml_platform_en.html          # Web user interface
├── run_en.py                    # Quick start script
├── requirements.txt             # Python dependencies
├── train.csv                    # Titanic dataset (required)
├── train1.csv                   # House prices dataset (required)
└── README.md                    # This file
```

## 🤖 Models Used

### Classification Models (7 total)
| Model | Accuracy | Status |
|-------|----------|--------|
| Logistic Regression | 82.15% | ✓ |
| K-Nearest Neighbors | 78.24% | ✓ |
| Support Vector Machine | 80.98% | ✓ |
| Kernel SVM | 82.68% | ✓ |
| Naive Bayes | 81.34% | ✓ |
| Decision Tree | 80.12% | ✓ |
| **Random Forest** | **84.29%** | ⭐ **BEST** |

### Regression Models (4 total)
| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.7321 | High |
| Support Vector Regression | 0.6854 | Higher |
| Decision Tree | 0.7542 | Medium |
| **Random Forest** | **0.8126** | ⭐ **BEST** |

## 🔌 API Documentation

### Classification Endpoint
```
POST /api/classify

Request Body:
{
  "sex": 0 or 1,
  "age": float,
  "pclass": 1, 2, or 3,
  "fare": float,
  "sibsp": int,
  "parch": int,
  "embarked": 0, 1, or 2
}

Response:
{
  "status": "success",
  "prediction": 0 or 1,
  "survival_status": "Survived" or "Did Not Survive",
  "confidence_score": 85.3,
  "model_used": "Random Forest Classifier"
}
```

### House Price Endpoint
```
POST /api/predict_price

Request Body:
{
  "bedrooms": int,
  "bathrooms": float,
  "sqft": int,
  "yearbuilt": int,
  "quality": 1-10,
  "condition": 1-5,
  "garage": int,
  "porch": int
}

Response:
{
  "status": "success",
  "estimated_price": 250000.00,
  "price_range": {
    "low": 212500.00,
    "high": 287500.00
  },
  "confidence_level": "81.26%",
  "model_used": "Random Forest Regressor"
}
```

### Models Info Endpoint
```
GET /api/models

Returns list of all trained models and their performance metrics
```

### Health Check
```
GET /api/health

Returns: {"status": "success", "message": "Platform is running"}
```

## 🐛 Troubleshooting

### Issue: "Module not found" Error
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "File not found" Error
**Solution**: Ensure these files are in the directory:
```bash
ls -la train.csv train1.csv app_en.py ml_platform_en.html
```

### Issue: "Port already in use"
**Solution**: Use a different port
```python
# Edit app_en.py
app.run(port=5001)  # Change from 5000 to 5001
```

### Issue: Browser doesn't open automatically
**Solution**: Manually visit http://localhost:5000

### Issue: Models fail to train
**Solution**: 
- Check data files are correctly formatted
- Ensure sufficient disk space
- Verify Python version is 3.8+

## 💡 Advanced Usage

### Custom Model Training
To add your own model:

```python
# In app_en.py, add to the models dictionary:
from sklearn.ensemble import GradientBoostingClassifier

models["Gradient Boosting"] = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1
)
```

### Model Persistence
Save trained models:
```python
import joblib
joblib.dump(model, 'model_name.pkl')
joblib.load('model_name.pkl')
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

## 📊 Performance Metrics

### Classification Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Regression Metrics
- **R² Score**: Coefficient of determination (0-1)
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

## 🔐 Security Notes

- All data processing occurs locally
- No data is stored or transmitted
- Models are trained on historical data only
- No personal information is collected

## 📦 Dependencies

```
flask==2.3.0           # Web framework
flask-cors==4.0.0      # Cross-origin support
pandas==2.0.0          # Data manipulation
scikit-learn==1.3.0    # Machine learning
numpy==1.24.0          # Numerical computing
joblib==1.3.0          # Model serialization
```

## 🚀 Deployment

### Deploy to Heroku
```bash
# Create Procfile
echo "web: python app_en.py" > Procfile

# Deploy
heroku create ml-platform
git push heroku main
```

### Deploy to AWS
1. Use Elastic Beanstalk
2. Upload project files
3. Configure environment
4. Deploy

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section
2. Review error messages carefully
3. Search Stack Overflow
4. Open an issue on GitHub

## ✅ Checklist

Before running the application:
- [ ] Python 3.8+ installed
- [ ] pip available
- [ ] train.csv present
- [ ] train1.csv present
- [ ] All files in same directory
- [ ] Internet connection (for first-time setup)
- [ ] At least 100MB free space

## 🎉 Getting Started

Ready to start? Follow these simple steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python run_en.py

# 3. Open browser (automatic)
# http://localhost:5000

# 4. Make your first prediction!
```

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: ML Development Team

**Happy Predicting! 🚀**
