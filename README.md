# README

## Overview

This Python script performs data preprocessing, classification, and clustering on a dataset (Iris dataset used as an example). It includes data cleaning, feature scaling, and multiple machine learning model implementations to classify and cluster the dataset.

## Features

- **Data Loading**: Supports loading datasets, with Iris dataset as the default example.
- **Data Preprocessing**:
  - Handles missing values.
  - Scales numerical features using StandardScaler.
  - Encodes categorical target variables.
- **Classification Models**:
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
  - Naive Bayes
- **Clustering**:
  - K-Means Clustering
- **Evaluation**:
  - Prints accuracy and classification reports for classification models.
  - Visualizes feature importance and clustering results.

## Dependencies

Make sure to install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Replace the dataset URL with your dataset file path.
2. Customize preprocessing and feature handling as per your dataset.
3. Run the script to train models, evaluate their performance, and visualize results.

## Code Breakdown

1. **Data Loading**:
   - Loads dataset from a given URL or file path.
   - Assigns column names as required.
2. **Data Preprocessing**:
   - Cleans data by handling missing values.
   - Scales numeric features.
   - Encodes categorical variables (if applicable).
3. **Model Training**:
   - Trains Random Forest, Decision Tree, SVM, and Naive Bayes classifiers.
   - Performs K-Means clustering.
4. **Evaluation**:
   - Prints accuracy and classification reports for classifiers.
   - Visualizes feature importance for Random Forest.
   - Visualizes K-Means clustering results.

## Example Outputs

- **Accuracy**: Displays the accuracy of classifiers.
- **Classification Report**: Detailed metrics such as precision, recall, and F1-score.
- **Feature Importance**: Bar chart showing feature contributions in Random Forest.
- **Clustering Visualization**: Scatter plot of K-Means clusters.

---

## راهنمای استفاده

### معرفی

این اسکریپت پایتون برای پیش‌پردازش داده‌ها، دسته‌بندی و خوشه‌بندی یک مجموعه داده (مانند مجموعه داده Iris) استفاده می‌شود. شامل پاکسازی داده‌ها، مقیاس‌بندی ویژگی‌ها و پیاده‌سازی مدل‌های مختلف یادگیری ماشین برای دسته‌بندی و خوشه‌بندی است.

### ویژگی‌ها

- **بارگذاری داده‌ها**: پشتیبانی از بارگذاری مجموعه‌های داده با استفاده از مجموعه داده Iris به عنوان مثال پیش‌فرض.
- **پیش‌پردازش داده‌ها**:
  - مدیریت مقادیر گم‌شده.
  - مقیاس‌بندی ویژگی‌های عددی با استفاده از StandardScaler.
  - کدگذاری متغیرهای هدف دسته‌بندی.
- **مدل‌های دسته‌بندی**:
  - جنگل تصادفی (Random Forest)
  - درخت تصمیم (Decision Tree)
  - ماشین بردار پشتیبان (SVM)
  - نایو بیز (Naive Bayes)
- **خوشه‌بندی**:
  - خوشه‌بندی K-Means
- **ارزیابی**:
  - نمایش دقت و گزارش‌های دسته‌بندی برای مدل‌ها.
  - نمایش اهمیت ویژگی‌ها و نتایج خوشه‌بندی.

### وابستگی‌ها

اطمینان حاصل کنید که کتابخانه‌های زیر نصب شده‌اند:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### نحوه استفاده

1. آدرس URL مجموعه داده را با مسیر فایل مجموعه داده خود جایگزین کنید.
2. پیش‌پردازش و مدیریت ویژگی‌ها را با توجه به مجموعه داده خود سفارشی کنید.
3. اسکریپت را اجرا کنید تا مدل‌ها آموزش ببینند، عملکردشان ارزیابی شود و نتایج بصری شوند.

### ساختار کد

1. **بارگذاری داده‌ها**:
   - بارگذاری مجموعه داده از URL یا مسیر فایل مشخص شده.
   - تخصیص نام ستون‌ها در صورت نیاز.
2. **پیش‌پردازش داده‌ها**:
   - پاکسازی داده‌ها با مدیریت مقادیر گم‌شده.
   - مقیاس‌بندی ویژگی‌های عددی.
   - کدگذاری متغیرهای هدف دسته‌بندی (در صورت وجود).
3. **آموزش مدل‌ها**:
   - آموزش مدل‌های جنگل تصادفی، درخت تصمیم، SVM و نایو بیز.
   - خوشه‌بندی K-Means.
4. **ارزیابی**:
   - نمایش دقت و گزارش دسته‌بندی برای مدل‌ها.
   - نمایش اهمیت ویژگی‌ها برای مدل جنگل تصادفی.
   - نمایش نتایج خوشه‌بندی K-Means.

### خروجی‌های نمونه

- **دقت**: نمایش دقت مدل‌های دسته‌بندی.
- **گزارش دسته‌بندی**: متریک‌های دقیق مانند Precision، Recall و F1-Score.
- **اهمیت ویژگی‌ها**: نمودار میله‌ای از مشارکت ویژگی‌ها در جنگل تصادفی.
- **بصری‌سازی خوشه‌بندی**: نمودار پراکندگی از خوشه‌بندی K-Means.
