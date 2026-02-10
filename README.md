# VOC-AQI Monitoring System with MQ Series Sensors

## Lung Disease Prediction & Air Quality Monitoring using Raspberry Pi

![System Architecture](Image%20(2).png)

---

##  System Overview

**Raspberry Pi Interface** integrates two key components for comprehensive health monitoring:

1. **VOC Detection** - Uses MQ series sensors to track volatile organic compounds linked to lung diseases
2. **AQI Monitoring (Model-0 SVM)** - Utilizes regional Air Quality Index data to enhance disease tracking

The interface effectively combines these elements to provide comprehensive health insights for respiratory disease prevention and monitoring.

---

## MQ Series Sensors - What They Detect

| Sensor | Target Gases | Applications |
|--------|-------------|--------------|
| **MQ-2** | LPG, Propane, Methane, Hydrogen, Smoke | General combustible gas detection, smoke detection |
| **MQ-3** | Alcohol, Ethanol, Benzene | Breathalyzer applications, alcohol vapor detection |
| **MQ-4** | Methane (CH₄), Natural Gas | Natural gas leak detection, methane monitoring |
| **MQ-5** | LPG, Natural Gas, Town Gas | Domestic gas leak detection |
| **MQ-6** | LPG, Propane, Isobutane | LPG-specific detection |
| **MQ-7** | Carbon Monoxide (CO) | CO poisoning prevention, indoor air quality |
| **MQ-8** | Hydrogen (H₂) | Hydrogen leak detection |
| **MQ-9** | Carbon Monoxide (CO), Methane | Dual gas detection for indoor safety |
| **MQ-135** | NH₃, NOₓ, Benzene, CO₂, Alcohol, Smoke | **Air quality monitoring**, pollution detection |
| **MQ-136** | Hydrogen Sulfide (H₂S) | Toxic gas detection, industrial safety |
| **MQ-137** | Ammonia (NH₃) | Agricultural, industrial ammonia detection |
| **MQ-138** | Benzene, Toluene, Alcohol, Acetone, Formaldehyde | **VOC detection**, indoor air quality |

### Key Sensors for Lung Disease Tracking:
- **MQ-135**: Ideal for general air quality index monitoring - detects multiple pollutants affecting respiratory health
- **MQ-138**: Specialized for VOC detection - formaldehyde, benzene, and other carcinogens linked to lung disease
- **MQ-7/MQ-9**: Carbon monoxide detection - critical for preventing CO poisoning and respiratory damage

---

## Predictive Analysis of COPD and Air Quality

#### Disclaimer:
Jupyter Notebook uploaded in this repo may occasionally show **render failure** due to github issues. If it happens, please refer to the link below to access Jupyter Notebook via nbviewer.
[**Link to Jupyter Notebook**](https://nbviewer.org/github/raachelssss/COPD-and-Air-Quality/blob/main/COPD_and_Low_Air_Quality.ipynb)

---

### Introduction
- This project aims to predict the onset of Chronic Obstructive Pulmonary Disease (COPD) by **analyzing risk factors that disproportionately affect underserved populations.** Key features include:
  - **Air Quality**: Research shows that poor air quality significantly increases the risk of COPD, with underserved communities often facing higher exposure.
  - **Race**: Data indicates that people of color are more likely to live in areas with poorer air quality [(Link to Article)](https://www.nytimes.com/2021/04/28/climate/air-pollution-minorities.html)
  - **Smoking**: Research suggests that smoking population is at more risk than non-smoking population

### ML/AI Techniques Used:
  - **Support Vector Machine (SVM)** - Linear, RBF, and Polynomial kernels
  - **Gradient Boosting Classifier** Best performing model (87.6% accuracy)
  - **Random Forest Classifier** (86.0% accuracy)
  - **Neural Network (MLP)** - Multi-layer Perceptron
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
  - Causal Inference (Outcome Regression, Inverse Propensity Weighting)
  - Frequentist & Bayesian General Linear Model (GLM) 

**By exploring these variables, this project seeks to contribute insights that could inform targeted interventions, particularly for those facing health inequities and limited access to quality care.**

---

## Quick Start

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests plotly
```

### Running the Notebook
1. Clone this repository
2. Open `COPD_and_Low_Air_Quality.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially

---

## Model Performance Summary

| Model | Test Accuracy | ROC-AUC |
|-------|--------------|---------|
| **Gradient Boosting** | **87.6%** | **0.944** |
| Random Forest | 86.0% | 0.924 |
| KNN | 76.0% | 0.830 |
| Neural Network (MLP) | 71.1% | 0.806 |
| SVM (RBF Kernel) | 70.2% | 0.787 |
| Logistic Regression | 62.8% | 0.644 |
| SVM (Linear) | 59.5% | 0.611 |

---

## Datasets

### 1. CDC Chronic Disease Indicators (COPD Data)
- **Source:** [CDC Open Data Portal](https://data.cdc.gov/resource/hksd-2xuw.csv)
- **Size:** 309,215 rows × 34 columns
- **Key Fields:** COPD prevalence rates, demographic stratification, geographic coordinates

### 2. Air Quality Data (Ozone & PM2.5)
- **Ozone (O₃):** Ground-level ozone concentration (ppb)
- **PM2.5:** Fine particulate matter (μg/m³)
- Both linked to respiratory disease risk

### 3. Training Dataset
- **Samples:** 602 records
- **Features:** COPD prevalence, ozone levels, PM2.5, latitude, longitude, year
- **Target:** Binary classification (High Risk / Low Risk)

---

### EDA:

#### Data Overview: 
- U.S. Chronic Disease Indicators: Chronic Obstructive Pulmonary Disease
- CDC: Daily Census-Tract PM 2.5 Concentration (allows checking of air pollution levels)
- CDC: Daily Census-Tract Ozone Concentrations
- State-level Population Density (2010-2019)
- County-level Smoking Prevalence Census
#### Data Visualization: 
- Relationship between smoking population and COPD incidence
- Comparing smoking population by race and gender with COPD patients


- Comparing Ozone ppb, PM 2.5 Concentration, and COPD Prevalence Rate by state using Geo Maps
    - It was observed that COPD prevalence rate distributions differ by gender and race.

- Is there a difference in air pollution between areas with predominantly minority populations and areas with predominantly White populations?
    - It was observed that areas with predominantly White populations have relatively lower air pollution compared to areas with predominantly minority populations.




### Research Question 1: Does air pollution cause COPD (chronic obstructive pulmonary disease)?
- Causal inference was used to verify the existence of causality.
  
**Outcome Regression:**
-   Fit a linear model of the following form:


- Confidence Interval at 5% Significance Level = [-0.661, 1.067]
- Due to its proximity to zero, the Outcome Regression could not establish a causal relationship.
- We assumed unconfoundedness that there are only 6 confounding variables that cause both treatment and outcome. There might be missing confounders not taken care of by the data and the model which lead us to this decision. Also, it is not clear that confounding variables would all have a linear effect on the prevalence of COPD. This leads us to an uncertainty that our linear model does not model any interactions between the variables.

  
**Inverse Propensity Weighting:**
- Propensity Score: P(State has a bad air quality | confounding variables) using logistic regression


  - IPW demonstrates the causal relationship between poor air quality and the onset of COPD under two assumptions:
  - 1) age-adjusted prevalence of COPD among adults is unconfounded given the variables
  - 2) propensity score is a good model of the probability of states that has a bad air quality given the listed confounding variables
  - IPW estimate means that **poor air quality causes people to have a 6.83% more prevalence** of being diagnosed with COPD than people who reside in good air quality.

    

### Research Question 2: Can we predict COPD occurrence based on a person's residential area?
As the causal relationship between air pollution and COPD could not be verified, the research direction was shifted to investigate whether COPD occurrence can be predicted based on residential area and race information.

Firstly, Bayesian GLM and Frequentist GLM models were adopted and trained.


**Bayesian GLM:**

- The Bayesian GLM model shows a possibility of overfitting.
- Hispanic individuals have the lowest probability of COPD compared to Multiracial individuals.
- The population living at a lower latitude and higher longitude may be more vulnerable to COPD.


**Frequentist GLM:**

- Hispanic, Multiracial, and White variables have p-values smaller than 0.05, indicating statistical significance.
- However, considering the high Pearson chi-squared value, the model may not fit the data well.

GLM has several inherent limitations. Firstly, it does not select features to incorporate into its model, but instead uses all the features provided. Taking all the features provided might be the reason why our model overfit to the data. Moreover, it has strict assumptions about the distributions’ shapes. Finally, it is sensitive to outliers and have lower predictive power than nonparametric models. 
We then chose the random forest model because it is less prone to overfitting compared to logistic regression GLMs because random forest consists of many classifiers that are independently trained on different subsets of the training data. Random forest, as an ensemble model that uses feature bagging, is less likely to overfit because it has low variance. 

**Random Forest Model:**


- The Feature Importance of variables indicating race is very low compared to latitude and longitude, suggesting that predicting COPD occurrence solely based on population demographics is challenging.
- It is regrettable that the performance of the model did not improve significantly even if different variables were adopted instead of race.
- The model showed an accuracy of around 98% on the train set and 82% on the test
  - **Limitation:**
    - One major limitation of random forest is that it cannot extrapolate. In other words, its prediction is only a mean of formerly observed labels. For instance, as shown in our EDA, the prevalence of COPD shows a yearly increasing trend up until 2017, and gradually decreases in the following years. Since the model cannot extrapolate, if the training data is missing year data, the model will likely under-predict or over-predict. For instance, New Jersey's COPD data is missing for 2019, when the overall state's COPD prevalence was decreasing. Likely, our model could not make accurate predictions in cases like this.

### Conclusion
- **Causal Inference:**
  - Identified the causal effect of air quality on the prevalence of COPD, although the relationship is relevant to 2011 (our dataset) 
- **Predictive Analysis:**
  - Results from the Bayesian model suggest that it can predict that multiracial individuals are most likely to have COPD, given the same location. It also can be predict that the hispanic individuals are least likely to have COPD compared to multiracial individuals. 
  - Key finding we were able to identify using Random Forest was that we can predict whether people have COPD by using longitude and latitude data. As suggested by feature importance, we could not predict whether people have COPD by using racial features. They had little to no feature importance, which means that geographical data alone suffice to predict COPD. This also means that excluding racial features will have little to no impact on the accuracy of the model.
- **Machine Learning Models (NEW):**
  - **Gradient Boosting achieved 87.6% accuracy** with ROC-AUC of 0.944, outperforming all other models
  - SVM models with different kernels were implemented and compared
  - Neural Network (MLP) achieved 71.1% accuracy
  - Feature importance analysis confirmed geographic location (longitude/latitude) as the most predictive features

---

### Discussion: Rooms for Improvement, Implications for Public Health & Society: 
- **Poor air quality causes COPD. Now what?**
  - The states with bad air quality can improve their legislation that regulates air pollutants generated by transportation, and manufacturers, or place more air filtering technologies to control air pollutants generated from wildfires. So that individuals could be less exposed to ozone and  pm 2.5 concentration which also decreases the prevalence of COPD.
- **We can predict the onset of COPD using geographical location. Now what?**
  - Based on random forest results, we have identified that we can predict the onset of COPD from geographical location. Hence, medical institutions and the federal government could further investigate geographical attributes of different locations to predict areas that are most likely to be prone to COPD. Through prediction, they can plan medical interventions in these areas, such as periodic diagnostic assessment, recommendation of clinic visit, symptom checks, and more. 

---

## 🔮 Future Improvements
- [ ] Implement deep learning models (CNN, LSTM)
- [ ] Add real-time air quality data integration
- [ ] Develop a web interface for predictions
- [ ] Include additional demographic features
- [ ] Temporal analysis with time-series models

---

