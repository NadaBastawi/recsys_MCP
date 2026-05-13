# TORCO Pest Control Recommendation System - Model Card

## Model Details

**Model Name:** TORCO Hybrid Recommender  
**Version:** 1.0.0  
**Date:** May 8, 2026  
**Authors:** AI Assistant (GitHub Copilot)  
**License:** MIT  

### Model Description
A hybrid recommendation system combining content-based filtering, collaborative filtering, and urgency scoring to recommend pest control services to customers in Tucson, Arizona. The system provides personalized service recommendations based on customer profiles, historical behavior, and contextual factors like season and time since last service.

### Model Architecture
- **Content-Based Filtering (40% weight):** Uses customer property features (type, age, location) and service affinity vectors
- **Collaborative Filtering (60% weight):** SVD matrix factorization on customer-service interaction matrix
- **Urgency Scorer:** Gradient Boosting Regressor predicting service urgency (1-10 scale)

## Intended Use

### Primary Use Cases
- Recommend pest control services to existing TORCO customers
- Provide cold-start recommendations for new customers
- Prioritize services based on urgency and seasonality

### Out-of-Scope Uses
- Real-time pest detection or identification
- Emergency response coordination
- Financial risk assessment

## Data

### Training Data
- **Customers:** 500 synthetic Tucson customers with property profiles
- **Interactions:** 3000 customer-service interactions with ratings (1-5)
- **Services:** 10 pest control services with pricing and categories
- **Time Period:** Multi-year historical data with seasonal patterns

### Data Sources
- Synthetic data generated using Faker library
- Realistic Tucson ZIP codes and property types
- Seasonally-adjusted interaction patterns (termite peaks in spring/summer)

### Data Limitations
- Synthetic data may not capture all real-world complexities
- Limited to Tucson metropolitan area
- No temporal trends beyond seasonal patterns

## Performance

### Metrics
- **Precision@5:** 96.2%
- **RMSE (Urgency Prediction):** 0.867
- **Coverage:** All 10 services can be recommended

### Evaluation Methodology
- Holdout evaluation: Last interaction per customer as ground truth
- 80/20 train/test split for urgency scorer
- Cross-validation not performed due to small dataset

### Performance Limitations
- Cold-start performance relies on profile features only
- May not generalize to non-Tucson locations
- Performance degrades with very sparse interaction data

## Ethical Considerations

### Fairness
- System may perpetuate biases in historical interaction data
- Property type and location features could lead to discriminatory recommendations
- No explicit fairness constraints implemented

### Privacy
- Stores customer property information and service history
- No personally identifiable information beyond customer IDs
- Model outputs do not include sensitive customer data

### Safety
- Recommendations are suggestions, not mandates
- High urgency scores should not trigger automatic actions
- Human oversight recommended for critical decisions

## Recommendations

### Monitoring
- Monitor prediction drift using KS test and PSI metrics
- Track recommendation acceptance rates
- Validate performance on new customer segments

### Maintenance
- Retrain quarterly to capture seasonal patterns
- Update service catalog as new offerings are added
- Validate synthetic data assumptions against real data

### Deployment
- Deploy behind API with rate limiting
- Implement A/B testing for recommendation changes
- Maintain audit logs of all recommendations

## Technical Details

### Dependencies
- Python 3.11+
- scikit-learn 1.8.0
- pandas 3.0.2
- numpy 2.4.4
- FastAPI 0.136.1

### Hardware Requirements
- CPU: 2+ cores recommended
- RAM: 4GB minimum
- Storage: 500MB for model and data

### Model Size
- Serialized model: ~50MB
- Training time: ~30 seconds
- Inference time: <100ms per recommendation

## Contact Information

For questions about this model:
- Repository: [TORCO RecSys Project](https://github.com/torco/recsys)
- Documentation: See project README.md
- Issues: File GitHub issues for bugs or improvements