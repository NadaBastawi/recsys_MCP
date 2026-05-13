"""Test script for monitoring module"""
from monitoring.drift_detector import log_prediction, get_drift_report

# Log some test predictions
log_prediction('C0001', 'S01', 0.85, 7)
log_prediction('C0002', 'S02', 0.92, 8)
log_prediction('C0003', 'S03', 0.78, 6)
log_prediction('C0004', 'S05', 0.88, 7)
log_prediction('C0005', 'S07', 0.91, 9)

# Get drift report
report = get_drift_report()
print('✓ Drift report generated successfully')
print(f'Overall status: {report.get("overall_status", "N/A")}')
print(f'Total predictions: {report.get("total_predictions", 0)}')
