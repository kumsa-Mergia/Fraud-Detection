# XGBoost Model

## 🔍 SHAP Summary Plot Interpretation

The SHAP summary plot below illustrates the global feature importance based on average impact on the model’s output:

- **`time_since_signup`** is the most critical driver of fraud prediction. Recent sign-ups are more likely to be flagged as fraud.
- **`hour_of_day`** and **`day_of_week`** show that temporal patterns strongly influence predictions.
- **`age`** and **`purchase_value`** also play a notable role, possibly indicating risk tied to certain demographics or high-value transactions.
- Features such as **browser type**, **gender**, and **source channel** have minimal impact on the model’s decisions.

This suggests that behavioral and time-based features are the key signals the XGBoost model uses to detect fraud.


| **Feature**             | **Insight**                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **`time_since_signup`** | The most influential feature. Users who signed up recently may be more likely to be flagged as fraud.                  |
| **`hour_of_day`**       | The time of day significantly affects predictions — fraud might occur more during off-peak hours.                      |
| **`day_of_week`**       | Certain days may see higher fraud patterns (e.g., weekends vs. weekdays).                                              |
| **`age`**               | User age has moderate influence — perhaps younger or older users are more at risk.                                     |
| **`purchase_value`**    | High-value transactions could trigger suspicion, making this an important fraud signal.                                |
| **Other features**      | Variables like `source_Direct`, `sex_M`, and browser types have **minimal impact** on model predictions in comparison. |
