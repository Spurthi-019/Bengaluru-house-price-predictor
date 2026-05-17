# 🏡 Bengaluru House Price Predictor

A sophisticated machine learning-powered real estate valuation platform for Bengaluru properties. This Streamlit-based web application predicts property prices using a trained Random Forest model and provides comprehensive financial analysis tools for potential buyers.

---

## ✨ Key Features

### 1. **Intelligent Property Valuation Engine**
- **ML-Powered Predictions**: Uses a pre-trained Random Forest model to predict property prices based on structural metrics
- **Location-Based Pricing**: Factors in 30+ Bengaluru neighborhoods with their unique market dynamics
- **Smart Feature Extraction**: Automatically matches column names from the trained model for 100% accuracy

### 2. **Comprehensive Property Assessment**
- **Structural Metrics**: Total square footage, BHK (bedrooms), bathrooms
- **Property Characteristics**: Age/condition, furnishing status
- **Luxury Amenities**: 24/7 gated security, swimming pool & gym, reserved parking, power backup

### 3. **Business Logic Adjustments**
Automatically applies real-world price adjustments:
- **Furnishing Premium**: +₹5L (fully furnished), +₹2L (semi-furnished)
- **Age Depreciation**: -2% (1-5 years), -5% (5-10 years), -10% (10+ years)
- **Amenity Bonus**: +₹1.5L per luxury amenity

### 4. **Market Analytics Dashboard**
- **Real-Time Metrics**: 
  - Base ML valuation
  - Adjusted final valuation
  - Price per square foot calculation
- **Benchmark Comparison**: Compares your property against premium Bengaluru hubs:
  - Indiranagar: ₹12,000/sqft (Prime Central)
  - Whitefield: ₹7,500/sqft (IT Corridor)
  - Electronic City: ₹5,500/sqft (Software Hub)

### 5. **💰 Buyer's Financial Toolkit**
- **EMI Calculator**: 
  - Interactive sliders for loan amount (50-90%), tenure (5-30 years), interest rate (7-12%)
  - Real-time monthly EMI calculation using industry-standard formula
  - Shows total interest payable over loan duration
- **Statutory Fees Breakdown**:
  - Stamp duty: 5% of property value
  - Registration fee: 1% of property value
  - Legal & documentation: Flat ₹50,000
- **Total On-Road Cost**: Complete cost estimate for property purchase

---

## 🏗️ Project Structure

```
Bengaluru_House_Predictor/
├── app.py                      # Main Streamlit application
├── house_price_model.pkl       # Pre-trained Random Forest model
├── columns.json                # Feature columns list from training
├── requirements.txt            # Python dependencies
├── Untitled9.ipynb            # Jupyter notebook (model training reference)
└── README.md                   # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Spurthi-019/Bengaluru-house-price-predictor.git
cd Bengaluru_House_Predictor
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 How to Run

### Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

### Using the Application

1. **Enter Property Details** (Left Panel):
   - Select neighborhood/location from dropdown
   - Input total square footage (300-15,000 sqft)
   - Set BHK and bathroom count using sliders
   - Choose property age & condition
   - Select furnishing status
   - Pick luxury amenities (multi-select)

2. **Calculate Valuation** (Click "Calculate Property Valuation"):
   - ML model predicts base price
   - Business logic applies adjustments
   - Final valuation displayed with breakdown

3. **Review Results** (Right Panel):
   - Base ML valuation
   - Price per square foot
   - Final adjusted price
   - Applied adjustments itemized

4. **Market Comparison** (Benchmark Chart):
   - Visual bar chart showing your property's price/sqft vs. major hubs
   - Benchmark location details

5. **Financial Planning** (Buyer's Dashboard):
   - Use EMI calculator sliders to plan loan options
   - See statutory fees breakdown
   - Review total on-road cost

---

## 🤖 Machine Learning Model Details

### Model Type
**Random Forest Regressor**
- Ensemble learning method combining multiple decision trees
- Excellent for capturing non-linear relationships in real estate pricing

### Training Data
- Historical property transaction data from Bengaluru
- ~13,000+ property records
- Locations: 30+ neighborhoods across the city

### Features Used
1. **Total_sqft**: Property area in square feet
2. **Bathrooms**: Number of bathrooms
3. **BHK**: Bedroom-hall-kitchen configuration
4. **Location**: One-hot encoded 30+ location categories

### Model Performance
- Captures complex market dynamics
- Handles location-specific price variations
- Robust to outliers through ensemble approach

### Feature Extraction
The app intelligently:
- Reads actual feature names from the trained model (`model.feature_names_in_`)
- Falls back to `columns.json` if needed
- Handles case-insensitive location matching
- Ensures 100% input-output compatibility

---

## 💡 Price Adjustment Logic

### Furnishing Premium
```
Fully Furnished     → Add ₹5 Lakhs
Semi-Furnished      → Add ₹2 Lakhs
Unfurnished         → No adjustment
```

### Property Age Depreciation (Applied to Adjusted Price)
```
Brand New/<Const    → No depreciation
1-5 Years Old       → Deduct 2%
5-10 Years Old      → Deduct 5%
10+ Years Old       → Deduct 10%
```

### Luxury Amenities Bonus
```
Each Selected Amenity → Add ₹1.5 Lakhs
Example: 3 amenities → +₹4.5 Lakhs
```

---

## 📐 Financial Calculations

### EMI Calculation Formula
```
EMI = P × [r(1+r)^n] / [(1+r)^n - 1]

Where:
P = Loan Amount
r = Monthly Interest Rate (Annual Rate / 12 / 100)
n = Number of Months (Tenure in Years × 12)
```

### On-Road Cost Calculation
```
Total On-Road Cost = Property Price + Stamp Duty + Registration Fee + Legal Fees

Where:
Stamp Duty       = 5% of Property Value
Registration Fee = 1% of Property Value
Legal Fees       = ₹50,000 (Flat)
```

---

## 🎨 UI/UX Design

### Dark Theme with Neon Green Aesthetic
- **Primary Color**: Neon Green (#00ffcc)
- **Secondary Color**: Deep Blue (#0077ff)
- **Background**: Dark (#0d1117)
- **Glow Effects**: Text shadows for modern appearance

### Responsive Layout
- **Wide Layout**: Side-by-side columns for optimal space usage
- **Two-Panel Design**: 
  - Left: Input controls
  - Right: Real-time results
- **Metric Cards**: Professional KPI displays
- **Interactive Charts**: Streamlit native bar chart for benchmarks

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | Latest | Web framework & UI |
| **joblib** | Latest | Model serialization |
| **pandas** | Latest | Data manipulation |
| **numpy** | Latest | Numerical operations |
| **scikit-learn** | Latest | ML model utilities |

---

## 🔍 Technical Implementation

### Session State Management
- Persistent result display across reruns
- Efficient UI updates without recalculation

### Model Loading
- `@st.cache_resource` decorator for single load
- Fallback mechanism for feature name compatibility
- Error handling with user-friendly messages

### Location Matching Algorithm
```python
Tries matches in order:
1. Exact match: "location_Electronic City"
2. Lowercase match: "location_electronic city"
3. Title case match: "location_Electronic City"
4. Capital L match: "Location_Electronic City"
```

### Input Validation
- Location dropdown with "Select a location..." placeholder
- Range validation on numeric inputs
- Error messages for invalid selections

---

## 📊 Sample Calculation Example

**Input:**
- Location: Electronic City
- Area: 1,200 sqft
- BHK: 2, Bathrooms: 2
- Property Age: 5-10 Years Old
- Furnishing: Semi-Furnished
- Amenities: Swimming Pool & Gym, Covered Parking

**Process:**
1. ML Model predicts: ₹45 Lakhs
2. Furnishing bonus: +₹2 Lakhs → ₹47 Lakhs
3. Depreciation (5%): -₹2.35 Lakhs → ₹44.65 Lakhs
4. Amenities (2 × ₹1.5L): +₹3 Lakhs → ₹47.65 Lakhs

**Output:**
- Final Valuation: ₹47.65 Lakhs
- Price/Sqft: ₹3,971/sqft
- EMI (80% loan, 20yr, 9.5%): ₹31,200/month
- Total On-Road Cost: ₹53.17 Lakhs

---

## 🐛 Troubleshooting

### Issue: "Error parsing structural model features"
**Solution**: Ensure `house_price_model.pkl` and `columns.json` exist in the project directory

### Issue: App doesn't start
**Solution**: Run `pip install -r requirements.txt` again to ensure all packages are installed

### Issue: Location not found in dropdown
**Solution**: Ensure you selected a valid location from the predefined list

---

## 🔐 Data & Privacy

- No personal data collected
- All calculations are local (no cloud uploads)
- Model predictions based on historical market data
- Prices are estimates; actual market rates may vary

---

## 📈 Future Enhancements

- [ ] Integration with live real estate APIs
- [ ] Price trend analysis over time
- [ ] Neighborhood comparison analytics
- [ ] PDF report generation
- [ ] Multi-city expansion
- [ ] Advanced filtering & search
- [ ] User authentication & saved searches

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Spurthi-019**
- GitHub: [@Spurthi-019](https://github.com/Spurthi-019)
- Project: [Bengaluru House Price Predictor](https://github.com/Spurthi-019/Bengaluru-house-price-predictor)

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub!

For issues, questions, or suggestions, please open an [Issue](https://github.com/Spurthi-019/Bengaluru-house-price-predictor/issues) on GitHub.

---

## 📚 Resources & References

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Scikit-learn Random Forest**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- **Bengaluru Real Estate Data**: Historical transaction records from multiple sources
- **EMI Calculation Formula**: Standard banking industry formula

---

**Last Updated**: May 2026
**Version**: 1.0.0
