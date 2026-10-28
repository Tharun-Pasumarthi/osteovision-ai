"""
Clinical and textual feature engineering module with strict label leakage controls.
Processes patient clinical data strictly adhering to the 13 clinical features contract:
1. num_age
2. num_height_m
3. num_weight_kg
4. num_bmi
5. num_menopause_age
6. num_max_walking_km
7. gender_female
8. joint_pain
9. smoker
10. diabetic
11. hypothyroidism
12. fracture
13. family_history
"""
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ClinicalFeaturePreprocessor:
    """
    Leakage-safe clinical metadata preprocessor conforming strictly to the 13 clinical features contract.
    Fits imputers and standardizers strictly on the training partition.
    """
    def __init__(self):
        self.numerical_cols = [
            "age",
            "height_m",
            "weight_kg",
            "bmi",
            "menopause_age",
            "max_walking_km",
        ]
        self.binary_cols = [
            "gender_female",
            "joint_pain",
            "smoker",
            "diabetic",
            "hypothyroidism",
            "fracture",
            "family_history",
        ]
        
        # Internal state learned on training data
        self.medians: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = [f"num_{c}" for c in self.numerical_cols] + self.binary_cols
        self.is_fitted: bool = False

    def _extract_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names, datatypes, and derived fields strictly into the 13 feature schema."""
        data = df.copy()
        clean = pd.DataFrame(index=data.index)
        
        # Patient ID if present
        if "patient_id" in data.columns:
            clean["patient_id"] = data["patient_id"].astype(str).str.strip().str.upper()
        elif "Patient Id" in data.columns:
            clean["patient_id"] = data["Patient Id"].astype(str).str.strip().str.upper()

        # 1. Age (num_age)
        if "num_age" in data.columns:
            clean["age"] = pd.to_numeric(data["num_age"], errors="coerce")
        elif "age" in data.columns:
            clean["age"] = pd.to_numeric(data["age"], errors="coerce")
        elif "Age" in data.columns:
            clean["age"] = pd.to_numeric(data["Age"], errors="coerce")
        else:
            clean["age"] = pd.Series(np.nan, index=data.index)

        # 2. Height (num_height_m) & 3. Weight (num_weight_kg)
        if "num_height_m" in data.columns:
            h_vals = pd.to_numeric(data["num_height_m"], errors="coerce")
        elif "height_m" in data.columns:
            h_vals = pd.to_numeric(data["height_m"], errors="coerce")
        else:
            h_cols = [c for c in data.columns if "height" in c.lower()]
            h_vals = pd.to_numeric(data[h_cols[0]], errors="coerce") if h_cols else pd.Series(np.nan, index=data.index)

        if "num_weight_kg" in data.columns:
            w_vals = pd.to_numeric(data["num_weight_kg"], errors="coerce")
        elif "weight_kg" in data.columns:
            w_vals = pd.to_numeric(data["weight_kg"], errors="coerce")
        else:
            w_cols = [c for c in data.columns if "weight" in c.lower() and "normal" not in c.lower()]
            w_vals = pd.to_numeric(data[w_cols[0]], errors="coerce") if w_cols else pd.Series(np.nan, index=data.index)

        clean["height_m"] = h_vals
        clean["weight_kg"] = w_vals

        # 4. BMI (num_bmi): convention w / (h^2) when h > 0.5 and w > 10
        calculated_bmi = np.where((h_vals > 0.5) & (w_vals > 10), w_vals / (h_vals ** 2), np.nan)
        if "num_bmi" in data.columns:
            raw_bmi = pd.to_numeric(data["num_bmi"], errors="coerce")
            clean["bmi"] = np.where(pd.isna(raw_bmi) | (raw_bmi == 0), calculated_bmi, raw_bmi)
        elif "bmi" in data.columns:
            raw_bmi = pd.to_numeric(data["bmi"], errors="coerce")
            clean["bmi"] = np.where(pd.isna(raw_bmi) | (raw_bmi == 0), calculated_bmi, raw_bmi)
        else:
            bmi_cols = [c for c in data.columns if "bmi" in c.lower()]
            if bmi_cols:
                raw_bmi = pd.to_numeric(data[bmi_cols[0]], errors="coerce")
                clean["bmi"] = np.where(pd.isna(raw_bmi) | (raw_bmi == 0), calculated_bmi, raw_bmi)
            else:
                clean["bmi"] = calculated_bmi

        # 5. Menopause Age (num_menopause_age)
        if "num_menopause_age" in data.columns:
            clean["menopause_age"] = pd.to_numeric(data["num_menopause_age"], errors="coerce")
        elif "menopause_age" in data.columns:
            clean["menopause_age"] = pd.to_numeric(data["menopause_age"], errors="coerce")
        else:
            m_cols = [c for c in data.columns if "menopause" in c.lower()]
            clean["menopause_age"] = pd.to_numeric(data[m_cols[0]], errors="coerce") if m_cols else pd.Series(np.nan, index=data.index)

        # 6. Walking Distance (num_max_walking_km)
        if "num_max_walking_km" in data.columns:
            clean["max_walking_km"] = pd.to_numeric(data["num_max_walking_km"], errors="coerce")
        elif "max_walking_km" in data.columns:
            clean["max_walking_km"] = pd.to_numeric(data["max_walking_km"], errors="coerce")
        else:
            w_cols = [c for c in data.columns if "walking" in c.lower()]
            clean["max_walking_km"] = pd.to_numeric(data[w_cols[0]], errors="coerce") if w_cols else pd.Series(np.nan, index=data.index)

        # Binary parse helper
        def parse_bin(series: pd.Series) -> pd.Series:
            return series.astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in ["yes", "y", "1", "1.0", "true", "female"] else (0.0 if x in ["no", "n", "0", "0.0", "false", "male", "nan", "none", ""] else 0.0)
            )

        # 7. Gender Female (gender_female)
        if "gender_female" in data.columns:
            clean["gender_female"] = pd.to_numeric(data["gender_female"], errors="coerce").fillna(0.0).map(lambda x: 1.0 if x >= 0.5 else 0.0)
        elif "gender" in data.columns:
            clean["gender_female"] = data["gender"].astype(str).str.lower().str.strip().map(lambda x: 1.0 if x == "female" else 0.0)
        else:
            g_cols = [c for c in data.columns if "gender" in c.lower()]
            clean["gender_female"] = parse_bin(data[g_cols[0]]) if g_cols else pd.Series(1.0, index=data.index)

        # 8. Joint Pain (joint_pain)
        if "joint_pain" in data.columns:
            clean["joint_pain"] = parse_bin(data["joint_pain"])
        else:
            jp_cols = [c for c in data.columns if "joint" in c.lower()]
            clean["joint_pain"] = parse_bin(data[jp_cols[0]]) if jp_cols else pd.Series(0.0, index=data.index)

        # 9. Smoker (smoker)
        if "smoker" in data.columns:
            clean["smoker"] = parse_bin(data["smoker"])
        else:
            sm_cols = [c for c in data.columns if "smoker" in c.lower()]
            clean["smoker"] = parse_bin(data[sm_cols[0]]) if sm_cols else pd.Series(0.0, index=data.index)

        # 10. Diabetic (diabetic)
        if "diabetic" in data.columns:
            clean["diabetic"] = parse_bin(data["diabetic"])
        else:
            db_cols = [c for c in data.columns if "diabet" in c.lower()]
            clean["diabetic"] = parse_bin(data[db_cols[0]]) if db_cols else pd.Series(0.0, index=data.index)

        # 11. Hypothyroidism (hypothyroidism)
        if "hypothyroidism" in data.columns:
            clean["hypothyroidism"] = parse_bin(data["hypothyroidism"])
        else:
            hy_cols = [c for c in data.columns if "hypo" in c.lower()]
            clean["hypothyroidism"] = parse_bin(data[hy_cols[0]]) if hy_cols else pd.Series(0.0, index=data.index)

        # 12. Fracture (fracture)
        if "fracture" in data.columns:
            clean["fracture"] = parse_bin(data["fracture"])
        elif "history_of_fracture" in data.columns:
            clean["fracture"] = parse_bin(data["history_of_fracture"])
        else:
            fr_cols = [c for c in data.columns if "fracture" in c.lower()]
            clean["fracture"] = parse_bin(data[fr_cols[0]]) if fr_cols else pd.Series(0.0, index=data.index)

        # 13. Family History (family_history)
        if "family_history" in data.columns:
            clean["family_history"] = parse_bin(data["family_history"])
        elif "family_history_osteoporosis" in data.columns:
            clean["family_history"] = parse_bin(data["family_history_osteoporosis"])
        else:
            fm_cols = [c for c in data.columns if "family" in c.lower()]
            clean["family_history"] = parse_bin(data[fm_cols[0]]) if fm_cols else pd.Series(0.0, index=data.index)

        return clean

    def fit(self, train_df: pd.DataFrame) -> "ClinicalFeaturePreprocessor":
        """Compute training medians and fit StandardScaler on training numerical columns."""
        clean = self._extract_clean_dataframe(train_df)

        # 1. Learn medians for numerical columns
        for col in self.numerical_cols:
            valid_vals = clean[col].dropna()
            med = float(valid_vals.median()) if not valid_vals.empty else 0.0
            self.medians[col] = med

        # 2. Impute training numerical columns with training medians
        imputed_num = clean[self.numerical_cols].fillna(self.medians)

        # 3. Standardize numerical columns
        self.scaler = StandardScaler()
        self.scaler.fit(imputed_num)

        # 4. Feature names in exact order
        self.feature_names = [f"num_{c}" for c in self.numerical_cols] + self.binary_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform dataframe into standardized float32 (N, 13) feature array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        clean = self._extract_clean_dataframe(df)

        # Impute numerical features with training medians, then scale
        imputed_num = clean[self.numerical_cols].fillna(self.medians)
        scaled_num = self.scaler.transform(imputed_num)

        # Impute binary features with 0.0
        imputed_bin = clean[self.binary_cols].fillna(0.0).values.astype(np.float32)

        # Assemble exact 13 features matrix
        features = np.hstack([scaled_num, imputed_bin]).astype(np.float32)
        return features, self.feature_names

    def save(self, filepath: str) -> None:
        """Persist fitted preprocessor state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "ClinicalFeaturePreprocessor":
        """Load fitted preprocessor state from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

def build_feature_dictionary() -> Dict[str, Any]:
    """Generate documented 13-feature clinical metadata dictionary."""
    return {
        "features": {
            "num_age": {"description": "Age of participant in years", "type": "numerical", "imputation": "median", "order": 1},
            "num_height_m": {"description": "Height in meters", "type": "numerical", "imputation": "median", "order": 2},
            "num_weight_kg": {"description": "Body weight in kilograms", "type": "numerical", "imputation": "median", "order": 3},
            "num_bmi": {"description": "Body Mass Index (kg/m^2)", "type": "numerical", "imputation": "median", "order": 4},
            "num_menopause_age": {"description": "Age at menopause in years for post-menopausal female subjects", "type": "numerical", "imputation": "median", "order": 5},
            "num_max_walking_km": {"description": "Maximum daily walking distance in km", "type": "numerical", "imputation": "median", "order": 6},
            "gender_female": {"description": "Indicator: 1.0 for female, 0.0 for male", "type": "binary", "imputation": "0.0", "order": 7},
            "joint_pain": {"description": "Presence of joint pain (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 8},
            "smoker": {"description": "Smoking history (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 9},
            "diabetic": {"description": "Diagnosed diabetes mellitus (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 10},
            "hypothyroidism": {"description": "Diagnosed hypothyroidism (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 11},
            "fracture": {"description": "History of bone fractures (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 12},
            "family_history": {"description": "Family history of osteoporosis (1.0 = yes, 0.0 = no)", "type": "binary", "imputation": "0.0", "order": 13},
        },
        "excluded_target_leakage_fields": [
            "T-score Value (Directly determines osteopenia/osteoporosis diagnosis)",
            "Z-Score Value (Bone density reference score)",
            "Diagnosis (Target label to be predicted)",
            "Patient Id / S.No (Identification variables)",
            "Site (Constant acquisition field)",
        ],
    }
