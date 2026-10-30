export interface HealthResponse {
  status: string;
  app_name: string;
  app_version: string;
  device: string;
  models_loaded: string[];
  checkpoints_found: Record<string, boolean>;
}

export interface ModelMetadata {
  id: string;
  name: string;
  family: string;
  input_type: string;
  supports_clinical: boolean;
  gradcam_supported: boolean;
  checkpoint_available: boolean;
  checkpoint_path?: string;
  test_accuracy?: number;
  test_macro_f1?: number;
  val_macro_f1?: number;
  normal_recall?: number;
  osteopenia_recall?: number;
  osteoporosis_recall?: number;
  inference_latency_ms?: number;
  parameter_count?: number;
  description: string;
}

export interface ModelsListResponse {
  models: ModelMetadata[];
  default_model_id: string;
}

export interface ClinicalFeatures {
  num_age?: number;
  num_height_m?: number;
  num_weight_kg?: number;
  num_bmi?: number;
  num_menopause_age?: number | null;
  num_max_walking_km?: number;
  gender_female: number;
  joint_pain: number;
  smoker: number;
  diabetic: number;
  hypothyroidism: number;
  fracture: number;
  family_history: number;
  // Optional legacy aliases for backwards compatibility
  age?: number;
  height_m?: number;
  weight_kg?: number;
  bmi?: number;
  gender?: string;
  menopause_age?: number;
  max_walking_km?: number;
  history_of_fracture?: boolean;
  family_history_osteoporosis?: boolean;
}

export interface ImageQualityAssessment {
  overall_status: 'acceptable' | 'review_recommended' | 'unusable';
  quality_score: number;
  is_acceptable: boolean;
  can_proceed_to_inference: boolean;
  filename: string;
  format: string;
  width: number;
  height: number;
  aspect_ratio: number;
  laplacian_variance: number;
  mean_brightness: number;
  contrast_std: number;
  blur_status: string;
  exposure_status: string;
  contrast_status: string;
  issues: string[];
  recommendations: string[];
  disclaimer: string;
}

export interface PredictionResponse {
  predicted_label: 'Normal' | 'Osteopenia' | 'Osteoporosis';
  confidence: number;
  probabilities: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  raw_probabilities?: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  model_id: string;
  model_name: string;
  input_type: string;
  inference_duration_ms: number;
  is_calibrated: boolean;
  calibrated_probabilities?: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  temperature_applied?: number;
  calibration_note: string;
  is_uncertain?: boolean;
  review_needed?: boolean;
  review_rationale?: string;
  model_version?: string;
  clinical_features_used?: string[];
  vision_probabilities?: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  clinical_probabilities?: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  image_meta?: {
    filename: string;
    width: number;
    height: number;
    format: string;
  };
  image_quality?: ImageQualityAssessment;
  gradcam?: any;
  disclaimer: string;
}

export interface GradCAMResponse {
  model_id: string;
  model_name: string;
  target_class: string;
  predicted_class: string;
  probabilities: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  original_image_base64: string;
  heatmap_base64: string;
  overlay_base64: string;
  alpha: number;
  notes: string;
  disclaimer: string;
}

export interface ShapContribution {
  feature_name: string;
  display_name: string;
  category: string;
  unit: string;
  description: string;
  feature_value?: any;
  standardized_value: number;
  display_value: string;
  shap_value: number;
  direction: 'increases_risk' | 'protective' | 'neutral';
  magnitude: number;
}

export interface WaterfallStep {
  feature_name: string;
  display_name: string;
  display_value: string;
  shap_value: number;
  start_value: number;
  end_value: number;
  direction: string;
}

export interface ShapExplanationResponse {
  model_name: string;
  predicted_class: string;
  target_class: string;
  calibrated_probabilities: {
    Normal: number;
    Osteopenia: number;
    Osteoporosis: number;
  };
  base_value: number;
  final_predicted_value: number;
  contributions: ShapContribution[];
  positive_factors: ShapContribution[];
  negative_factors: ShapContribution[];
  waterfall_steps: WaterfallStep[];
  summary: string;
  disclaimer: string;
}

export interface GlobalFeatureImportanceItem {
  feature_name: string;
  display_name: string;
  category: string;
  unit: string;
  importance_score: number;
  percentage: number;
  description: string;
}

export interface CalibrationBinData {
  bin_center: number;
  avg_confidence: number;
  raw_accuracy: number;
  calibrated_accuracy: number;
  sample_count: number;
}

export interface CalibrationProfile {
  model_family: string;
  calibration_method: string;
  fitted_temperature?: number;
  cv_folds?: number;
  validation_partition_samples?: number;
  raw_metrics: {
    ece: number;
    mce: number;
    brier_score: number;
    log_loss: number;
  };
  calibrated_metrics: {
    ece: number;
    mce: number;
    brier_score: number;
    log_loss: number;
  };
  ece_reduction_percent: number;
  status: string;
  reliability_diagram: CalibrationBinData[];
  notes: string;
}

export interface CalibrationReportResponse {
  image_model_calibration: CalibrationProfile;
  clinical_model_calibration: CalibrationProfile;
  calibration_thresholds: {
    well_calibrated_ece: string;
    moderate_ece: string;
    poor_ece: string;
    uncertainty_margin: number;
  };
  disclaimer: string;
}

export interface DatasetSummaryResponse {
  total_images_discovered: number;
  readable_images: number;
  unique_exact_sha256_hashes: number;
  dimension_summary: {
    min_width?: number;
    max_width?: number;
    mean_width?: number;
    min_height?: number;
    max_height?: number;
    mean_height?: number;
  };
  class_distribution: Record<string, number>;
  sources: Record<string, any>;
  metadata_summary: Record<string, any>;
  patient_split_counts: {
    train_patients: number;
    val_patients: number;
    test_patients: number;
    total_clean_cohort: number;
  };
  provenance_notes: Record<string, string>;
}

export interface BenchmarkModelRecord {
  experiment_id: string;
  model_name: string;
  model_family: string;
  input_type: string;
  dataset_config: string;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  val_macro_f1?: number;
  val_accuracy?: number;
  val_balanced_acc?: number;
  test_accuracy?: number;
  test_balanced_acc?: number;
  test_macro_precision?: number;
  test_macro_recall?: number;
  test_macro_f1?: number;
  test_weighted_f1?: number;
  normal_recall?: number;
  osteopenia_recall?: number;
  osteoporosis_recall?: number;
  roc_auc?: number;
  training_duration_s?: number;
  inference_latency_ms?: number;
  parameter_count?: number;
  checkpoint_path?: string;
  status: string;
  raw_metrics_val?: any;
  raw_metrics_test?: any;
}

export interface BenchmarkMetricsResponse {
  total_models_evaluated: number;
  best_validation_model: string;
  best_test_macro_f1_model: string;
  models: BenchmarkModelRecord[];
}

export interface ClinicalFeatureImportance {
  feature_name: string;
  importance_mean: number;
  importance_std: number;
}

export interface SampleImage {
  id: string;
  filename: string;
  true_label: string;
  patient_id: string;
  source: string;
  rel_path: string;
  clinical_data?: ClinicalFeatures;
}
