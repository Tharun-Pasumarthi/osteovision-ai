import {
  HealthResponse,
  ModelsListResponse,
  PredictionResponse,
  GradCAMResponse,
  DatasetSummaryResponse,
  BenchmarkMetricsResponse,
  ClinicalFeatureImportance,
  SampleImage,
  ClinicalFeatures,
  ImageQualityAssessment,
  ShapExplanationResponse,
  GlobalFeatureImportanceItem,
  CalibrationReportResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

class ApiClient {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        let errorMsg = `Server error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData && errorData.detail) {
            errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
          }
        } catch (_) {
          // Non-JSON response body
        }
        throw new Error(errorMsg);
      }
      return (await response.json()) as T;
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw err;
      }
      console.error(`API error on ${endpoint}:`, err);
      throw err;
    }
  }

  async getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health', { signal });
  }

  async getModels(signal?: AbortSignal): Promise<ModelsListResponse> {
    return this.request<ModelsListResponse>('/api/models', { signal });
  }

  async getMetrics(signal?: AbortSignal): Promise<BenchmarkMetricsResponse> {
    return this.request<BenchmarkMetricsResponse>('/api/metrics', { signal });
  }

  async getDatasetSummary(signal?: AbortSignal): Promise<DatasetSummaryResponse> {
    return this.request<DatasetSummaryResponse>('/api/dataset-summary', { signal });
  }

  async getClinicalImportance(signal?: AbortSignal): Promise<{ features: ClinicalFeatureImportance[] }> {
    return this.request<{ features: ClinicalFeatureImportance[] }>('/api/clinical-importance', { signal });
  }

  async getSampleImages(signal?: AbortSignal): Promise<{ samples: SampleImage[] }> {
    return this.request<{ samples: SampleImage[] }>('/api/sample-images', { signal });
  }

  getSampleImageUrl(sampleId: string): string {
    return `${API_BASE_URL}/api/sample-images/${sampleId}/file`;
  }

  async assessImageQuality(file: File | Blob, signal?: AbortSignal): Promise<ImageQualityAssessment> {
    const formData = new FormData();
    formData.append('file', file, (file as File).name || 'radiograph.jpg');
    return this.request<ImageQualityAssessment>('/api/image-quality', {
      method: 'POST',
      body: formData,
      signal,
    });
  }

  async explainClinicalShap(
    clinicalData: ClinicalFeatures,
    targetClass?: string,
    signal?: AbortSignal
  ): Promise<ShapExplanationResponse> {
    const query = targetClass ? `?target_class=${encodeURIComponent(targetClass)}` : '';
    return this.request<ShapExplanationResponse>(`/api/shap-explain${query}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(clinicalData),
      signal,
    });
  }

  async getShapGlobal(signal?: AbortSignal): Promise<GlobalFeatureImportanceItem[]> {
    return this.request<GlobalFeatureImportanceItem[]>('/api/shap-global', { signal });
  }

  async getCalibrationStatus(signal?: AbortSignal): Promise<CalibrationReportResponse> {
    return this.request<CalibrationReportResponse>('/api/calibration-status', { signal });
  }

  async predict(
    file: File | Blob,
    modelId: string,
    clinicalData?: ClinicalFeatures,
    signal?: AbortSignal
  ): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file, (file as File).name || 'radiograph.jpg');
    formData.append('model_id', modelId);
    if (clinicalData && Object.keys(clinicalData).length > 0) {
      formData.append('clinical_data', JSON.stringify(clinicalData));
    }

    return this.request<PredictionResponse>('/api/predict', {
      method: 'POST',
      body: formData,
      signal,
    });
  }

  async predictClinical(
    clinicalData: ClinicalFeatures,
    signal?: AbortSignal
  ): Promise<PredictionResponse> {
    return this.request<PredictionResponse>('/api/predict-clinical', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(clinicalData),
      signal,
    });
  }

  async generateGradCAM(
    file: File | Blob,
    modelId: string,
    targetClass?: string,
    alpha: number = 0.45,
    colormap: string = 'jet',
    threshold: number = 0.0,
    signal?: AbortSignal
  ): Promise<GradCAMResponse> {
    const formData = new FormData();
    formData.append('file', file, (file as File).name || 'radiograph.jpg');
    formData.append('model_id', modelId);
    if (targetClass) {
      formData.append('target_class', targetClass);
    }
    formData.append('alpha', alpha.toString());
    formData.append('colormap', colormap);
    formData.append('threshold', threshold.toString());

    return this.request<GradCAMResponse>('/api/gradcam', {
      method: 'POST',
      body: formData,
      signal,
    });
  }
}

export const api = new ApiClient();
