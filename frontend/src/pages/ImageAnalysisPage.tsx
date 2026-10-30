import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  FileImage,
  X,
  AlertCircle,
  Sparkles,
  Flame,
  CheckCircle2,
  Info,
  RotateCcw,
  Eye,
  ScanEye,
  ShieldCheck,
  ZoomIn,
  Sliders,
  Layers,
  Activity,
  Check,
} from 'lucide-react';
import { api } from '../services/api';
import {
  PredictionResponse,
  GradCAMResponse,
  SampleImage,
  ImageQualityAssessment,
} from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

const CLASS_CONFIG = {
  Normal: {
    variant: 'normal' as const,
    bg: 'bg-emerald-500',
    lightBg: 'bg-emerald-50 dark:bg-emerald-950/40',
    text: 'text-emerald-700 dark:text-emerald-300',
    border: 'border-emerald-200 dark:border-emerald-800',
    desc: 'Bone mineral density and cortical margins appear preserved within expected healthy limits.',
  },
  Osteopenia: {
    variant: 'osteopenia' as const,
    bg: 'bg-amber-500',
    lightBg: 'bg-amber-50 dark:bg-amber-950/40',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-200 dark:border-amber-800',
    desc: 'Mild radiolucency with moderate trabecular attenuation indicative of low bone mineral mass.',
  },
  Osteoporosis: {
    variant: 'osteoporosis' as const,
    bg: 'bg-rose-500',
    lightBg: 'bg-rose-50 dark:bg-rose-950/40',
    text: 'text-rose-700 dark:text-rose-300',
    border: 'border-rose-200 dark:border-rose-800',
    desc: 'Severe bone mineral depletion, cortical thinning, and high fragility fracture risk.',
  },
};

const IMAGE_MODELS = [
  { id: 'densenet121', name: 'DenseNet121 (Recommended)', desc: 'Hierarchical trabecular micro-architecture feature extraction (92.37% Accuracy)' },
  { id: 'resnet50', name: 'ResNet50', desc: '50-layer deep residual network with identity shortcut mappings' },
  { id: 'efficientnet_b0', name: 'EfficientNet-B0', desc: 'Compound-scaled lightweight cortical bone feature extractor' },
  { id: 'mobilenet_v3_large', name: 'MobileNetV3-Large', desc: 'Edge-optimized hardware efficient convolutional neural network' },
];

const COLORMAP_OPTIONS = [
  { id: 'jet', label: 'Jet (Diagnostic Classic)' },
  { id: 'turbo', label: 'Turbo (High Dynamic Range)' },
  { id: 'viridis', label: 'Viridis (Perceptually Uniform)' },
  { id: 'inferno', label: 'Inferno (High Contrast)' },
  { id: 'bone', label: 'Bone (Radiographic Grayscale)' },
];

export const ImageAnalysisPage: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Uploaded / Sample radiograph state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('densenet121');

  // Pre-inference IQA state
  const [isAssessingQuality, setIsAssessingQuality] = useState<boolean>(false);
  const [imageQuality, setImageQuality] = useState<ImageQualityAssessment | null>(null);
  const [qualityError, setQualityError] = useState<string | null>(null);

  // Prediction state
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // Grad-CAM state
  const [isGeneratingCam, setIsGeneratingCam] = useState<boolean>(false);
  const [gradcam, setGradcam] = useState<GradCAMResponse | null>(null);
  const [camTargetClass, setCamTargetClass] = useState<string>('');
  const [camAlpha, setCamAlpha] = useState<number>(0.45);
  const [camColormap, setCamColormap] = useState<string>('jet');
  const [camThreshold, setCamThreshold] = useState<number>(0.0);
  const [camViewMode, setCamViewMode] = useState<'overlay' | 'heatmap' | 'original' | 'split'>('overlay');

  // Load sample test radiographs
  useEffect(() => {
    let isMounted = true;
    api.getSampleImages()
      .then((res) => {
        if (isMounted) setSamples(res.samples || []);
      })
      .catch((err) => console.error('Failed to load sample radiographs:', err));
    return () => {
      isMounted = false;
    };
  }, []);

  // Run IQA whenever file changes
  const handleFileSelected = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPrediction(null);
    setGradcam(null);
    setPredictionError(null);
    setQualityError(null);

    setIsAssessingQuality(true);
    try {
      const qRes = await api.assessImageQuality(file);
      setImageQuality(qRes);
    } catch (err: any) {
      console.error('Image quality assessment error:', err);
      setQualityError(err.message || 'Failed to complete image quality screening.');
    } finally {
      setIsAssessingQuality(false);
    }
  };

  const handleSelectSample = async (sample: SampleImage) => {
    try {
      const imgUrl = api.getSampleImageUrl(sample.id);
      const res = await fetch(imgUrl);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      handleFileSelected(file);
    } catch (err) {
      console.error('Failed to load sample file:', err);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const pred = await api.predict(selectedFile, selectedModelId);
      setPrediction(pred);
      if (pred.image_quality) {
        setImageQuality(pred.image_quality);
      }
      // Automatically generate Grad-CAM for predicted class
      handleGenerateCam(pred.predicted_label);
    } catch (err: any) {
      setPredictionError(err.message || 'Image inference failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleGenerateCam = async (targetCls?: string) => {
    if (!selectedFile) return;
    setIsGeneratingCam(true);
    const clsToExplain = targetCls || camTargetClass || prediction?.predicted_label || 'Osteoporosis';
    try {
      const cam = await api.generateGradCAM(
        selectedFile,
        selectedModelId,
        clsToExplain,
        camAlpha,
        camColormap,
        camThreshold
      );
      setGradcam(cam);
      setCamTargetClass(clsToExplain);
    } catch (err: any) {
      console.error('Grad-CAM generation error:', err);
    } finally {
      setIsGeneratingCam(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <ScanEye className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            Knee Radiograph Image Analysis
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Pre-inference radiographic quality assessment, deep convolutional bone mineral classification, and calibrated probabilities.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="teal" size="sm">
            Independent Vision Pathway
          </Badge>
          <Badge variant="blue" size="sm">
            4 CNN Backbones
          </Badge>
        </div>
      </div>

      {/* Main Grid: Upload & Quality (Left) vs Predictions & GradCAM (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 cols): Radiograph Upload, Sample Library, & IQA */}
        <div className="lg:col-span-5 space-y-5">
          {/* Upload Card */}
          <Card>
            <CardHeader
              title="Knee Radiograph Acquisition"
              subtitle="Upload standard AP or Lateral Knee Plain-Film Radiograph (PNG, JPG)"
            />
            <CardContent className="space-y-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelected(e.target.files[0]);
                  }
                }}
              />

              {!previewUrl ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                      handleFileSelected(e.dataTransfer.files[0]);
                    }
                  }}
                  className="border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-teal-500 dark:hover:border-teal-500 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition bg-slate-50/50 dark:bg-slate-900/50 hover:bg-teal-50/20 dark:hover:bg-teal-950/20"
                >
                  <div className="w-12 h-12 rounded-full bg-teal-100 dark:bg-teal-950/80 text-teal-600 dark:text-teal-400 flex items-center justify-center mb-3">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                    Click to select or drag &amp; drop knee X-ray
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Supports high-resolution plain-film radiographs up to 15MB
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center max-h-72">
                    <img
                      src={previewUrl}
                      alt="Selected radiograph preview"
                      className="max-h-72 w-auto object-contain"
                    />
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setImageQuality(null);
                        setPrediction(null);
                        setGradcam(null);
                      }}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/80 text-white hover:bg-rose-600 transition"
                      title="Clear image"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-slate-950/75 text-[11px] font-mono text-slate-200">
                      {selectedFile?.name}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      variant="outline"
                      size="sm"
                      className="flex-1"
                    >
                      Change Radiograph
                    </Button>
                    <Button
                      onClick={handleRunInference}
                      variant="primary"
                      size="sm"
                      className="flex-1"
                      disabled={isPredicting || (imageQuality?.can_proceed_to_inference === false)}
                      isLoading={isPredicting}
                      leftIcon={<Sparkles className="w-4 h-4" />}
                    >
                      Analyze Radiograph
                    </Button>
                  </div>
                </div>
              )}

              {/* Sample Images Palette */}
              {samples.length > 0 && (
                <div className="pt-2 border-t border-slate-100 dark:border-slate-800/80">
                  <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                    Verified Test Radiographs Library
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {samples.slice(0, 3).map((sample) => (
                      <button
                        key={sample.id}
                        type="button"
                        onClick={() => handleSelectSample(sample)}
                        className="text-left p-2 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-teal-500/60 dark:hover:border-teal-500/60 bg-white dark:bg-slate-900 transition hover:shadow-xs group"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] font-bold text-slate-700 dark:text-slate-300">
                            {sample.true_label}
                          </span>
                          <span className="text-[9px] text-teal-600 dark:text-teal-400 opacity-0 group-hover:opacity-100 transition">
                            Load &rarr;
                          </span>
                        </div>
                        <p className="text-[10px] text-slate-400 font-mono truncate">
                          {sample.patient_id}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Model Architecture Selector */}
          <Card>
            <CardHeader
              title="Convolutional Neural Backbone"
              subtitle="Select trained PyTorch vision model for independent feature extraction"
            />
            <CardContent className="space-y-2">
              {IMAGE_MODELS.map((m) => {
                const isSelected = selectedModelId === m.id;
                return (
                  <div
                    key={m.id}
                    onClick={() => setSelectedModelId(m.id)}
                    className={`p-2.5 rounded-lg border cursor-pointer transition ${
                      isSelected
                        ? 'border-teal-500 bg-teal-50/40 dark:bg-teal-950/30'
                        : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className={`text-xs font-semibold ${isSelected ? 'text-teal-700 dark:text-teal-300' : 'text-slate-800 dark:text-slate-200'}`}>
                        {m.name}
                      </span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />}
                    </div>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                      {m.desc}
                    </p>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          {/* Pre-Inference Image Quality Assessment (IQA) */}
          {selectedFile && (
            <Card className={imageQuality?.overall_status === 'unusable' ? 'border-rose-300 dark:border-rose-900' : ''}>
              <CardHeader
                title="Radiographic Image Quality Assessment"
                subtitle="Computer vision screening for blurriness, contrast, exposure, and dimensions"
                action={
                  isAssessingQuality ? (
                    <Badge variant="blue" size="sm">Analyzing...</Badge>
                  ) : imageQuality?.overall_status === 'acceptable' ? (
                    <Badge variant="teal" size="sm">Acceptable for Processing</Badge>
                  ) : imageQuality?.overall_status === 'review_recommended' ? (
                    <Badge variant="osteopenia" size="sm">Review Recommended</Badge>
                  ) : imageQuality?.overall_status === 'unusable' ? (
                    <Badge variant="osteoporosis" size="sm">Unusable Image</Badge>
                  ) : null
                }
              />
              <CardContent className="space-y-3 text-xs">
                {imageQuality && (
                  <>
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        <span className="text-slate-500 dark:text-slate-400">Quality Score:</span>
                        <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                          {imageQuality.quality_score} / 100
                        </p>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        <span className="text-slate-500 dark:text-slate-400">Resolution &amp; Ratio:</span>
                        <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                          {imageQuality.width}x{imageQuality.height} ({imageQuality.aspect_ratio}:1)
                        </p>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        <span className="text-slate-500 dark:text-slate-400">Laplacian Blur Var:</span>
                        <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                          {imageQuality.laplacian_variance}
                        </p>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        <span className="text-slate-500 dark:text-slate-400">Contrast Std Dev:</span>
                        <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                          {imageQuality.contrast_std}
                        </p>
                      </div>
                    </div>

                    {imageQuality.issues.length > 0 && (
                      <div className="p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 text-amber-900 dark:text-amber-200 text-[11px] space-y-1">
                        <div className="font-semibold flex items-center gap-1.5">
                          <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                          Screening Observations:
                        </div>
                        <ul className="list-disc pl-4 space-y-0.5">
                          {imageQuality.issues.map((issue, idx) => (
                            <li key={idx}>{issue}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <p className="text-[10px] text-slate-400 dark:text-slate-500 leading-tight">
                      {imageQuality.disclaimer}
                    </p>
                  </>
                )}
                {qualityError && (
                  <p className="text-xs text-rose-600 dark:text-rose-400">{qualityError}</p>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column (7 cols): Prediction Result, Probabilities & Grad-CAM */}
        <div className="lg:col-span-7 space-y-5">
          {predictionError && (
            <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Inference Error</p>
                <p className="mt-0.5">{predictionError}</p>
              </div>
            </div>
          )}

          {/* Prediction Result Card */}
          {prediction && (
            <Card className="border-t-4 border-t-teal-500">
              <CardHeader
                title="Deep Learning Classification Output"
                subtitle={`Evaluated via ${prediction.model_name} in ${prediction.inference_duration_ms} ms`}
                action={
                  <Badge variant={CLASS_CONFIG[prediction.predicted_label].variant} size="md">
                    {prediction.predicted_label}
                  </Badge>
                }
              />
              <CardContent className="space-y-4">
                {/* Confidence Hero Box */}
                <div className={`p-4 rounded-xl border ${CLASS_CONFIG[prediction.predicted_label].border} ${CLASS_CONFIG[prediction.predicted_label].lightBg} flex items-center justify-between`}>
                  <div>
                    <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
                      Calibrated Prediction Confidence
                    </span>
                    <h3 className={`text-2xl font-bold mt-0.5 ${CLASS_CONFIG[prediction.predicted_label].text}`}>
                      {(prediction.confidence * 100).toFixed(1)}% {prediction.predicted_label}
                    </h3>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-md">
                      {CLASS_CONFIG[prediction.predicted_label].desc}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] uppercase font-mono text-slate-400">Status</span>
                    <p className="text-xs font-bold text-teal-600 dark:text-teal-400">
                      {prediction.is_calibrated ? 'Calibrated Probabilities' : 'Raw Scores'}
                    </p>
                    {prediction.temperature_applied && (
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                        T = {prediction.temperature_applied}
                      </p>
                    )}
                  </div>
                </div>

                {/* Probability Distribution: Raw vs Calibrated Comparison */}
                <div className="space-y-2.5">
                  <div className="flex justify-between items-center text-xs font-semibold text-slate-700 dark:text-slate-300">
                    <span>Multiclass Probability Spectrum</span>
                    <span className="text-[11px] text-slate-400 font-normal">
                      Solid: Calibrated | Faded: Raw Model Logit Softmax
                    </span>
                  </div>

                  {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cls) => {
                    const calibVal = (prediction.calibrated_probabilities?.[cls] ?? prediction.probabilities[cls]) * 100;
                    const rawVal = (prediction.raw_probabilities?.[cls] ?? prediction.probabilities[cls]) * 100;
                    const isPred = prediction.predicted_label === cls;

                    return (
                      <div key={cls} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className={`font-medium ${isPred ? 'text-slate-900 dark:text-slate-100 font-bold' : 'text-slate-600 dark:text-slate-400'}`}>
                            {cls}
                          </span>
                          <span className="font-mono text-slate-700 dark:text-slate-300">
                            {calibVal.toFixed(1)}% <span className="text-slate-400 text-[10px]">(Raw: {rawVal.toFixed(1)}%)</span>
                          </span>
                        </div>
                        {/* Progress Bar with Dual Calibrated and Raw representation */}
                        <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden relative">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              cls === 'Normal' ? 'bg-emerald-500' : cls === 'Osteopenia' ? 'bg-amber-500' : 'bg-rose-500'
                            }`}
                            style={{ width: `${Math.max(calibVal, 1.5)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Clinical Decision Support Rationale */}
                {prediction.review_needed && (
                  <div className="p-3 rounded-lg bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200/80 dark:border-amber-800/50 text-xs text-amber-900 dark:text-amber-200 space-y-1">
                    <span className="font-semibold flex items-center gap-1.5 text-amber-800 dark:text-amber-300">
                      <AlertCircle className="w-3.5 h-3.5" />
                      Clinical Decision Support Notice:
                    </span>
                    <p className="leading-relaxed text-[11px]">
                      {prediction.review_rationale}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Interactive Grad-CAM Explainability Card */}
          <Card>
            <CardHeader
              title="Grad-CAM Radiographic Explainability"
              subtitle="Convolutional layer activation heatmaps localizing anatomical bone density features"
              action={
                <div className="flex items-center gap-1">
                  {(['overlay', 'heatmap', 'original', 'split'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setCamViewMode(mode)}
                      className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                        camViewMode === mode
                          ? 'bg-teal-600 text-white'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900'
                      }`}
                    >
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </button>
                  ))}
                </div>
              }
            />
            <CardContent className="space-y-4">
              {/* Image Display Canvas */}
              {gradcam ? (
                <div className="space-y-3">
                  {camViewMode === 'split' ? (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex flex-col items-center">
                        <img src={gradcam.original_image_base64} alt="Original" className="max-h-56 object-contain" />
                        <span className="text-[10px] text-slate-300 py-1 font-mono">Original Radiograph</span>
                      </div>
                      <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex flex-col items-center">
                        <img src={gradcam.overlay_base64} alt="Overlay" className="max-h-56 object-contain" />
                        <span className="text-[10px] text-slate-300 py-1 font-mono">Grad-CAM Overlay</span>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center p-2 min-h-64">
                      <img
                        src={
                          camViewMode === 'overlay'
                            ? gradcam.overlay_base64
                            : camViewMode === 'heatmap'
                            ? gradcam.heatmap_base64
                            : gradcam.original_image_base64
                        }
                        alt="Grad-CAM visualization"
                        className="max-h-80 w-auto object-contain rounded"
                      />
                    </div>
                  )}

                  {/* Grad-CAM Controls */}
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-3 text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-500 font-medium">Explain Target Class:</span>
                        <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-700 p-0.5 bg-white dark:bg-slate-800">
                          {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cls) => (
                            <button
                              key={cls}
                              type="button"
                              onClick={() => {
                                setCamTargetClass(cls);
                                handleGenerateCam(cls);
                              }}
                              className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                                (camTargetClass || gradcam.target_class) === cls
                                  ? 'bg-teal-600 text-white'
                                  : 'text-slate-600 dark:text-slate-300 hover:text-slate-900'
                              }`}
                            >
                              {cls}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-slate-500 font-medium">Colormap:</span>
                        <select
                          value={camColormap}
                          onChange={(e) => {
                            setCamColormap(e.target.value);
                          }}
                          className="px-2 py-1 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-[11px] font-medium text-slate-800 dark:text-slate-200"
                        >
                          {COLORMAP_OPTIONS.map((opt) => (
                            <option key={opt.id} value={opt.id}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Sliders: Alpha Blending & Threshold */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 border-t border-slate-200 dark:border-slate-800">
                      <div>
                        <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                          <span>Overlay Transparency (Alpha):</span>
                          <span className="font-mono">{camAlpha.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min="0.1"
                          max="0.9"
                          step="0.05"
                          value={camAlpha}
                          onChange={(e) => setCamAlpha(parseFloat(e.target.value))}
                          onMouseUp={() => handleGenerateCam()}
                          className="w-full accent-teal-600 cursor-pointer"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                          <span>Activation Cutoff:</span>
                          <span className="font-mono">{Math.round(camThreshold * 100)}%</span>
                        </div>
                        <input
                          type="range"
                          min="0.0"
                          max="0.8"
                          step="0.05"
                          value={camThreshold}
                          onChange={(e) => setCamThreshold(parseFloat(e.target.value))}
                          onMouseUp={() => handleGenerateCam()}
                          className="w-full accent-teal-600 cursor-pointer"
                        />
                      </div>
                    </div>

                    <div className="pt-2 text-[11px] text-slate-600 dark:text-slate-400 space-y-1">
                      <p>{gradcam.notes}</p>
                      <p className="text-[10px] text-slate-400 dark:text-slate-500">{gradcam.disclaimer}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-56 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl flex flex-col items-center justify-center text-center p-6 text-slate-400">
                  <Flame className="w-8 h-8 mb-2 text-slate-300 dark:text-slate-600" />
                  <p className="text-xs font-medium">Grad-CAM visualization will appear here after analysis.</p>
                  <p className="text-[11px] mt-0.5">Upload a Knee X-ray and click "Analyze Radiograph" above.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
