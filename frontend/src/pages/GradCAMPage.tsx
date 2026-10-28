import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Flame,
  Download,
  Upload,
  Layers,
  ChevronDown,
  Info,
  Sliders,
  Sparkles,
  AlertCircle,
  FileImage,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../services/api';
import { GradCAMResponse, ModelsListResponse, SampleImage } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

export const GradCAMPage: React.FC = () => {
  const location = useLocation();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>('densenet121');
  const [targetClass, setTargetClass] = useState<string>('Osteoporosis');
  const [alpha, setAlpha] = useState<number>(0.45);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedSampleInfo, setSelectedSampleInfo] = useState<SampleImage | null>(null);

  // Grad-CAM response state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [gradcamData, setGradcamData] = useState<GradCAMResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchSamples = async () => {
      try {
        const [modelsRes, samplesRes] = await Promise.all([
          api.getModels(),
          api.getSampleImages(),
        ]);
        if (isMounted) {
          setSamples(samplesRes.samples);
          // If state passed from Analysis page
          if (location.state?.modelId) {
            setSelectedModelId(location.state.modelId);
          }
          if (location.state?.targetClass) {
            setTargetClass(location.state.targetClass);
          }
          // Default load the first sample image so page shows authentic Grad-CAM immediately
          if (samplesRes.samples.length > 0 && !selectedFile) {
            loadSampleImage(samplesRes.samples[1]); // e.g. OP10 or OS14
          }
        }
      } catch (err) {
        console.error('Error loading initial Grad-CAM data:', err);
      }
    };
    fetchSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  const loadSampleImage = async (sample: SampleImage) => {
    setSelectedSampleInfo(sample);
    setErrorMessage(null);
    try {
      const imgUrl = api.getSampleImageUrl(sample.id);
      const resp = await fetch(imgUrl);
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, { type: blob.type || 'image/jpeg' });
      setSelectedFile(file);
      // Run Grad-CAM automatically
      executeGradCAM(file, selectedModelId, targetClass, alpha);
    } catch (err) {
      setErrorMessage(`Failed to load test sample ${sample.filename}`);
    }
  };

  const handleFileChange = (file: File) => {
    setSelectedSampleInfo(null);
    setErrorMessage(null);
    setSelectedFile(file);
    executeGradCAM(file, selectedModelId, targetClass, alpha);
  };

  const executeGradCAM = async (
    file: File,
    modelId: string,
    target: string,
    overlayAlpha: number
  ) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const res = await api.generateGradCAM(file, modelId, target, overlayAlpha);
      setGradcamData(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to generate Grad-CAM visualization.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelOrTargetChange = (newModelId?: string, newTarget?: string, newAlpha?: number) => {
    const m = newModelId ?? selectedModelId;
    const t = newTarget ?? targetClass;
    const a = newAlpha ?? alpha;
    if (newModelId) setSelectedModelId(newModelId);
    if (newTarget) setTargetClass(newTarget);
    if (newAlpha !== undefined) setAlpha(newAlpha);

    if (selectedFile) {
      executeGradCAM(selectedFile, m, t, a);
    }
  };

  const handleDownloadOverlay = () => {
    if (!gradcamData?.overlay_base64) return;
    const link = document.createElement('a');
    link.href = gradcamData.overlay_base64;
    link.download = `gradcam_${gradcamData.model_id}_${gradcamData.target_class.toLowerCase()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Flame className="w-5 h-5 text-amber-500" />
            Grad-CAM Visual Explainability Inspector
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Compute gradient-weighted class activation heatmaps on authentic knee radiographs.
          </p>
        </div>

        {/* Download Overlay Action */}
        {gradcamData && (
          <Button
            onClick={handleDownloadOverlay}
            variant="outline"
            size="sm"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Download Overlay
          </Button>
        )}
      </div>

      {/* Quick Test Samples Selector */}
      {samples.length > 0 && (
        <div className="bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/80 rounded-xl p-3 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300 shrink-0">
            <Sparkles className="w-4 h-4 text-teal-600" />
            <span>Select Test Radiograph:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => loadSampleImage(s)}
                className={`px-2.5 py-1 rounded-md text-xs border transition-all cursor-pointer flex items-center gap-1.5 ${
                  selectedSampleInfo?.id === s.id
                    ? 'bg-teal-50 dark:bg-teal-950 border-teal-500 text-teal-700 dark:text-teal-300 font-semibold'
                    : 'bg-white dark:bg-slate-850 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:border-teal-500'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    s.true_label === 'Normal'
                      ? 'bg-emerald-500'
                      : s.true_label === 'Osteopenia'
                      ? 'bg-amber-500'
                      : 'bg-rose-500'
                  }`}
                />
                <span>{s.patient_id}</span>
                <span className="text-[10px] text-slate-400">({s.true_label})</span>
              </button>
            ))}
          </div>
          <div className="ml-auto">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-xs text-teal-600 hover:text-teal-700 dark:text-teal-400 font-medium underline flex items-center gap-1 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              Upload Custom Image
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
              className="hidden"
            />
          </div>
        </div>
      )}

      {/* Control Panel Card */}
      <Card>
        <CardContent className="p-4 grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
          {/* Target Layer Model */}
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Target CNN Model:
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => handleModelOrTargetChange(e.target.value, undefined, undefined)}
              className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold"
            >
              <option value="densenet121">DenseNet121 (features.norm5)</option>
              <option value="efficientnet_b0">EfficientNet-B0 (features[-1])</option>
              <option value="mobilenet_v3_large">MobileNetV3-Large (features[-1])</option>
              <option value="resnet50">ResNet50 (layer4)</option>
              <option value="multimodal_fusion">Multimodal Fusion (Image Branch)</option>
            </select>
          </div>

          {/* Target Class Selector */}
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Explain Target Class:
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {['Normal', 'Osteopenia', 'Osteoporosis'].map((cls) => (
                <button
                  key={cls}
                  onClick={() => handleModelOrTargetChange(undefined, cls, undefined)}
                  className={`py-1.5 px-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                    targetClass === cls
                      ? 'bg-slate-900 text-white dark:bg-teal-600 font-semibold shadow-xs'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200'
                  }`}
                >
                  {cls}
                </button>
              ))}
            </div>
          </div>

          {/* Opacity Slider */}
          <div>
            <div className="flex justify-between text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              <span>Overlay Opacity (α):</span>
              <span className="font-mono font-semibold text-teal-600 dark:text-teal-400">
                {Math.round(alpha * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={alpha}
              onChange={(e) => handleModelOrTargetChange(undefined, undefined, parseFloat(e.target.value))}
              className="w-full accent-teal-600"
            />
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-800 text-xs text-rose-700 dark:text-rose-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <p>{errorMessage}</p>
        </div>
      )}

      {/* Three Panel Visual Display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Panel 1: Original Knee Radiograph */}
        <Card className="flex flex-col">
          <CardHeader
            title="1. Native Knee Radiograph"
            subtitle="Original aspect ratio preserved (no letterbox distortion)"
            action={
              selectedSampleInfo && (
                <Badge variant={selectedSampleInfo.true_label === 'Normal' ? 'normal' : selectedSampleInfo.true_label === 'Osteopenia' ? 'osteopenia' : 'osteoporosis'} size="sm">
                  True: {selectedSampleInfo.true_label}
                </Badge>
              )
            }
          />
          <CardContent className="flex-1 flex flex-col items-center justify-center p-4 bg-black/95 rounded-b-xl">
            {gradcamData?.original_image_base64 ? (
              <img
                src={gradcamData.original_image_base64}
                alt="Native Knee Radiograph"
                className="w-full max-h-[380px] object-contain rounded-lg border border-slate-800"
              />
            ) : (
              <div className="h-64 flex items-center justify-center text-xs text-slate-500">
                {isLoading ? 'Processing radiograph...' : 'No image loaded'}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Panel 2: Grad-CAM Heatmap */}
        <Card className="flex flex-col">
          <CardHeader
            title="2. Grad-CAM Activation Heatmap"
            subtitle={`Spatial gradients for target '${targetClass}'`}
            action={
              <Badge variant="teal" size="sm">
                Colormap JET
              </Badge>
            }
          />
          <CardContent className="flex-1 flex flex-col items-center justify-center p-4 bg-black/95 rounded-b-xl">
            {gradcamData?.heatmap_base64 ? (
              <img
                src={gradcamData.heatmap_base64}
                alt="Grad-CAM Activation Heatmap"
                className="w-full max-h-[380px] object-contain rounded-lg border border-slate-800"
              />
            ) : (
              <div className="h-64 flex items-center justify-center text-xs text-slate-500">
                {isLoading ? 'Computing activations...' : 'Awaiting computation'}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Panel 3: Blended Overlay */}
        <Card className="flex flex-col">
          <CardHeader
            title="3. Diagnostic Heatmap Overlay"
            subtitle={`Blended with α = ${alpha.toFixed(2)}`}
            action={
              gradcamData && (
                <Badge variant={gradcamData.predicted_class === 'Normal' ? 'normal' : gradcamData.predicted_class === 'Osteopenia' ? 'osteopenia' : 'osteoporosis'} size="sm">
                  Pred: {gradcamData.predicted_class}
                </Badge>
              )
            }
          />
          <CardContent className="flex-1 flex flex-col items-center justify-center p-4 bg-black/95 rounded-b-xl">
            {gradcamData?.overlay_base64 ? (
              <img
                src={gradcamData.overlay_base64}
                alt="Blended Radiograph Overlay"
                className="w-full max-h-[380px] object-contain rounded-lg border border-slate-800 shadow-md"
              />
            ) : (
              <div className="h-64 flex items-center justify-center text-xs text-slate-500">
                {isLoading ? 'Generating overlay...' : 'Awaiting computation'}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Scientific Explanation & Diagnostic Notes Card */}
      {gradcamData && (
        <Card>
          <CardHeader
            title="Grad-CAM Interpretation & Clinical Caveat"
            subtitle="Gradient-weighted Class Activation Mapping methodology"
          />
          <CardContent className="text-xs space-y-3 text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              <strong className="text-slate-900 dark:text-slate-100">Physiological Significance:</strong> Red and warm
              regions indicate spatial coordinates where high feature activations combined with positive gradients
              increase the logit for <strong>{gradcamData.target_class}</strong>. In knee radiographs, these
              typically localize along the distal femoral condyles, proximal tibial plateau, and subchondral cancellous
              trabeculae where osteopenic thinning is prominent.
            </p>
            {gradcamData.model_id === 'multimodal_fusion' && (
              <div className="p-3 rounded-lg bg-teal-50 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800/60 text-teal-900 dark:text-teal-200">
                <strong>Multimodal Image-Branch Clarification:</strong> Grad-CAM visualizes activations in the image
                encoder branch (DenseNet121 backbone) only. The tabular clinical risk branch operates via MLP dense
                connections and is evaluated separately using permutation feature importance.
              </div>
            )}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-[11px] text-slate-500">
              <strong>Ethical Notice:</strong> Heatmap activations demonstrate neural correlation with learned training
              representations. They do not constitute formal histological or clinical causal evidence.
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
