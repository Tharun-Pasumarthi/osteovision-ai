import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Flame,
  BarChart2,
  Layers,
  Eye,
  Info,
  ShieldCheck,
  AlertCircle,
  Stethoscope,
  ScanEye,
  Sliders,
  HelpCircle,
  Activity,
  Check,
} from 'lucide-react';
import { api } from '../services/api';
import {
  GradCAMResponse,
  ShapExplanationResponse,
  GlobalFeatureImportanceItem,
  SampleImage,
  ClinicalFeatures,
} from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

export const ExplainableAIPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'image_cam' | 'clinical_shap'>('image_cam');

  // Grad-CAM State
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleImage | null>(null);
  const [camModelId, setCamModelId] = useState<string>('densenet121');
  const [camTargetClass, setCamTargetClass] = useState<string>('Osteoporosis');
  const [camAlpha, setCamAlpha] = useState<number>(0.45);
  const [camColormap, setCamColormap] = useState<string>('jet');
  const [camThreshold, setCamThreshold] = useState<number>(0.0);
  const [gradcam, setGradcam] = useState<GradCAMResponse | null>(null);
  const [isGeneratingCam, setIsGeneratingCam] = useState<boolean>(false);
  const [camViewMode, setCamViewMode] = useState<'overlay' | 'heatmap' | 'original' | 'split'>('overlay');

  // SHAP State
  const [globalImportance, setGlobalImportance] = useState<GlobalFeatureImportanceItem[]>([]);
  const [shapExplanation, setShapExplanation] = useState<ShapExplanationResponse | null>(null);
  const [isGeneratingShap, setIsGeneratingShap] = useState<boolean>(false);
  const [shapTargetClass, setShapTargetClass] = useState<string>('Osteoporosis');

  // Load sample images and global SHAP on mount
  useEffect(() => {
    let isMounted = true;
    api.getSampleImages()
      .then((res) => {
        if (isMounted && res.samples && res.samples.length > 0) {
          setSamples(res.samples);
          setSelectedSample(res.samples[0]);
          loadGradCAMForSample(res.samples[0], 'densenet121', 'Osteoporosis');
        }
      })
      .catch((err) => console.error('Failed to load sample radiographs:', err));

    api.getShapGlobal()
      .then((items) => {
        if (isMounted) setGlobalImportance(items);
      })
      .catch((err) => console.error('Failed to load global SHAP importance:', err));

    // Load default SHAP explanation
    loadDefaultShap('Osteoporosis');

    return () => {
      isMounted = false;
    };
  }, []);

  const loadGradCAMForSample = async (sample: SampleImage, modelId: string, targetClass: string) => {
    setIsGeneratingCam(true);
    try {
      const url = api.getSampleImageUrl(sample.id);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      const cam = await api.generateGradCAM(file, modelId, targetClass, camAlpha, camColormap, camThreshold);
      setGradcam(cam);
    } catch (err) {
      console.error('Failed to generate sample Grad-CAM:', err);
    } finally {
      setIsGeneratingCam(false);
    }
  };

  const loadDefaultShap = async (targetClass: string) => {
    setIsGeneratingShap(true);
    const demoPatient: ClinicalFeatures = {
      num_age: 68.0,
      num_height_m: 1.56,
      num_weight_kg: 52.0,
      num_bmi: 21.37,
      num_menopause_age: 44.0,
      num_max_walking_km: 1.2,
      gender_female: 1.0,
      joint_pain: 1.0,
      smoker: 0.0,
      diabetic: 0.0,
      hypothyroidism: 1.0,
      fracture: 1.0,
      family_history: 1.0,
    };
    try {
      const exp = await api.explainClinicalShap(demoPatient, targetClass);
      setShapExplanation(exp);
      setShapTargetClass(targetClass);
    } catch (err) {
      console.error('Failed to generate demo SHAP:', err);
    } finally {
      setIsGeneratingShap(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            Explainable AI (XAI) Intelligence Console
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Dual independent interpretability frameworks: Radiograph convolutional Grad-CAM and Clinical feature SHAP.
          </p>
        </div>

        {/* Clear Model Separation Badge */}
        <div className="p-2 rounded-lg bg-teal-50/60 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800 text-[11px] text-teal-900 dark:text-teal-200 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-teal-600" />
          <span>Independent Models: Vision activations vs. Clinical attributions do not mix.</span>
        </div>
      </div>

      {/* Main Tab Bar */}
      <div className="flex border-b border-slate-200 dark:border-slate-800">
        <button
          type="button"
          onClick={() => setActiveTab('image_cam')}
          className={`py-3 px-6 text-xs font-bold border-b-2 transition flex items-center gap-2 ${
            activeTab === 'image_cam'
              ? 'border-teal-500 text-teal-600 dark:text-teal-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <Flame className="w-4 h-4" />
          Image Explainability (Grad-CAM)
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('clinical_shap')}
          className={`py-3 px-6 text-xs font-bold border-b-2 transition flex items-center gap-2 ${
            activeTab === 'clinical_shap'
              ? 'border-teal-500 text-teal-600 dark:text-teal-400'
              : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          Clinical Feature Explainability (SHAP)
        </button>
      </div>

      {/* Tab 1: Image Grad-CAM */}
      {activeTab === 'image_cam' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column (5 cols): Sample Selector & Controls */}
          <div className="lg:col-span-5 space-y-5">
            <Card>
              <CardHeader
                title="Radiograph Sample Inspection"
                subtitle="Select verified patient radiograph from trial cohort to generate activation maps"
              />
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  {samples.map((s) => {
                    const isSelected = selectedSample?.id === s.id;
                    return (
                      <div
                        key={s.id}
                        onClick={() => {
                          setSelectedSample(s);
                          loadGradCAMForSample(s, camModelId, camTargetClass);
                        }}
                        className={`p-2.5 rounded-lg border cursor-pointer transition flex items-center justify-between ${
                          isSelected
                            ? 'border-teal-500 bg-teal-50/50 dark:bg-teal-950/40 text-teal-900 dark:text-teal-200'
                            : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900'
                        }`}
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold">{s.true_label}</span>
                            <span className="text-[10px] text-slate-400 font-mono">ID: {s.patient_id}</span>
                          </div>
                          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
                            {s.filename} &bull; {s.source}
                          </p>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-teal-600" />}
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader
                title="Grad-CAM Visualization Controls"
                subtitle="Colormaps, opacity blend, and class-specific gradient targeting"
              />
              <CardContent className="space-y-4 text-xs">
                {/* Target Class Selector */}
                <div>
                  <label className="block text-slate-500 font-medium mb-1.5">Class to Explain:</label>
                  <div className="grid grid-cols-3 gap-1.5">
                    {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cls) => (
                      <button
                        key={cls}
                        type="button"
                        onClick={() => {
                          setCamTargetClass(cls);
                          if (selectedSample) loadGradCAMForSample(selectedSample, camModelId, cls);
                        }}
                        className={`py-1.5 px-2 rounded-lg border text-center font-medium transition ${
                          camTargetClass === cls
                            ? 'border-teal-500 bg-teal-50 dark:bg-teal-950 text-teal-700 dark:text-teal-300 font-bold'
                            : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400'
                        }`}
                      >
                        {cls}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Colormap Selector */}
                <div>
                  <label className="block text-slate-500 font-medium mb-1.5">Diagnostic Colormap:</label>
                  <select
                    value={camColormap}
                    onChange={(e) => {
                      setCamColormap(e.target.value);
                      if (selectedSample) loadGradCAMForSample(selectedSample, camModelId, camTargetClass);
                    }}
                    className="w-full p-2 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-medium"
                  >
                    <option value="jet">Jet (Classic Diagnostic)</option>
                    <option value="turbo">Turbo (Wide Dynamic Range)</option>
                    <option value="viridis">Viridis (Perceptually Uniform)</option>
                    <option value="inferno">Inferno (High Contrast Density)</option>
                    <option value="bone">Bone (Radiographic Monochromatic)</option>
                  </select>
                </div>

                {/* Alpha Slider */}
                <div>
                  <div className="flex justify-between text-slate-500 mb-1">
                    <span>Heatmap Opacity (Alpha):</span>
                    <span className="font-mono">{camAlpha.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={camAlpha}
                    onChange={(e) => setCamAlpha(parseFloat(e.target.value))}
                    onMouseUp={() => selectedSample && loadGradCAMForSample(selectedSample, camModelId, camTargetClass)}
                    className="w-full accent-teal-600 cursor-pointer"
                  />
                </div>

                {/* Threshold Slider */}
                <div>
                  <div className="flex justify-between text-slate-500 mb-1">
                    <span>Activation Floor Cutoff:</span>
                    <span className="font-mono">{Math.round(camThreshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="0.8"
                    step="0.05"
                    value={camThreshold}
                    onChange={(e) => setCamThreshold(parseFloat(e.target.value))}
                    onMouseUp={() => selectedSample && loadGradCAMForSample(selectedSample, camModelId, camTargetClass)}
                    className="w-full accent-teal-600 cursor-pointer"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column (7 cols): Visualizer & Explanation */}
          <div className="lg:col-span-7 space-y-5">
            <Card>
              <CardHeader
                title="Convolutional Activation Map"
                subtitle={`Target Conv Layer: ${camModelId === 'densenet121' ? 'denseblock4.denselayer16.conv2' : 'layer4.2.conv3'}`}
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
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                        }`}
                      >
                        {mode.charAt(0).toUpperCase() + mode.slice(1)}
                      </button>
                    ))}
                  </div>
                }
              />
              <CardContent className="space-y-4">
                {gradcam ? (
                  <div className="space-y-3">
                    {camViewMode === 'split' ? (
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex flex-col items-center">
                          <img src={gradcam.original_image_base64} alt="Original" className="max-h-64 object-contain" />
                          <span className="text-[10px] text-slate-300 py-1 font-mono">Original Radiograph</span>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex flex-col items-center">
                          <img src={gradcam.overlay_base64} alt="Overlay" className="max-h-64 object-contain" />
                          <span className="text-[10px] text-slate-300 py-1 font-mono">Grad-CAM Overlay</span>
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center p-2 min-h-72">
                        <img
                          src={
                            camViewMode === 'overlay'
                              ? gradcam.overlay_base64
                              : camViewMode === 'heatmap'
                              ? gradcam.heatmap_base64
                              : gradcam.original_image_base64
                          }
                          alt="Grad-CAM Display"
                          className="max-h-80 w-auto object-contain rounded"
                        />
                      </div>
                    )}

                    <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-800 dark:text-slate-200">
                          Class Attribution: {gradcam.target_class}
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          Predicted: {gradcam.predicted_class}
                        </span>
                      </div>
                      <p className="text-slate-600 dark:text-slate-400 leading-relaxed text-[11px]">
                        {gradcam.notes}
                      </p>
                      <p className="text-[10px] text-slate-400 dark:text-slate-500">
                        {gradcam.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-xs text-slate-400">
                    Loading Grad-CAM activation map...
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader
                title="Grad-CAM Methodology &amp; Mathematical Basis"
                subtitle="Class Activation Mapping via Guided Backpropagation Gradients"
              />
              <CardContent className="text-xs text-slate-600 dark:text-slate-400 space-y-2.5">
                <p>
                  Grad-CAM calculates the gradient of the classification score $y^c$ for class $c$ with respect to feature activation maps $A^k$ of the convolutional layer:
                </p>
                <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-slate-900 font-mono text-[11px] text-teal-700 dark:text-teal-300 text-center">
                  &alpha;_k^c = (1 / Z) &sum;_i &sum;_j (&part; y^c / &part; A_{"{i,j}"}^k)
                </div>
                <p>
                  The weights $\alpha_k^c$ capture the "importance" of feature map $k$ for target class $c$. A ReLU is applied to the weighted combination to highlight features with a positive influence on the selected bone mineral category.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Tab 2: Clinical SHAP */}
      {activeTab === 'clinical_shap' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column (6 cols): Local Waterfall Attribution */}
          <div className="lg:col-span-6 space-y-5">
            <Card>
              <CardHeader
                title="SHAP Local Waterfall Explanation"
                subtitle="Additive feature contributions for an individual patient profile"
                action={
                  <div className="flex items-center gap-1">
                    {(['Osteoporosis', 'Osteopenia', 'Normal'] as const).map((cls) => (
                      <button
                        key={cls}
                        type="button"
                        onClick={() => loadDefaultShap(cls)}
                        className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                          shapTargetClass === cls
                            ? 'bg-teal-600 text-white'
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                        }`}
                      >
                        {cls}
                      </button>
                    ))}
                  </div>
                }
              />
              <CardContent className="space-y-4 text-xs">
                {shapExplanation ? (
                  <>
                    <div className="p-3 rounded-xl bg-teal-50/50 dark:bg-teal-950/30 border border-teal-200/80 dark:border-teal-800/50 text-teal-900 dark:text-teal-200 text-[11px] leading-relaxed">
                      <span className="font-bold flex items-center gap-1.5 mb-1 text-teal-800 dark:text-teal-300">
                        <Sparkles className="w-3.5 h-3.5" />
                        Clinical Risk Narrative:
                      </span>
                      {shapExplanation.summary}
                    </div>

                    <div className="flex justify-between items-center text-[11px] text-slate-500 font-mono px-1">
                      <span>Baseline Population Expected: {(shapExplanation.base_value * 100).toFixed(1)}%</span>
                      <span>Model Probability: {(shapExplanation.final_predicted_value * 100).toFixed(1)}%</span>
                    </div>

                    <div className="space-y-1.5 border border-slate-200 dark:border-slate-800 rounded-xl p-3 bg-white dark:bg-slate-950">
                      {shapExplanation.waterfall_steps.map((step, idx) => {
                        const isPos = step.shap_value > 0;
                        const pct = Math.abs(step.shap_value) * 100;
                        return (
                          <div key={idx} className="flex items-center gap-2.5 text-xs">
                            <span className="w-36 font-medium text-slate-700 dark:text-slate-300 truncate text-[11px]">
                              {step.display_name}
                            </span>
                            <span className="w-20 text-[10px] text-slate-400 truncate font-mono">
                              {step.display_value}
                            </span>
                            <div className="flex-1 flex items-center h-3 relative">
                              <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${isPos ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                  style={{ width: `${Math.min(pct * 2, 100)}%` }}
                                />
                              </div>
                            </div>
                            <span className={`w-14 text-right font-mono font-semibold text-[11px] ${
                              isPos ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'
                            }`}>
                              {isPos ? '+' : ''}{(step.shap_value * 100).toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    <p className="text-[10px] text-slate-400 dark:text-slate-500">
                      {shapExplanation.disclaimer}
                    </p>
                  </>
                ) : (
                  <div className="h-48 flex items-center justify-center text-slate-400 text-xs">
                    Computing SHAP local attribution...
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column (6 cols): Global Feature Importance & Clinical Dictionary */}
          <div className="lg:col-span-6 space-y-5">
            <Card>
              <CardHeader
                title="Global Clinical Predictor Ranking"
                subtitle="Relative Gini importance across all 13 clinical features"
              />
              <CardContent>
                <div className="space-y-3">
                  {globalImportance.slice(0, 8).map((item, idx) => (
                    <div key={item.feature_name} className="space-y-1">
                      <div className="flex justify-between items-center text-xs">
                        <span className="font-semibold text-slate-800 dark:text-slate-200">
                          {idx + 1}. {item.display_name}
                        </span>
                        <span className="font-mono font-bold text-teal-600 dark:text-teal-400 text-[11px]">
                          {item.percentage.toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-teal-500 rounded-full transition-all duration-500"
                          style={{ width: `${item.percentage}%` }}
                        />
                      </div>
                      <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                        {item.description}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

