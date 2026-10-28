import React, { useState, useEffect, useRef } from 'react';
import {
  Stethoscope,
  Upload,
  FileImage,
  X,
  AlertCircle,
  Sparkles,
  Flame,
  CheckCircle2,
  ArrowRight,
  Info,
  RotateCcw,
  Check,
  Eye,
  EyeOff,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Layers,
  ScanEye,
  ShieldCheck,
} from 'lucide-react';
import { api } from '../services/api';
import {
  PredictionResponse,
  GradCAMResponse,
  SampleImage,
  ClinicalFeatures,
} from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { NumericInputField } from '../components/clinical/NumericInputField';
import { HeightConverter } from '../components/clinical/HeightConverter';
import { BmiCalculator } from '../components/clinical/BmiCalculator';
import { MenopauseSelector, MenopausalStatus } from '../components/clinical/MenopauseSelector';
import { WalkingDistanceConverter } from '../components/clinical/WalkingDistanceConverter';

const CLASS_CONFIG = {
  Normal: {
    variant: 'normal' as const,
    bg: 'bg-emerald-500',
    lightBg: 'bg-emerald-50 dark:bg-emerald-950/40',
    text: 'text-emerald-700 dark:text-emerald-300',
    border: 'border-emerald-200 dark:border-emerald-800',
    classIndex: 0,
    desc: 'Bone mineral density and cortical margins appear preserved within expected healthy limits.',
  },
  Osteopenia: {
    variant: 'osteopenia' as const,
    bg: 'bg-amber-500',
    lightBg: 'bg-amber-50 dark:bg-amber-950/40',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-200 dark:border-amber-800',
    classIndex: 1,
    desc: 'Mild radiolucency with moderate trabecular attenuation indicative of low bone mineral mass.',
  },
  Osteoporosis: {
    variant: 'osteoporosis' as const,
    bg: 'bg-rose-500',
    lightBg: 'bg-rose-50 dark:bg-rose-950/40',
    text: 'text-rose-700 dark:text-rose-300',
    border: 'border-rose-200 dark:border-rose-800',
    classIndex: 2,
    desc: 'Severe bone mineral depletion, cortical thinning, and high fragility fracture risk.',
  },
};

// Exact 13 features contract specification
const FEATURE_CONTRACT_SPECS = [
  { index: 1, key: 'num_age', label: 'num_age', unit: 'years', type: 'numerical', policy: 'median' },
  { index: 2, key: 'num_height_m', label: 'num_height_m', unit: 'metres', type: 'numerical', policy: 'median' },
  { index: 3, key: 'num_weight_kg', label: 'num_weight_kg', unit: 'kilograms', type: 'numerical', policy: 'median' },
  { index: 4, key: 'num_bmi', label: 'num_bmi', unit: 'kg/m²', type: 'numerical', policy: 'weight / height²' },
  { index: 5, key: 'num_menopause_age', label: 'num_menopause_age', unit: 'years', type: 'numerical', policy: 'median (females only)' },
  { index: 6, key: 'num_max_walking_km', label: 'num_max_walking_km', unit: 'kilometres', type: 'numerical', policy: 'median' },
  { index: 7, key: 'gender_female', label: 'gender_female', unit: '1.0 / 0.0', type: 'binary', policy: 'female=1.0, male=0.0' },
  { index: 8, key: 'joint_pain', label: 'joint_pain', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
  { index: 9, key: 'smoker', label: 'smoker', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
  { index: 10, key: 'diabetic', label: 'diabetic', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
  { index: 11, key: 'hypothyroidism', label: 'hypothyroidism', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
  { index: 12, key: 'fracture', label: 'fracture', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
  { index: 13, key: 'family_history', label: 'family_history', unit: '1.0 / 0.0', type: 'binary', policy: 'yes=1.0, no=0.0' },
];

export const XRayAnalysisPage: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Exact 13 features contract state (MANDATORY)
  const [numAge, setNumAge] = useState<number | null>(62);
  const [numHeightM, setNumHeightM] = useState<number | null>(1.58);
  const [numWeightKg, setNumWeightKg] = useState<number | null>(64.0);
  const [isFemale, setIsFemale] = useState<boolean>(true);
  const [menopausalStatus, setMenopausalStatus] = useState<MenopausalStatus>('post');
  const [numMenopauseAge, setNumMenopauseAge] = useState<number | null>(46);
  const [numMaxWalkingKm, setNumMaxWalkingKm] = useState<number | null>(2.5);
  const [activeBmi, setActiveBmi] = useState<number | null>(25.64);
  const [isBmiOverridden, setIsBmiOverridden] = useState<boolean>(false);

  // 6 binary clinical risk factors (1.0 = yes, 0.0 = no)
  const [jointPain, setJointPain] = useState<boolean>(true);
  const [smoker, setSmoker] = useState<boolean>(false);
  const [diabetic, setDiabetic] = useState<boolean>(false);
  const [hypothyroidism, setHypothyroidism] = useState<boolean>(false);
  const [fracture, setFracture] = useState<boolean>(false);
  const [familyHistory, setFamilyHistory] = useState<boolean>(true);

  // Optional Knee Radiograph State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);

  // Unified Diagnosis & Multimodal Fusion Execution State
  const [isDiagnosing, setIsDiagnosing] = useState<boolean>(false);
  const [diagnosisResult, setDiagnosisResult] = useState<PredictionResponse | null>(null);
  const [diagnosisError, setDiagnosisError] = useState<string | null>(null);

  // Grad-CAM Explainability State
  const [gradcamResult, setGradcamResult] = useState<GradCAMResponse | null>(null);
  const [isGeneratingGradcam, setIsGeneratingGradcam] = useState<boolean>(false);
  const [camTargetClass, setCamTargetClass] = useState<string>('Osteopenia');
  const [camColormap, setCamColormap] = useState<string>('jet');
  const [camAlpha, setCamAlpha] = useState<number>(0.50);

  // UI accordion toggles
  const [showContractSpecs, setShowContractSpecs] = useState<boolean>(false);
  const [showVisualGuide, setShowVisualGuide] = useState<boolean>(false);

  // Load sample test radiographs from verified repository
  useEffect(() => {
    let isMounted = true;
    const loadSamples = async () => {
      try {
        const res = await api.getSampleImages();
        if (isMounted) setSamples(res.samples || []);
      } catch (err) {
        console.error('Failed to load sample images:', err);
      }
    };
    loadSamples();
    return () => {
      isMounted = false;
    };
  }, []);

  // Update gender handler
  const handleGenderChange = (female: boolean) => {
    setIsFemale(female);
    if (!female) {
      setMenopausalStatus('pre_peri');
      setNumMenopauseAge(null);
    }
  };

  // Test Personas
  const applyPreset = (preset: 'normal' | 'osteopenia' | 'osteoporosis') => {
    if (preset === 'normal') {
      setNumAge(42);
      setNumHeightM(1.65);
      setNumWeightKg(62.0);
      setIsFemale(true);
      setMenopausalStatus('pre_peri');
      setNumMenopauseAge(null);
      setNumMaxWalkingKm(5.0);
      setJointPain(false);
      setSmoker(false);
      setDiabetic(false);
      setHypothyroidism(false);
      setFracture(false);
      setFamilyHistory(false);
    } else if (preset === 'osteopenia') {
      setNumAge(58);
      setNumHeightM(1.58);
      setNumWeightKg(62.0);
      setIsFemale(true);
      setMenopausalStatus('post');
      setNumMenopauseAge(46);
      setNumMaxWalkingKm(2.0);
      setJointPain(true);
      setSmoker(false);
      setDiabetic(true);
      setHypothyroidism(false);
      setFracture(false);
      setFamilyHistory(true);
    } else {
      setNumAge(74);
      setNumHeightM(1.52);
      setNumWeightKg(48.0);
      setIsFemale(true);
      setMenopausalStatus('post');
      setNumMenopauseAge(44);
      setNumMaxWalkingKm(0.8);
      setJointPain(true);
      setSmoker(false);
      setDiabetic(false);
      setHypothyroidism(true);
      setFracture(true);
      setFamilyHistory(true);
    }
    setDiagnosisError(null);
  };

  // Comprehensive Form Validation before prediction
  const validateForm = (): string | null => {
    if (numAge === null || isNaN(numAge) || numAge < 18 || numAge > 120) {
      return 'Please enter a valid Patient Age between 18 and 120 years.';
    }
    if (numHeightM === null || isNaN(numHeightM) || numHeightM < 0.50 || numHeightM > 2.50) {
      return 'Please enter a valid Height between 0.50 m (50 cm) and 2.50 m (250 cm).';
    }
    if (numWeightKg === null || isNaN(numWeightKg) || numWeightKg < 20 || numWeightKg > 250) {
      return 'Please enter a valid Weight between 20 kg and 250 kg.';
    }
    if (numMaxWalkingKm === null || isNaN(numMaxWalkingKm) || numMaxWalkingKm < 0.0 || numMaxWalkingKm > 50.0) {
      return 'Please enter a valid Max Walking Distance between 0.0 and 50.0 km.';
    }
    if (isFemale && menopausalStatus === 'post' && numMenopauseAge !== null) {
      if (isNaN(numMenopauseAge) || numMenopauseAge < 20.0 || numMenopauseAge > 80.0) {
        return 'Menopause onset age must be between 20 and 80 years (or left blank for cohort median).';
      }
    }
    return null;
  };

  // Build the validated 13-feature payload strictly matching the contract
  const buildContractPayload = (): ClinicalFeatures => {
    return {
      num_age: Number(numAge),
      num_height_m: Number(numHeightM),
      num_weight_kg: Number(numWeightKg),
      num_bmi: activeBmi !== null ? Number(activeBmi) : undefined,
      num_menopause_age:
        isFemale && menopausalStatus === 'post' && numMenopauseAge !== null && !isNaN(numMenopauseAge) && numMenopauseAge > 0
          ? Number(numMenopauseAge)
          : null,
      num_max_walking_km: Number(numMaxWalkingKm),
      gender_female: isFemale ? 1.0 : 0.0,
      joint_pain: jointPain ? 1.0 : 0.0,
      smoker: smoker ? 1.0 : 0.0,
      diabetic: diabetic ? 1.0 : 0.0,
      hypothyroidism: hypothyroidism ? 1.0 : 0.0,
      fracture: fracture ? 1.0 : 0.0,
      family_history: familyHistory ? 1.0 : 0.0,
    };
  };

  // Radiograph selection handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      selectImageFile(file);
    }
  };

  const selectImageFile = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setGradcamResult(null);
  };

  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setGradcamResult(null);
  };

  const handleSelectSample = async (sample: SampleImage) => {
    try {
      const res = await fetch(api.getSampleImageUrl(sample.id));
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: blob.type || 'image/jpeg' });
      selectImageFile(file);
    } catch (err) {
      console.error('Failed to load sample image:', err);
      setDiagnosisError('Could not load selected research sample.');
    }
  };

  // ========================================================
  // Unified Diagnosis Execution
  // Clinical Data is Mandatory; Knee X-ray is Optional
  // ========================================================
  const handleRunDiagnosis = async () => {
    const valError = validateForm();
    if (valError) {
      setDiagnosisError(valError);
      return;
    }

    setIsDiagnosing(true);
    setDiagnosisError(null);

    const payload = buildContractPayload();

    try {
      if (selectedFile) {
        // Mode 1: Multimodal Fusion (Clinical Mandatory + Knee X-Ray Optional)
        // Evaluates both metrics through the trained DeepMultimodalFusionClassifier
        const res = await api.predict(selectedFile, 'multimodal_fusion', payload);
        setDiagnosisResult(res);
        setCamTargetClass(res.predicted_label);

        // If prediction indicates pathological bone loss (Osteopenia or Osteoporosis), generate Grad-CAM
        // If prediction is Normal, DO NOT display the misleading red disease heatmap overlay!
        if (res.predicted_label === 'Normal') {
          setGradcamResult(null);
          setIsGeneratingGradcam(false);
        } else {
          setIsGeneratingGradcam(true);
          try {
            const camRes = await api.generateGradCAM(
              selectedFile,
              'multimodal_fusion',
              res.predicted_label,
              camAlpha,
              camColormap,
              0.0
            );
            setGradcamResult(camRes);
          } catch (camErr) {
            console.warn('Grad-CAM generation failed:', camErr);
          } finally {
            setIsGeneratingGradcam(false);
          }
        }
      } else {
        // Mode 2: Clinical Model Diagnosis (Mandatory Clinical Data Only)
        const res = await api.predictClinical(payload);
        setDiagnosisResult(res);
        setGradcamResult(null);
      }
    } catch (err: any) {
      console.error('Diagnosis failed:', err);
      setDiagnosisError(err.message || 'Diagnosis execution failed. Please verify backend status.');
    } finally {
      setIsDiagnosing(false);
    }
  };

  // Refresh Grad-CAM when target class, colormap, or opacity changes
  const handleUpdateGradCAM = async (newTarget?: string, newColormap?: string, newAlpha?: number) => {
    if (!selectedFile) return;
    setIsGeneratingGradcam(true);
    try {
      const target = newTarget !== undefined ? newTarget : camTargetClass;
      const cmap = newColormap !== undefined ? newColormap : camColormap;
      const alpha = newAlpha !== undefined ? newAlpha : camAlpha;

      const camRes = await api.generateGradCAM(
        selectedFile,
        'multimodal_fusion',
        target,
        alpha,
        cmap,
        0.0
      );
      setGradcamResult(camRes);
    } catch (err: any) {
      console.error('Grad-CAM update failed:', err);
    } finally {
      setIsGeneratingGradcam(false);
    }
  };

  const isMultimodal = Boolean(diagnosisResult && diagnosisResult.input_type.includes('clinical') && selectedFile);

  return (
    <div className="space-y-5 sm:space-y-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 sm:pb-5 flex flex-col md:flex-row md:items-center justify-between gap-3 sm:gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="px-2 py-0.5 rounded-full bg-teal-100 dark:bg-teal-900/60 text-teal-800 dark:text-teal-200 text-[10px] sm:text-[11px] font-semibold tracking-wide uppercase">
              Clinical Mandatory (13 Features)
            </span>
            <span className="px-2 py-0.5 rounded-full bg-sky-100 dark:bg-sky-900/60 text-sky-800 dark:text-sky-200 text-[10px] sm:text-[11px] font-semibold tracking-wide uppercase">
              Knee X-Ray Optional (Multimodal)
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2 mt-1.5 sm:mt-2">
            <ScanEye className="w-5 h-5 sm:w-6 sm:h-6 text-teal-600 shrink-0" />
            <span>Multimodal Diagnostic Console</span>
          </h1>
          <p className="text-[11px] sm:text-xs text-slate-500 dark:text-slate-400 mt-1">
            Clinical systemic variables are mandatory. Attach an optional knee radiograph to run deep multimodal fusion.
          </p>
        </div>

        {/* Quick Test Presets */}
        <div className="flex flex-col items-start md:items-end gap-1.5 text-xs">
          <span className="text-slate-400 font-medium text-[10px] sm:text-[11px]">Quick Clinical Personas:</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <button
              onClick={() => applyPreset('normal')}
              className="px-2 sm:px-2.5 py-1 rounded-lg bg-emerald-50 hover:bg-emerald-100 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 font-medium transition text-[11px] sm:text-xs border border-emerald-200 dark:border-emerald-800 cursor-pointer"
            >
              Normal (42y)
            </button>
            <button
              onClick={() => applyPreset('osteopenia')}
              className="px-2 sm:px-2.5 py-1 rounded-lg bg-amber-50 hover:bg-amber-100 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300 font-medium transition text-[11px] sm:text-xs border border-amber-200 dark:border-amber-800 cursor-pointer"
            >
              Osteopenia (58y)
            </button>
            <button
              onClick={() => applyPreset('osteoporosis')}
              className="px-2 sm:px-2.5 py-1 rounded-lg bg-rose-50 hover:bg-rose-100 dark:bg-rose-950/50 text-rose-800 dark:text-rose-300 font-medium transition text-[11px] sm:text-xs border border-rose-200 dark:border-rose-800 cursor-pointer"
            >
              Osteoporosis (74y)
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: Clinical Form (Mandatory) on Left, Knee X-ray (Optional) on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-8">
        {/* ======================================================== */}
        {/* LEFT COLUMN: Mandatory Clinical Data Form (13 Features) */}
        {/* ======================================================== */}
        <div className="lg:col-span-6 space-y-6">
          <Card className="border-teal-200/80 dark:border-teal-900 shadow-sm">
            <CardHeader
              title="Patient Clinical Profile"
              subtitle="Mandatory 13 features adhering strictly to the training contract"
              action={
                <button
                  onClick={() => setShowContractSpecs(!showContractSpecs)}
                  className="flex items-center gap-1 text-[11px] text-teal-700 dark:text-teal-300 hover:underline"
                >
                  <Info className="w-3.5 h-3.5" />
                  <span>{showContractSpecs ? 'Hide Contract' : 'View Contract'}</span>
                </button>
              }
            />
            <CardContent className="space-y-4 text-xs">
              {/* Feature Contract Accordion */}
              {showContractSpecs && (
                <div className="p-3.5 rounded-xl bg-teal-50/70 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800 text-[11px] space-y-2">
                  <div className="font-semibold text-teal-900 dark:text-teal-200 flex items-center justify-between">
                    <span>13-Feature Training Contract Schema</span>
                    <span className="font-mono text-[10px]">Outputs (1, 13) Tensor</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 font-mono text-[10px] text-teal-800 dark:text-teal-300">
                    {FEATURE_CONTRACT_SPECS.map((spec) => (
                      <div key={spec.key} className="flex justify-between p-1 rounded bg-white/70 dark:bg-slate-900/60 border border-teal-200/50 dark:border-teal-800/50">
                        <span>{spec.index}. {spec.key}</span>
                        <span className="text-slate-500">{spec.unit}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 1. Age Input */}
              <NumericInputField
                id="input-age"
                label="1. Patient Age"
                unit="years"
                value={numAge}
                onChange={setNumAge}
                min={18}
                max={120}
                required
                helperText="Patient age in completed years (must be 18–120)."
              />

              {/* 2 & 3. Height (m) and Weight (kg) */}
              <div className="space-y-4">
                <HeightConverter
                  valueMetres={numHeightM}
                  onChangeMetres={setNumHeightM}
                  required
                />

                <NumericInputField
                  id="input-weight"
                  label="3. Weight"
                  unit="kg"
                  value={numWeightKg}
                  onChange={setNumWeightKg}
                  min={20}
                  max={250}
                  step="0.1"
                  required
                  helperText="Body weight in kilograms (20–250 kg)."
                />
              </div>

              {/* 4. Calculated BMI */}
              <BmiCalculator
                heightM={numHeightM}
                weightKg={numWeightKg}
                onBmiChange={(bmi, overridden) => {
                  setActiveBmi(bmi);
                  setIsBmiOverridden(overridden);
                }}
              />

              {/* 7. Gender Female Indicator */}
              <div className="space-y-1.5">
                <span className="block font-medium text-slate-700 dark:text-slate-300">
                  7. Biological Sex (gender_female)
                </span>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => handleGenderChange(true)}
                    className={`py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-2 transition ${
                      isFemale
                        ? 'bg-teal-600 text-white border-teal-600'
                        : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700'
                    }`}
                  >
                    {isFemale && <Check className="w-3.5 h-3.5" />}
                    <span>Female (1.0)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleGenderChange(false)}
                    className={`py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-2 transition ${
                      !isFemale
                        ? 'bg-teal-600 text-white border-teal-600'
                        : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700'
                    }`}
                  >
                    {!isFemale && <Check className="w-3.5 h-3.5" />}
                    <span>Male (0.0)</span>
                  </button>
                </div>
              </div>

              {/* 5. Menopause Age (Females Only) */}
              <MenopauseSelector
                isFemale={isFemale}
                status={menopausalStatus}
                onsetAge={numMenopauseAge}
                onStatusChange={setMenopausalStatus}
                onOnsetAgeChange={setNumMenopauseAge}
              />

              {/* 6. Max Walking Distance (km) */}
              <WalkingDistanceConverter
                valueKm={numMaxWalkingKm}
                onChangeKm={setNumMaxWalkingKm}
                required
              />

              {/* 8-13. Six Binary Systemic Risk Factors */}
              <div className="space-y-2 pt-1 border-t border-slate-200 dark:border-slate-800">
                <span className="block font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Binary Systemic Indicators (1.0 = Positive, 0.0 = Negative)
                </span>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={jointPain}
                      onChange={(e) => setJointPain(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>8. Joint Pain</span>
                  </label>

                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={smoker}
                      onChange={(e) => setSmoker(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>9. Smoker History</span>
                  </label>

                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={diabetic}
                      onChange={(e) => setDiabetic(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>10. Diabetic</span>
                  </label>

                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={hypothyroidism}
                      onChange={(e) => setHypothyroidism(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>11. Hypothyroidism</span>
                  </label>

                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={fracture}
                      onChange={(e) => setFracture(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>12. Prior Fracture</span>
                  </label>

                  <label className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800">
                    <input
                      type="checkbox"
                      checked={familyHistory}
                      onChange={(e) => setFamilyHistory(e.target.checked)}
                      className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                    />
                    <span>13. Family History</span>
                  </label>
                </div>
              </div>

              {/* Primary Diagnostic Action Button */}
              <div className="pt-2">
                <Button
                  onClick={handleRunDiagnosis}
                  isLoading={isDiagnosing}
                  className={`w-full text-white font-semibold py-3 rounded-xl shadow transition ${
                    selectedFile
                      ? 'bg-gradient-to-r from-teal-600 to-sky-600 hover:from-teal-700 hover:to-sky-700'
                      : 'bg-teal-600 hover:bg-teal-700'
                  }`}
                >
                  {selectedFile ? (
                    <>
                      <Sparkles className="w-4 h-4 mr-2" />
                      Run Multimodal Fusion Diagnosis (Clinical + X-Ray)
                    </>
                  ) : (
                    <>
                      <Stethoscope className="w-4 h-4 mr-2" />
                      Run Clinical Diagnosis (13 Mandatory Features)
                    </>
                  )}
                </Button>
                <p className="text-[10px] text-slate-400 text-center mt-1.5">
                  {selectedFile
                    ? 'Multimodal Fusion: Combines systemic clinical indicators and knee radiograph microarchitecture.'
                    : 'Clinical Mode: Knee X-ray is optional. Attach an X-ray on the right to run Multimodal Fusion.'}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ======================================================== */}
        {/* RIGHT COLUMN: Optional Knee X-ray & Radiological Guide */}
        {/* ======================================================== */}
        <div className="lg:col-span-6 space-y-6">
          <Card className="border-sky-200/80 dark:border-sky-900 shadow-sm">
            <CardHeader
              title="Knee Radiograph (Optional)"
              subtitle="Attach an AP radiograph to activate Deep Multimodal Fusion and Grad-CAM"
              action={
                <Badge variant={selectedFile ? 'teal' : 'slate'} size="sm">
                  {selectedFile ? 'X-Ray Attached' : 'Optional Modality'}
                </Badge>
              }
            />
            <CardContent className="space-y-4 text-xs">
              {/* Multimodal Guidance Banner */}
              <div className="p-3 rounded-lg bg-sky-50 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800 text-sky-900 dark:text-sky-200 text-[11px] flex items-start gap-2">
                <Info className="w-4 h-4 shrink-0 mt-0.5 text-sky-600" />
                <div>
                  <strong>Multimodal Integration:</strong> When an X-ray is attached, both metrics (clinical systemic risk factors and visual trabecular microarchitecture) are jointly evaluated by the trained multimodal neural network.
                </div>
              </div>

              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                onChange={handleFileChange}
                className="hidden"
              />

              {/* Upload Dropzone */}
              {!previewUrl ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-sky-500 dark:hover:border-sky-500 rounded-2xl p-6 text-center cursor-pointer bg-slate-50/50 dark:bg-slate-900/50 hover:bg-sky-50/30 dark:hover:bg-sky-950/20 transition-all flex flex-col items-center justify-center min-h-[170px]"
                >
                  <div className="w-12 h-12 rounded-full bg-sky-100 dark:bg-sky-950/60 text-sky-600 flex items-center justify-center mb-2.5">
                    <Upload className="w-6 h-6" />
                  </div>
                  <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                    Upload Knee Radiograph (AP View)
                  </h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    Accepts JPG, JPEG, or PNG radiographs. Aspect ratio preserved via letterbox padding.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center max-h-[240px]">
                    <img
                      src={previewUrl}
                      alt="Uploaded Radiograph"
                      className="max-h-[240px] object-contain"
                    />
                    <button
                      onClick={clearImage}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-black/70 hover:bg-black text-white transition cursor-pointer"
                      title="Remove image"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-[10px] text-white font-mono">
                      {selectedFile?.name} ({(selectedFile?.size ? selectedFile.size / 1024 : 0).toFixed(0)} KB)
                    </div>
                  </div>
                </div>
              )}

              {/* Research Test Samples Picker */}
              <div>
                <span className="block text-[11px] font-medium text-slate-500 dark:text-slate-400 mb-1.5">
                  Or select a test radiograph from the authentic cohort:
                </span>
                <div className="flex flex-wrap gap-2">
                  {samples.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleSelectSample(s)}
                      className="px-2.5 py-1 text-xs rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-sky-500 dark:hover:border-sky-500 text-slate-700 dark:text-slate-300 flex items-center gap-1.5 transition cursor-pointer"
                    >
                      <FileImage className="w-3.5 h-3.5 text-sky-600" />
                      <span>{s.filename}</span>
                      <span className="text-[10px] px-1 rounded bg-slate-100 dark:bg-slate-800 font-medium">
                        {s.true_label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Collapsible Visual Radiographic Guide */}
          <Card className="border-slate-200 dark:border-slate-800">
            <div
              onClick={() => setShowVisualGuide(!showVisualGuide)}
              className="p-4 flex items-center justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-850/50 transition rounded-xl select-none"
            >
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-teal-50 dark:bg-teal-950/60 text-teal-600 flex items-center justify-center">
                  <BookOpen className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                    Visual Radiographic Guide: Normal vs. Osteoporotic Bone
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    How clinicians differentiate healthy bone from osteoporosis visually
                  </p>
                </div>
              </div>
              <button
                type="button"
                className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                {showVisualGuide ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>

            {showVisualGuide && (
              <CardContent className="pt-0 space-y-4 text-xs border-t border-slate-100 dark:border-slate-800 mt-2">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-3">
                  {/* Normal Features */}
                  <div className="p-3.5 rounded-xl bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800 space-y-2">
                    <div className="flex items-center gap-2 font-bold text-emerald-800 dark:text-emerald-300 text-xs">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                      <span>Normal Healthy Bone</span>
                    </div>
                    <ul className="text-[11px] text-slate-700 dark:text-slate-300 space-y-1.5 leading-relaxed">
                      <li>
                        <strong>Cortical Thickness:</strong> Dense, solid white, continuous cortical margins along distal femur and proximal tibia (&ge;45% shaft diameter).
                      </li>
                      <li>
                        <strong>Trabecular Lattice:</strong> Rich criss-cross network of primary and secondary cancellous bone struts in femoral condyles.
                      </li>
                      <li>
                        <strong>Uniform Radiopacity:</strong> Homogeneous light gray-white density without dark washed-out hollow areas.
                      </li>
                      <li>
                        <strong>Smooth Subchondral Plate:</strong> Intact articular surfaces with well-preserved joint space.
                      </li>
                    </ul>
                  </div>

                  {/* Osteoporotic Features */}
                  <div className="p-3.5 rounded-xl bg-rose-50/60 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-800 space-y-2">
                    <div className="flex items-center gap-2 font-bold text-rose-800 dark:text-rose-300 text-xs">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                      <span>Osteoporotic Bone Signs</span>
                    </div>
                    <ul className="text-[11px] text-slate-700 dark:text-slate-300 space-y-1.5 leading-relaxed">
                      <li>
                        <strong>&ldquo;Pencil-Thin&rdquo; Cortex:</strong> Severe thinning and scalloping of outer cortical bone due to endosteal resorption (&lt;30% shaft diameter).
                      </li>
                      <li>
                        <strong>&ldquo;Picture-Frame&rdquo; Sign:</strong> Marked medullary radiolucency (dark, washed-out marrow cavity) bordered by a paper-thin cortex.
                      </li>
                      <li>
                        <strong>Trabecular Rarefaction:</strong> Preferential loss of transverse/horizontal trabeculae; remaining vertical trabeculae appear faint and sparse.
                      </li>
                      <li>
                        <strong>Absence of Sclerosis:</strong> Demineralization occurs without the dense ivory-white subchondral sclerosis characteristic of osteoarthritis.
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400">
                  <strong className="text-slate-800 dark:text-slate-200">Clinical Distinction from Osteoarthritis (OA):</strong>{' '}
                  OA involves localized joint space narrowing, subchondral bone thickening (sclerosis), and osteophyte spurs. In contrast, pure osteoporosis produces generalized cortical thinning and diffuse bone lucency without subchondral whitening.
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>

      {/* Error Alert */}
      {diagnosisError && (
        <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="text-xs text-rose-800 dark:text-rose-200 space-y-1">
            <strong>Diagnosis Error:</strong>
            <p>{diagnosisError}</p>
          </div>
        </div>
      )}

      {/* ======================================================== */}
      {/* SHOWCASE SECTION: Diagnostic Output Considering Both Metrics */}
      {/* ======================================================== */}
      {diagnosisResult && (
        <Card className={`border-2 shadow-lg transition-all ${isMultimodal ? 'border-sky-400 dark:border-sky-600' : 'border-teal-400 dark:border-teal-600'}`}>
          <CardHeader
            title={isMultimodal ? 'Multimodal Fusion Diagnosis' : 'Clinical Model Prediction'}
            subtitle={
              isMultimodal
                ? `Multimodal Architecture: ${diagnosisResult.model_name} (Considers Both Clinical & X-Ray Metrics)`
                : `Clinical Classifier: ${diagnosisResult.model_name} (Evaluated on Mandatory 13 Features)`
            }
            action={
              <div className="flex items-center gap-2">
                <Badge variant={isMultimodal ? 'blue' : 'teal'} size="sm">
                  {isMultimodal ? 'Multimodal Fusion' : 'Clinical Only'}
                </Badge>
                <Badge variant={CLASS_CONFIG[diagnosisResult.predicted_label as keyof typeof CLASS_CONFIG]?.variant || 'slate'} size="md">
                  {diagnosisResult.predicted_label}
                </Badge>
              </div>
            }
          />
          <CardContent className="space-y-6 text-xs">
            {/* Primary Category Banner */}
            <div
              className={`p-5 rounded-2xl border ${
                CLASS_CONFIG[diagnosisResult.predicted_label as keyof typeof CLASS_CONFIG]?.lightBg
              } ${CLASS_CONFIG[diagnosisResult.predicted_label as keyof typeof CLASS_CONFIG]?.border}`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                <div>
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
                    Diagnostic Outcome:
                  </span>
                  <span className="text-2xl font-extrabold text-slate-900 dark:text-slate-100">
                    {diagnosisResult.predicted_label}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-semibold text-slate-500 block">
                    Calibrated Confidence:
                  </span>
                  <span className="text-2xl font-bold font-mono text-slate-900 dark:text-slate-100">
                    {(diagnosisResult.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                {CLASS_CONFIG[diagnosisResult.predicted_label as keyof typeof CLASS_CONFIG]?.desc}
              </p>
              <div className="mt-3 pt-2.5 border-t border-slate-200/60 dark:border-slate-800/60 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-2">
                <span>Class Mapping Index: {CLASS_CONFIG[diagnosisResult.predicted_label as keyof typeof CLASS_CONFIG]?.classIndex}</span>
                <span>Inference Duration: {diagnosisResult.inference_duration_ms} ms</span>
                <span>Input Mode: {isMultimodal ? 'Dual-Modal (Clinical + Radiograph)' : 'Single-Modal (Clinical Mandatory)'}</span>
              </div>
            </div>

            {/* 3-Class Probability Distribution */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">
                Three-Class Calibrated Probabilities:
              </span>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cName) => {
                  const pVal = (diagnosisResult.calibrated_probabilities || diagnosisResult.probabilities)[cName] ?? 0;
                  const isWinner = cName === diagnosisResult.predicted_label;
                  return (
                    <div
                      key={cName}
                      className={`p-3 rounded-xl border ${
                        isWinner
                          ? 'border-slate-900 dark:border-slate-100 bg-white dark:bg-slate-900 shadow-xs'
                          : 'border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50'
                      }`}
                    >
                      <div className="flex justify-between items-center text-xs mb-1.5">
                        <span className={`font-semibold ${isWinner ? 'text-slate-900 dark:text-slate-100' : 'text-slate-600 dark:text-slate-400'}`}>
                          {cName}
                        </span>
                        <span className="font-mono font-bold text-sm">{(pVal * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${CLASS_CONFIG[cName].bg}`}
                          style={{ width: `${Math.max(3, pVal * 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* ======================================================== */}
            {/* DUAL METRICS SHOWCASE (Both Metrics Considered) */}
            {/* ======================================================== */}
            <div className="space-y-3 pt-3 border-t border-slate-200 dark:border-slate-800">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-teal-600" />
                  <span>Metrics Considered in Diagnosis</span>
                </span>
                <span className="text-[11px] text-slate-400">
                  {isMultimodal ? 'Both Non-Imaging & Imaging Metrics Evaluated' : 'Clinical Non-Imaging Metric Evaluated'}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Metric 1: Systemic Clinical Risk Indicators */}
                <div className="p-4 rounded-xl bg-teal-50/50 dark:bg-teal-950/20 border border-teal-200 dark:border-teal-800 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-teal-900 dark:text-teal-200 text-xs flex items-center gap-1.5">
                      <Stethoscope className="w-4 h-4 text-teal-600" />
                      <span>Metric 1: Mandatory Clinical Indicators</span>
                    </span>
                    <Badge variant="teal" size="sm">
                      13 Features Verified
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-700 dark:text-slate-300">
                    <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-100 dark:border-teal-900">
                      <span className="text-slate-400 block text-[10px]">Age &amp; Sex:</span>
                      <span className="font-semibold">{numAge ?? '--'}y &bull; {isFemale ? 'Female' : 'Male'}</span>
                    </div>
                    <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-100 dark:border-teal-900">
                      <span className="text-slate-400 block text-[10px]">BMI (kg/m²):</span>
                      <span className="font-semibold font-mono">{activeBmi ? `${activeBmi} kg/m²` : '--'}</span>
                    </div>
                    <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-100 dark:border-teal-900">
                      <span className="text-slate-400 block text-[10px]">Menopause Status:</span>
                      <span className="font-semibold">
                        {isFemale
                          ? menopausalStatus === 'post'
                            ? numMenopauseAge
                              ? `Onset: ${numMenopauseAge}y`
                              : 'Post (Median 47y)'
                            : 'Pre-Menopausal'
                          : 'N/A (Male)'}
                      </span>
                    </div>
                    <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-100 dark:border-teal-900">
                      <span className="text-slate-400 block text-[10px]">Max Walking:</span>
                      <span className="font-semibold font-mono">{numMaxWalkingKm !== null ? `${numMaxWalkingKm} km` : '--'}</span>
                    </div>
                  </div>
                  <div className="text-[10px] text-slate-500 flex flex-wrap gap-1.5 pt-1">
                    <span className="font-semibold">Risk Flags:</span>
                    {fracture && <span className="px-1.5 py-0.5 rounded bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-300">Prior Fracture</span>}
                    {familyHistory && <span className="px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300">Family History</span>}
                    {jointPain && <span className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">Joint Pain</span>}
                    {diabetic && <span className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">Diabetic</span>}
                    {hypothyroidism && <span className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">Hypothyroidism</span>}
                    {smoker && <span className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">Smoker</span>}
                    {!fracture && !familyHistory && !jointPain && !diabetic && !hypothyroidism && !smoker && (
                      <span className="text-emerald-600">No active systemic flags</span>
                    )}
                  </div>

                  {/* Clinical probabilities breakdown */}
                  {diagnosisResult.clinical_probabilities && (
                    <div className="pt-2 border-t border-teal-200/60 dark:border-teal-800/60 space-y-1.5">
                      <span className="text-[10px] font-semibold text-teal-800 dark:text-teal-300 block">
                        Clinical Risk Probabilities (Non-Imaging Metric):
                      </span>
                      <div className="grid grid-cols-3 gap-1.5 text-[10px]">
                        {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cat) => {
                          const prob = diagnosisResult.clinical_probabilities?.[cat] ?? 0;
                          return (
                            <div key={cat} className="p-1 rounded bg-white/70 dark:bg-slate-900/70 border border-teal-100 dark:border-teal-900 text-center">
                              <div className="text-slate-500 text-[9px]">{cat}</div>
                              <div className="font-mono font-bold text-teal-950 dark:text-teal-100">{(prob * 100).toFixed(1)}%</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Metric 2: Knee Radiograph Bone Microarchitecture */}
                <div className="p-4 rounded-xl bg-sky-50/50 dark:bg-sky-950/20 border border-sky-200 dark:border-sky-800 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-sky-900 dark:text-sky-200 text-xs flex items-center gap-1.5">
                      <ScanEye className="w-4 h-4 text-sky-600" />
                      <span>Metric 2: Radiographic Microarchitecture</span>
                    </span>
                    <Badge variant={isMultimodal ? 'blue' : 'slate'} size="sm">
                      {isMultimodal ? 'X-Ray Analyzed' : 'Not Attached'}
                    </Badge>
                  </div>
                  {isMultimodal && diagnosisResult.image_meta ? (
                    <div className="space-y-2 text-[11px] text-slate-700 dark:text-slate-300">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-sky-100 dark:border-sky-900">
                          <span className="text-slate-400 block text-[10px]">Image Dimensions:</span>
                          <span className="font-semibold font-mono">{diagnosisResult.image_meta.width} &times; {diagnosisResult.image_meta.height} px</span>
                        </div>
                        <div className="p-1.5 rounded bg-white/80 dark:bg-slate-900/80 border border-sky-100 dark:border-sky-900">
                          <span className="text-slate-400 block text-[10px]">File &amp; Format:</span>
                          <span className="font-semibold truncate block">{diagnosisResult.image_meta.filename} ({diagnosisResult.image_meta.format})</span>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-500 leading-relaxed">
                        Evaluated via DenseNet-121 convolutional encoder extracting distal femoral condyle cortical margins and proximal tibial cancellous trabecular patterns.
                      </p>
                      {/* Vision probabilities breakdown */}
                      {diagnosisResult.vision_probabilities && (
                        <div className="pt-2 border-t border-sky-200/60 dark:border-sky-800/60 space-y-1.5">
                          <span className="text-[10px] font-semibold text-sky-800 dark:text-sky-300 block">
                            Radiographic Visual Probabilities (Imaging Metric):
                          </span>
                          <div className="grid grid-cols-3 gap-1.5 text-[10px]">
                            {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cat) => {
                              const prob = diagnosisResult.vision_probabilities?.[cat] ?? 0;
                              return (
                                <div key={cat} className="p-1 rounded bg-white/70 dark:bg-slate-900/70 border border-sky-100 dark:border-sky-900 text-center">
                                  <div className="text-slate-500 text-[9px]">{cat}</div>
                                  <div className="font-mono font-bold text-sky-950 dark:text-sky-100">{(prob * 100).toFixed(1)}%</div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-3 text-center space-y-1 text-slate-500">
                      <p className="text-[11px]">Knee radiograph was not attached for this diagnosis.</p>
                      <p className="text-[10px] text-slate-400">
                        Upload or select a test radiograph above to evaluate visual bone microarchitecture in multimodal fusion.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Multimodal balance indicator */}
              {isMultimodal && (
                <div className="p-2.5 rounded-lg bg-indigo-50/70 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800 flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 text-[11px] text-indigo-900 dark:text-indigo-200">
                  <span className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                    <strong>Multimodal Integration Balance:</strong> 60% Radiographic Visual Microarchitecture + 40% Systemic Clinical Risk Indicators
                  </span>
                  <span className="text-[10px] text-indigo-700 dark:text-indigo-300 font-medium">
                    Modifying clinical inputs dynamically modulates the fused outcome
                  </span>
                </div>
              )}
            </div>

            {/* ======================================================== */}
            {/* GRAD-CAM VISUAL ATTRIBUTION (Multimodal Mode) */}
            {/* ======================================================== */}
            {isMultimodal && (
              <div className="space-y-3 pt-3 border-t border-slate-200 dark:border-slate-800">
                {isGeneratingGradcam ? (
                  <div className="h-56 flex flex-col items-center justify-center text-xs text-slate-400 bg-slate-50 dark:bg-slate-900 rounded-xl border border-dashed border-slate-200 dark:border-slate-800">
                    <Flame className="w-8 h-8 text-amber-500 animate-pulse mb-2" />
                    <span>Computing multimodal fusion layer gradients...</span>
                  </div>
                ) : diagnosisResult.predicted_label === 'Normal' && !gradcamResult ? (
                  /* Reassurance Banner for Normal: Do NOT show misleading red disease heatmap! */
                  <div className="p-4 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 space-y-2.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                        <span className="font-semibold text-emerald-900 dark:text-emerald-200 text-xs">
                          Normal Healthy Bone Mineral Density &bull; Heatmap Not Indicated
                        </span>
                      </div>
                      <Badge variant="normal" size="sm">
                        Healthy Bone
                      </Badge>
                    </div>
                    <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
                      No focal osteoporotic lesions, cortical thinning, or trabecular demineralization were identified.
                      Pathological heatmaps are suppressed for normal findings to prevent visual misinterpretation of healthy knee anatomy.
                    </p>
                    <div className="pt-2 border-t border-emerald-200/60 dark:border-emerald-800/60 flex items-center justify-between">
                      <span className="text-[10px] text-slate-500">
                        Cortical margins and trabecular density are preserved within normal reference bounds.
                      </span>
                      <button
                        onClick={() => handleUpdateGradCAM('Normal', camColormap, camAlpha)}
                        className="text-[10px] font-medium text-emerald-700 dark:text-emerald-300 hover:underline flex items-center gap-1 cursor-pointer"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Inspect Normal Feature Salience Map</span>
                      </button>
                    </div>
                  </div>
                ) : gradcamResult ? (
                  /* Genuine Grad-CAM Heatmap when pathology is present or explicitly requested */
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-800 dark:text-slate-200 text-xs flex items-center gap-1.5">
                        <Flame className={`w-4 h-4 ${gradcamResult.target_class === 'Normal' ? 'text-emerald-500' : 'text-amber-500'}`} />
                        <span>
                          {gradcamResult.target_class === 'Normal'
                            ? 'Normal Architecture Feature Salience (Reference)'
                            : 'Genuine Grad-CAM Pathology Heatmap'}
                        </span>
                      </span>
                      <div className="flex items-center gap-2">
                        {gradcamResult.target_class === 'Normal' && (
                          <button
                            onClick={() => setGradcamResult(null)}
                            className="text-[10px] text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 flex items-center gap-1"
                          >
                            <EyeOff className="w-3 h-3" />
                            <span>Hide Normal Overlay</span>
                          </button>
                        )}
                        <Badge
                          variant={gradcamResult.target_class === 'Normal' ? 'normal' : 'teal'}
                          size="sm"
                        >
                          Target: {gradcamResult.target_class}
                        </Badge>
                      </div>
                    </div>

                    {/* Interactive Controls */}
                    <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 grid grid-cols-3 gap-2.5 text-xs">
                      <div>
                        <label className="block text-[10px] text-slate-400 font-medium mb-1">
                          Attribution Class:
                        </label>
                        <select
                          value={camTargetClass}
                          onChange={(e) => {
                            setCamTargetClass(e.target.value);
                            handleUpdateGradCAM(e.target.value, camColormap, camAlpha);
                          }}
                          className="w-full px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200"
                        >
                          <option value="Normal">Normal</option>
                          <option value="Osteopenia">Osteopenia</option>
                          <option value="Osteoporosis">Osteoporosis</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] text-slate-400 font-medium mb-1">
                          Thermal Colormap:
                        </label>
                        <select
                          value={camColormap}
                          onChange={(e) => {
                            setCamColormap(e.target.value);
                            handleUpdateGradCAM(camTargetClass, e.target.value, camAlpha);
                          }}
                          className="w-full px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200"
                        >
                          <option value="jet">JET</option>
                          <option value="turbo">TURBO</option>
                          <option value="viridis">VIRIDIS</option>
                          <option value="inferno">INFERNO</option>
                          <option value="bone">BONE</option>
                        </select>
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] text-slate-400 font-medium mb-1">
                          <span>Opacity (&alpha;):</span>
                          <span className="font-mono text-teal-600 font-bold">{Math.round(camAlpha * 100)}%</span>
                        </div>
                        <input
                          type="range"
                          min={0.15}
                          max={0.85}
                          step={0.05}
                          value={camAlpha}
                          onChange={(e) => {
                            const newA = Number(e.target.value);
                            setCamAlpha(newA);
                            handleUpdateGradCAM(camTargetClass, camColormap, newA);
                          }}
                          className="w-full accent-teal-600 mt-1"
                        />
                      </div>
                    </div>

                    {/* Side-by-side Images: Original vs Grad-CAM Overlay */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <span className="text-[10px] font-semibold text-slate-500 block mb-1">
                          Original Patient Radiograph
                        </span>
                        <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center h-[200px]">
                          <img
                            src={gradcamResult.original_image_base64}
                            alt="Original Radiograph"
                            className="max-h-[200px] object-contain"
                          />
                        </div>
                      </div>

                      <div>
                        <span className={`text-[10px] font-semibold block mb-1 ${gradcamResult.target_class === 'Normal' ? 'text-emerald-600' : 'text-teal-600'}`}>
                          {gradcamResult.target_class === 'Normal' ? 'Normal Feature Salience Overlay' : 'Grad-CAM Heatmap Overlay'}
                        </span>
                        <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black flex items-center justify-center h-[200px]">
                          <img
                            src={gradcamResult.overlay_base64}
                            alt="Grad-CAM Overlay"
                            className="max-h-[200px] object-contain"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Gradient Legend Bar */}
                    <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1">
                      <div className="flex justify-between text-[9px] text-slate-400">
                        <span>Low Gradient (0.0)</span>
                        <span>
                          {gradcamResult.target_class === 'Normal'
                            ? 'High Normal Salience (1.0)'
                            : 'High Demineralization Attribution (1.0)'}
                        </span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-amber-300 to-red-600 shadow-inner" />
                      <p className="text-[9px] text-slate-400 leading-tight">
                        {gradcamResult.target_class === 'Normal'
                          ? 'Highlights the intact cortical margins and trabecular density patterns that the model used to confirm healthy bone mineral architecture.'
                          : `Warm regions indicate spatial receptive fields in the distal femoral condyles and proximal tibial plateau that drove the neural activation for ${gradcamResult.target_class}.`}
                      </p>
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            {/* Review Rationale */}
            {diagnosisResult.review_rationale && (
              <div className="p-3.5 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 text-amber-900 dark:text-amber-200 text-xs flex items-start gap-2.5">
                <Info className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                <div>
                  <strong>Clinical Assessment Rationale:</strong> {diagnosisResult.review_rationale}
                </div>
              </div>
            )}

            {/* Safety & Medical Research Disclaimer */}
            <div className="p-3.5 rounded-xl bg-slate-100 dark:bg-slate-850 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 text-[11px] leading-relaxed">
              <strong>Research Safety &amp; Non-Diagnostic Notice:</strong> {diagnosisResult.disclaimer}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default XRayAnalysisPage;
