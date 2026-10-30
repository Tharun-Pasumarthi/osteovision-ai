import React, { useState, useEffect } from 'react';
import {
  Stethoscope,
  BarChart2,
  TrendingDown,
  Info,
  CheckCircle2,
  Sliders,
  Scale,
  Calendar,
  Activity,
  Sparkles,
  AlertCircle,
  HelpCircle,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { api } from '../services/api';
import {
  PredictionResponse,
  ClinicalFeatures,
  ShapExplanationResponse,
  GlobalFeatureImportanceItem,
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
    desc: 'Clinical risk indicators are consistent with preserved bone mineral density.',
  },
  Osteopenia: {
    variant: 'osteopenia' as const,
    bg: 'bg-amber-500',
    lightBg: 'bg-amber-50 dark:bg-amber-950/40',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-200 dark:border-amber-800',
    desc: 'Elevated systemic risk indicators suggest mild to moderate bone mineral loss.',
  },
  Osteoporosis: {
    variant: 'osteoporosis' as const,
    bg: 'bg-rose-500',
    lightBg: 'bg-rose-50 dark:bg-rose-950/40',
    text: 'text-rose-700 dark:text-rose-300',
    border: 'border-rose-200 dark:border-rose-800',
    desc: 'High systemic risk profile indicating pronounced fracture vulnerability. DXA scan recommended.',
  },
};

export const ClinicalAnalysisPage: React.FC = () => {
  // 13 Clinical features state
  const [numAge, setNumAge] = useState<number | null>(62);
  const [numHeightM, setNumHeightM] = useState<number | null>(1.58);
  const [numWeightKg, setNumWeightKg] = useState<number | null>(64.0);
  const [isFemale, setIsFemale] = useState<boolean>(true);
  const [menopausalStatus, setMenopausalStatus] = useState<MenopausalStatus>('post');
  const [numMenopauseAge, setNumMenopauseAge] = useState<number | null>(46);
  const [numMaxWalkingKm, setNumMaxWalkingKm] = useState<number | null>(2.5);
  const [activeBmi, setActiveBmi] = useState<number | null>(25.64);
  const [isBmiOverridden, setIsBmiOverridden] = useState<boolean>(false);

  // 6 binary clinical indicators
  const [jointPain, setJointPain] = useState<boolean>(true);
  const [smoker, setSmoker] = useState<boolean>(false);
  const [diabetic, setDiabetic] = useState<boolean>(false);
  const [hypothyroidism, setHypothyroidism] = useState<boolean>(false);
  const [fracture, setFracture] = useState<boolean>(false);
  const [familyHistory, setFamilyHistory] = useState<boolean>(true);

  // Prediction & SHAP explainability state
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [shapExplanation, setShapExplanation] = useState<ShapExplanationResponse | null>(null);
  const [globalImportance, setGlobalImportance] = useState<GlobalFeatureImportanceItem[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedShapClass, setSelectedShapClass] = useState<string>('Osteoporosis');

  // Load global feature importance on mount
  useEffect(() => {
    let isMounted = true;
    api.getShapGlobal()
      .then((items) => {
        if (isMounted) setGlobalImportance(items);
      })
      .catch((err) => console.error('Failed to load global SHAP importance:', err));
    return () => {
      isMounted = false;
    };
  }, []);

  const buildPayload = (): ClinicalFeatures => ({
    num_age: numAge !== null ? numAge : undefined,
    num_height_m: numHeightM !== null ? numHeightM : undefined,
    num_weight_kg: numWeightKg !== null ? numWeightKg : undefined,
    num_bmi: activeBmi !== null ? activeBmi : undefined,
    num_menopause_age: isFemale && menopausalStatus === 'post' ? numMenopauseAge : null,
    num_max_walking_km: numMaxWalkingKm !== null ? numMaxWalkingKm : undefined,
    gender_female: isFemale ? 1.0 : 0.0,
    joint_pain: jointPain ? 1.0 : 0.0,
    smoker: smoker ? 1.0 : 0.0,
    diabetic: diabetic ? 1.0 : 0.0,
    hypothyroidism: hypothyroidism ? 1.0 : 0.0,
    fracture: fracture ? 1.0 : 0.0,
    family_history: familyHistory ? 1.0 : 0.0,
  });

  const handleRunClinicalPrediction = async (targetCls?: string) => {
    setIsAnalyzing(true);
    setErrorMsg(null);
    const payload = buildPayload();
    try {
      // 1. Run independent clinical machine learning prediction
      const pred = await api.predictClinical(payload);
      setPrediction(pred);

      // 2. Run SHAP local attribution explanation
      const clsToExplain = targetCls || selectedShapClass || pred.predicted_label;
      const shap = await api.explainClinicalShap(payload, clsToExplain);
      setShapExplanation(shap);
      setSelectedShapClass(clsToExplain);
    } catch (err: any) {
      setErrorMsg(err.message || 'Clinical risk analysis failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            Clinical Risk Indicator Analysis &amp; SHAP Explainability
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Independent clinical prediction model with SHAP local waterfall attribution and global Gini feature importance.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="teal" size="sm">
            13 Clinical Features
          </Badge>
          <Badge variant="blue" size="sm">
            Platt Calibrated Random Forest
          </Badge>
        </div>
      </div>

      {/* Main Layout: Form (Left) vs Results & SHAP (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (5 cols): 13 Clinical Features Form */}
        <div className="lg:col-span-5 space-y-5">
          <Card>
            <CardHeader
              title="Patient Clinical Profile"
              subtitle="Enter verified patient demographic, anthropometric, and systemic risk data"
            />
            <CardContent className="space-y-4">
              {/* Biological Sex Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                  Biological Gender
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setIsFemale(true);
                      setMenopausalStatus('post');
                      if (numMenopauseAge === null) setNumMenopauseAge(46);
                    }}
                    className={`py-2 px-3 rounded-lg border text-xs font-semibold transition ${
                      isFemale
                        ? 'border-teal-500 bg-teal-50/50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 shadow-xs'
                        : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400'
                    }`}
                  >
                    Female (Post-Menopausal Profile)
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsFemale(false);
                      setMenopausalStatus('pre_peri');
                      setNumMenopauseAge(null);
                    }}
                    className={`py-2 px-3 rounded-lg border text-xs font-semibold transition ${
                      !isFemale
                        ? 'border-teal-500 bg-teal-50/50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 shadow-xs'
                        : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400'
                    }`}
                  >
                    Male
                  </button>
                </div>
              </div>

              {/* Age Field */}
              <NumericInputField
                id="clinical-age"
                label="Patient Age"
                unit="years"
                min={18}
                max={110}
                step={1}
                value={numAge}
                onChange={setNumAge}
                placeholder="e.g., 62"
                helperText="Age is the primary demographic determinant of systemic cortical bone density decline."
              />

              {/* Height Unit Converter */}
              <HeightConverter
                valueMetres={numHeightM}
                onChangeMetres={setNumHeightM}
              />

              {/* Weight Field */}
              <NumericInputField
                id="clinical-weight"
                label="Patient Weight"
                unit="kg"
                min={25}
                max={220}
                step={0.5}
                value={numWeightKg}
                onChange={setNumWeightKg}
                placeholder="e.g., 64.0"
                helperText="Body weight in kilograms. Low body mass index correlates with increased fracture risk."
              />

              {/* Real-time Reactive BMI Calculator */}
              <BmiCalculator
                heightM={numHeightM}
                weightKg={numWeightKg}
                onBmiChange={(bmi, overridden) => {
                  setActiveBmi(bmi);
                  setIsBmiOverridden(overridden);
                }}
              />

              {/* Menopausal Status and Onset Age */}
              <MenopauseSelector
                isFemale={isFemale}
                status={menopausalStatus}
                onsetAge={numMenopauseAge}
                onStatusChange={setMenopausalStatus}
                onOnsetAgeChange={setNumMenopauseAge}
              />

              {/* Walking Distance Multi-Unit Converter */}
              <WalkingDistanceConverter
                valueKm={numMaxWalkingKm}
                onChangeKm={setNumMaxWalkingKm}
              />

              {/* 6 Binary Risk Factors */}
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800/80 space-y-2.5">
                <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Systemic Comorbidities &amp; Clinical History
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { label: 'Prior Fragility Fracture', val: fracture, set: setFracture },
                    { label: 'Family History of Osteo', val: familyHistory, set: setFamilyHistory },
                    { label: 'Knee Joint Pain', val: jointPain, set: setJointPain },
                    { label: 'Smoker History', val: smoker, set: setSmoker },
                    { label: 'Type II Diabetes', val: diabetic, set: setDiabetic },
                    { label: 'Hypothyroidism', val: hypothyroidism, set: setHypothyroidism },
                  ].map((item, idx) => (
                    <label
                      key={idx}
                      className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer select-none transition ${
                        item.val
                          ? 'border-teal-500/80 bg-teal-50/50 dark:bg-teal-950/30 text-teal-800 dark:text-teal-200'
                          : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900 text-slate-700 dark:text-slate-400'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={item.val}
                        onChange={(e) => item.set(e.target.checked)}
                        className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                      />
                      <span className="font-medium text-[11px] leading-tight">{item.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Submit Button */}
              <div className="pt-2">
                <Button
                  onClick={() => handleRunClinicalPrediction()}
                  variant="primary"
                  size="md"
                  className="w-full"
                  isLoading={isAnalyzing}
                  leftIcon={<Sparkles className="w-4 h-4" />}
                >
                  Evaluate Clinical Risk &amp; Generate SHAP
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column (7 cols): Prediction, SHAP Waterfall, & Global Importance */}
        <div className="lg:col-span-7 space-y-5">
          {errorMsg && (
            <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Analysis Failure</p>
                <p className="mt-0.5">{errorMsg}</p>
              </div>
            </div>
          )}

          {/* Prediction Result Card */}
          {prediction && (
            <Card className="border-t-4 border-t-teal-500">
              <CardHeader
                title="Clinical Model Assessment"
                subtitle="Independent classification based on 13 patient systemic indicators (Platt Sigmoid Calibrated)"
                action={
                  <Badge variant={CLASS_CONFIG[prediction.predicted_label].variant} size="md">
                    {prediction.predicted_label}
                  </Badge>
                }
              />
              <CardContent className="space-y-4">
                <div className={`p-4 rounded-xl border ${CLASS_CONFIG[prediction.predicted_label].border} ${CLASS_CONFIG[prediction.predicted_label].lightBg} flex items-center justify-between`}>
                  <div>
                    <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
                      Systemic Bone Mineral Risk
                    </span>
                    <h3 className={`text-2xl font-bold mt-0.5 ${CLASS_CONFIG[prediction.predicted_label].text}`}>
                      {(prediction.confidence * 100).toFixed(1)}% {prediction.predicted_label}
                    </h3>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-md">
                      {CLASS_CONFIG[prediction.predicted_label].desc}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] uppercase font-mono text-slate-400">Calibration</span>
                    <p className="text-xs font-bold text-teal-600 dark:text-teal-400">
                      3-Fold Platt Scaling
                    </p>
                    <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                      {prediction.inference_duration_ms} ms
                    </p>
                  </div>
                </div>

                {/* Multiclass Probability Distribution */}
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  {(['Normal', 'Osteopenia', 'Osteoporosis'] as const).map((cls) => {
                    const prob = (prediction.calibrated_probabilities?.[cls] ?? prediction.probabilities[cls]) * 100;
                    const rawProb = (prediction.raw_probabilities?.[cls] ?? prediction.probabilities[cls]) * 100;
                    return (
                      <div key={cls} className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
                        <span className="text-slate-500 font-medium">{cls}</span>
                        <p className="text-base font-bold text-slate-800 dark:text-slate-200 mt-0.5">
                          {prob.toFixed(1)}%
                        </p>
                        <span className="text-[10px] text-slate-400 font-mono">
                          Raw: {rawProb.toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* SHAP Waterfall & Individual Explanation */}
          <Card>
            <CardHeader
              title="SHAP Local Feature Attribution (Waterfall Plot)"
              subtitle="Mathematical contribution of each patient indicator shifting probability from cohort base rate"
              action={
                <div className="flex items-center gap-1">
                  {(['Osteoporosis', 'Osteopenia', 'Normal'] as const).map((cls) => (
                    <button
                      key={cls}
                      type="button"
                      onClick={() => {
                        setSelectedShapClass(cls);
                        handleRunClinicalPrediction(cls);
                      }}
                      className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                        selectedShapClass === cls
                          ? 'bg-teal-600 text-white'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900'
                      }`}
                    >
                      {cls}
                    </button>
                  ))}
                </div>
              }
            />
            <CardContent className="space-y-4">
              {shapExplanation ? (
                <div className="space-y-4">
                  {/* Summary Box */}
                  <div className="p-3 rounded-xl bg-teal-50/50 dark:bg-teal-950/30 border border-teal-200/80 dark:border-teal-800/50 text-xs text-teal-900 dark:text-teal-200 space-y-1">
                    <span className="font-semibold flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-teal-600" />
                      SHAP Clinical Attribution Summary:
                    </span>
                    <p className="leading-relaxed text-[11px]">
                      {shapExplanation.summary}
                    </p>
                  </div>

                  {/* Waterfall Steps Representation */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-xs text-slate-500 font-mono px-1">
                      <span>Baseline Expected Prior: {(shapExplanation.base_value * 100).toFixed(1)}%</span>
                      <span>Target Prediction: {(shapExplanation.final_predicted_value * 100).toFixed(1)}%</span>
                    </div>

                    <div className="space-y-1.5 border border-slate-200 dark:border-slate-800 rounded-xl p-3 bg-white dark:bg-slate-950">
                      {shapExplanation.waterfall_steps.slice(0, 8).map((step, idx) => {
                        const isPositive = step.shap_value > 0;
                        const magnitude = Math.abs(step.shap_value) * 100;

                        return (
                          <div key={idx} className="flex items-center gap-3 text-xs">
                            <span className="w-36 font-medium text-slate-700 dark:text-slate-300 truncate text-[11px]">
                              {step.display_name}
                            </span>
                            <span className="w-24 text-[10px] text-slate-400 truncate font-mono">
                              {step.display_value}
                            </span>
                            <div className="flex-1 flex items-center h-4 relative">
                              <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full relative overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    isPositive ? 'bg-rose-500' : 'bg-emerald-500'
                                  }`}
                                  style={{ width: `${Math.min(magnitude * 2, 100)}%` }}
                                />
                              </div>
                            </div>
                            <span className={`w-16 text-right font-mono font-semibold text-[11px] ${
                              isPositive ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'
                            }`}>
                              {isPositive ? '+' : ''}{(step.shap_value * 100).toFixed(2)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <p className="text-[10px] text-slate-400 dark:text-slate-500">
                    {shapExplanation.disclaimer}
                  </p>
                </div>
              ) : (
                <div className="h-48 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl flex flex-col items-center justify-center text-center p-6 text-slate-400">
                  <BarChart2 className="w-8 h-8 mb-2 text-slate-300 dark:text-slate-600" />
                  <p className="text-xs font-medium">SHAP local explanation will generate upon submitting the clinical form.</p>
                  <p className="text-[11px] mt-0.5">Click "Evaluate Clinical Risk &amp; Generate SHAP" to view feature attribution.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Global Clinical Predictor Importance Table */}
          <Card>
            <CardHeader
              title="Global Feature Importance Ranking"
              subtitle="Permutation and Gini importance across the complete 13-feature clinical contract"
            />
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                      <th className="pb-2 font-semibold">Rank</th>
                      <th className="pb-2 font-semibold">Feature Name</th>
                      <th className="pb-2 font-semibold">Category</th>
                      <th className="pb-2 font-semibold text-right">Relative Weight</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                    {globalImportance.slice(0, 7).map((f, i) => (
                      <tr key={f.feature_name} className="hover:bg-slate-50 dark:hover:bg-slate-900/50">
                        <td className="py-2 font-mono text-slate-400">#{i + 1}</td>
                        <td className="py-2 font-medium text-slate-800 dark:text-slate-200">
                          {f.display_name}
                        </td>
                        <td className="py-2">
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[10px] text-slate-600 dark:text-slate-400">
                            {f.category}
                          </span>
                        </td>
                        <td className="py-2 text-right font-mono font-bold text-teal-600 dark:text-teal-400">
                          {f.percentage.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
