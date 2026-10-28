import React from 'react';
import {
  Info,
  ShieldCheck,
  BookOpen,
  GitBranch,
  FileCode2,
  Cpu,
  Layers,
  HeartPulse,
  ExternalLink,
} from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Info className="w-5 h-5 text-teal-600" />
          About the Research & Scientific Architecture
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Multimodal Deep Learning-Based Osteoporosis Classification Using Knee X-Ray Images.
        </p>
      </div>

      {/* Abstract Card */}
      <Card>
        <CardHeader
          title="Research Objective & Motivation"
          subtitle="Opportunistic screening for osteoporosis from routine knee radiographs"
        />
        <CardContent className="text-xs space-y-3 text-slate-700 dark:text-slate-300 leading-relaxed">
          <p>
            Osteoporosis is a systemic skeletal disease characterized by low bone mineral density (BMD) and microarchitectural
            deterioration, predisposing patients to fragility fractures. Dual-energy X-ray absorptiometry (DXA) is the clinical gold
            standard for BMD assessment; however, DXA scanners are specialized, expensive, and often unavailable in primary or rural care.
          </p>
          <p>
            Knee radiographs, routinely ordered for osteoarthritis, trauma, and joint pain, capture distal femoral and proximal tibial
            cancellous bone containing rich trabecular textures and cortical boundaries. This research platform investigates the feasibility
            of opportunistic screening by combining deep convolutional image representations with patient clinical risk factors.
          </p>
        </CardContent>
      </Card>

      {/* Multimodal Architecture Diagram Box */}
      <Card>
        <CardHeader
          title="Multimodal Pipeline Flowchart"
          subtitle="Dual-branch late fusion combining visual features and clinical vectors"
        />
        <CardContent>
          <div className="p-5 rounded-xl bg-slate-900 text-slate-100 font-mono text-xs overflow-x-auto space-y-3">
            <div className="text-teal-400 font-bold mb-2">// MULTIMODAL INFERENCE ARCHITECTURE</div>
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-3 rounded-lg bg-slate-800/80 border border-slate-700">
              <div className="text-center p-2 rounded bg-slate-700/60 w-full md:w-auto">
                <div className="font-semibold text-white">Knee Radiograph</div>
                <div className="text-[10px] text-slate-400">RGB 224×224 (Letterbox)</div>
              </div>
              <div className="text-teal-400 font-bold">➔</div>
              <div className="text-center p-2 rounded bg-slate-700/60 w-full md:w-auto">
                <div className="font-semibold text-teal-300">DenseNet121 CNN</div>
                <div className="text-[10px] text-slate-400">1024-dim Image Embedding</div>
              </div>
              <div className="text-teal-400 font-bold">➔</div>
              <div className="text-center p-3 rounded bg-teal-950/80 border border-teal-500 w-full md:w-auto">
                <div className="font-bold text-white">Fusion Classifier</div>
                <div className="text-[10px] text-teal-300">1056-dim ➔ 3-Class Softmax</div>
              </div>
              <div className="text-teal-400 font-bold">➔</div>
              <div className="text-center p-2 rounded bg-slate-700/60 w-full md:w-auto">
                <div className="font-semibold text-emerald-400">Class Probs</div>
                <div className="text-[10px] text-slate-400">Normal / Penia / Porosis</div>
              </div>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-3 rounded-lg bg-slate-800/80 border border-slate-700">
              <div className="text-center p-2 rounded bg-slate-700/60 w-full md:w-auto">
                <div className="font-semibold text-white">Clinical Features</div>
                <div className="text-[10px] text-slate-400">Age, BMI, Menopause, etc.</div>
              </div>
              <div className="text-teal-400 font-bold">➔</div>
              <div className="text-center p-2 rounded bg-slate-700/60 w-full md:w-auto">
                <div className="font-semibold text-sky-300">Clinical MLP Encoder</div>
                <div className="text-[10px] text-slate-400">19-dim ➔ 32-dim Embedding</div>
              </div>
              <div className="text-teal-400 font-bold">➔</div>
              <div className="text-center p-2 text-slate-400 italic text-[11px] w-full md:w-auto">
                (Concatenated with Image Embedding)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Ethical Statement & Diagnostic Disclaimers */}
      <Card className="border-amber-300 dark:border-amber-800">
        <CardHeader
          title="Ethical Declarations & Research Safeguards"
          subtitle="Adherence to scientific integrity, data privacy, and medical caution"
          action={<Badge variant="osteopenia" size="sm">Strict Governance</Badge>}
        />
        <CardContent className="text-xs space-y-3 text-slate-600 dark:text-slate-400">
          <div className="flex items-start gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Zero Diagnostic Automation:</strong> This platform is designed solely for retrospective clinical
              investigation and algorithmic benchmarking. It does not provide medical diagnoses, treatment decisions, or
              prescriptions.
            </p>
          </div>
          <div className="flex items-start gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Data Privacy & Zero PHI Retention:</strong> Uploaded radiographs are processed in memory and
              ephemeral runtime storage only; no raw images, patient names, or medical records are retained or transmitted
              to external third parties.
            </p>
          </div>
          <div className="flex items-start gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              <strong>Transparent Ground Truth Provenance:</strong> Diagnostic classifications in the open Mendeley cohort are
              QUS-derived calcaneal ultrasound proxies. The authors refrain from conflating ultrasound T-scores with definitive DXA
              central axial measurements.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
