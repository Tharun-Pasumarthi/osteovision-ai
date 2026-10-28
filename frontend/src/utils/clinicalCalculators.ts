/**
 * Pure calculation and conversion utilities for clinical osteoporosis screening features.
 * Adheres strictly to the 13 clinical features contract and WHO clinical definitions.
 */

export interface HeightEquivalents {
  metres: number;
  centimetres: number;
  feet: number;
  inches: number;
  formattedFtIn: string;
}

export type BmiCategory = 'Underweight' | 'Normal weight' | 'Overweight' | 'Obese' | 'Invalid';

export interface BmiResult {
  bmi: number | null;
  category: BmiCategory;
  categoryColor: string;
  isValid: boolean;
  message?: string;
}

export interface DistanceEquivalents {
  km: number;
  metres: number;
  steps: number;
  assumedStepLengthM: number;
}

// -------------------------------------------------------------
// Height Conversions
// -------------------------------------------------------------

/**
 * Convert centimetres to metres.
 * Example: 163 cm -> 1.63 m
 */
export function cmToMetres(cm: number): number {
  if (isNaN(cm) || cm <= 0) return 0;
  return Number((cm / 100).toFixed(4));
}

/**
 * Convert metres to centimetres.
 * Example: 1.63 m -> 163 cm
 */
export function metresToCm(m: number): number {
  if (isNaN(m) || m <= 0) return 0;
  return Number((m * 100).toFixed(2));
}

/**
 * Convert feet and inches to metres.
 * 1 inch = 0.0254 m, 1 foot = 12 inches = 0.3048 m
 * Example: 5 ft 8 in = (5 * 12 + 8) * 0.0254 = 68 * 0.0254 = 1.7272 m
 */
export function ftInToMetres(feet: number, inches: number): number {
  const totalInches = (Number(feet) || 0) * 12 + (Number(inches) || 0);
  if (totalInches <= 0) return 0;
  return Number((totalInches * 0.0254).toFixed(4));
}

/**
 * Convert metres to feet and decimal inches.
 * Example: 1.70 m -> 5 ft 6.9 in
 */
export function metresToFtIn(m: number): { feet: number; inches: number; formatted: string } {
  if (isNaN(m) || m <= 0) {
    return { feet: 0, inches: 0, formatted: "0 ft 0 in" };
  }
  const totalInches = m / 0.0254;
  const feet = Math.floor(totalInches / 12);
  const inches = Number((totalInches % 12).toFixed(1));
  return {
    feet,
    inches,
    formatted: `${feet} ft ${inches} in`,
  };
}

/**
 * Compute all height representations for a given height in metres.
 */
export function computeHeightEquivalents(heightM: number): HeightEquivalents | null {
  if (!heightM || isNaN(heightM) || heightM <= 0) {
    return null;
  }
  const cm = metresToCm(heightM);
  const { feet, inches, formatted } = metresToFtIn(heightM);
  return {
    metres: Number(heightM.toFixed(4)),
    centimetres: cm,
    feet,
    inches,
    formattedFtIn: formatted,
  };
}

// -------------------------------------------------------------
// BMI Calculation
// -------------------------------------------------------------

/**
 * Calculate BMI following the standard clinical convention:
 * BMI = weight (kg) / [height (m)]^2
 *
 * Requirements:
 * - Height must be in metres (> 0.5 m, <= 2.5 m)
 * - Weight must be in kilograms (>= 20 kg, <= 250 kg)
 * - Returns null if inputs are missing, zero, or invalid.
 */
export function calculateBmi(heightM: number | null | undefined, weightKg: number | null | undefined): BmiResult {
  if (
    heightM === null ||
    heightM === undefined ||
    weightKg === null ||
    weightKg === undefined ||
    isNaN(heightM) ||
    isNaN(weightKg) ||
    heightM <= 0.5 ||
    heightM > 2.5 ||
    weightKg < 20 ||
    weightKg > 250
  ) {
    return {
      bmi: null,
      category: 'Invalid',
      categoryColor: 'text-slate-400 bg-slate-100 dark:bg-slate-800',
      isValid: false,
      message: 'Enter valid height (0.50–2.50 m) and weight (20–250 kg) to calculate BMI.',
    };
  }

  const rawBmi = weightKg / (heightM * heightM);
  if (!isFinite(rawBmi) || isNaN(rawBmi) || rawBmi <= 0) {
    return {
      bmi: null,
      category: 'Invalid',
      categoryColor: 'text-slate-400 bg-slate-100 dark:bg-slate-800',
      isValid: false,
      message: 'Unable to calculate BMI from provided measurements.',
    };
  }

  const roundedBmi = Number(rawBmi.toFixed(2));

  let category: BmiCategory = 'Normal weight';
  let categoryColor = 'text-emerald-700 bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800';

  if (roundedBmi < 18.5) {
    category = 'Underweight';
    categoryColor = 'text-amber-700 bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800';
  } else if (roundedBmi < 25.0) {
    category = 'Normal weight';
    categoryColor = 'text-emerald-700 bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800';
  } else if (roundedBmi < 30.0) {
    category = 'Overweight';
    categoryColor = 'text-amber-700 bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800';
  } else {
    category = 'Obese';
    categoryColor = 'text-rose-700 bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800';
  }

  return {
    bmi: roundedBmi,
    category,
    categoryColor,
    isValid: true,
  };
}

// -------------------------------------------------------------
// Walking Distance Conversions
// -------------------------------------------------------------

export const DEFAULT_STEP_LENGTH_M = 0.75; // Average adult stride length ~ 0.75m / step

/**
 * Convert steps to kilometres using assumed step length.
 * Formula: km = (steps * step_length_m) / 1000
 * Example: 1,000 steps * 0.75 m / 1,000 = 0.75 km
 */
export function stepsToKm(steps: number, stepLengthM: number = DEFAULT_STEP_LENGTH_M): number {
  if (isNaN(steps) || steps <= 0) return 0;
  return Number(((steps * stepLengthM) / 1000).toFixed(2));
}

/**
 * Convert metres to kilometres.
 * Formula: km = metres / 1000
 * Example: 2,500 m -> 2.50 km
 */
export function metresToKm(metres: number): number {
  if (isNaN(metres) || metres <= 0) return 0;
  return Number((metres / 1000).toFixed(2));
}

/**
 * Convert kilometres to steps using assumed step length.
 * Formula: steps = round((km * 1000) / step_length_m)
 */
export function kmToSteps(km: number, stepLengthM: number = DEFAULT_STEP_LENGTH_M): number {
  if (isNaN(km) || km <= 0 || stepLengthM <= 0) return 0;
  return Math.round((km * 1000) / stepLengthM);
}

/**
 * Compute all walking distance equivalents for a given km value.
 */
export function computeDistanceEquivalents(km: number, stepLengthM: number = DEFAULT_STEP_LENGTH_M): DistanceEquivalents {
  const safeKm = isNaN(km) || km < 0 ? 0 : Number(km.toFixed(2));
  return {
    km: safeKm,
    metres: Math.round(safeKm * 1000),
    steps: kmToSteps(safeKm, stepLengthM),
    assumedStepLengthM: stepLengthM,
  };
}

