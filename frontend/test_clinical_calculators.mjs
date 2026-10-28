/**
 * Frontend Clinical Calculators Unit Verification Suite
 */
import assert from 'node:assert';
import {
  cmToMetres,
  metresToCm,
  ftInToMetres,
  metresToFtIn,
  computeHeightEquivalents,
  calculateBmi,
  stepsToKm,
  metresToKm,
  kmToSteps,
  computeDistanceEquivalents,
  DEFAULT_STEP_LENGTH_M,
} from './src/utils/clinicalCalculators.ts';

console.log('Running Frontend Clinical Calculator Verification Suite...\n');

let passed = 0;
let total = 0;

function it(name, fn) {
  total++;
  try {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error('    Error:', err.message);
  }
}

// -------------------------------------------------------------
// 1. Height Conversions
// -------------------------------------------------------------
console.log('[Test Suite 1: Height Conversions]');

it('163 cm equals 1.63 m exactly', () => {
  const m = cmToMetres(163);
  assert.strictEqual(m, 1.63);
});

it('1.63 m equals 163 cm', () => {
  const cm = metresToCm(1.63);
  assert.strictEqual(cm, 163);
});

it('5 ft 8 in equals 1.7272 m', () => {
  const m = ftInToMetres(5, 8);
  assert.strictEqual(m, 1.7272);
});

it('1.70 m converts to approx 5 ft 6.9 in', () => {
  const res = metresToFtIn(1.70);
  assert.strictEqual(res.feet, 5);
  assert.strictEqual(res.inches, 6.9);
  assert.strictEqual(res.formatted, '5 ft 6.9 in');
});

it('computeHeightEquivalents produces consistent representations', () => {
  const eq = computeHeightEquivalents(1.65);
  assert.ok(eq !== null);
  assert.strictEqual(eq.metres, 1.65);
  assert.strictEqual(eq.centimetres, 165);
  assert.strictEqual(eq.feet, 5);
});

it('Invalid height returns 0 or null equivalents', () => {
  assert.strictEqual(cmToMetres(-10), 0);
  assert.strictEqual(computeHeightEquivalents(0), null);
  assert.strictEqual(computeHeightEquivalents(-1.5), null);
});

// -------------------------------------------------------------
// 2. BMI Calculations
// -------------------------------------------------------------
console.log('\n[Test Suite 2: BMI Calculations & WHO Categorization]');

it('Calculates standard BMI: 64kg / (1.58m)^2 -> 25.64 (Overweight)', () => {
  const res = calculateBmi(1.58, 64.0);
  assert.strictEqual(res.isValid, true);
  assert.strictEqual(res.bmi, 25.64);
  assert.strictEqual(res.category, 'Overweight');
});

it('Calculates normal weight BMI: 60kg / (1.70m)^2 -> 20.76 (Normal weight)', () => {
  const res = calculateBmi(1.70, 60.0);
  assert.strictEqual(res.isValid, true);
  assert.strictEqual(res.bmi, 20.76);
  assert.strictEqual(res.category, 'Normal weight');
});

it('Calculates underweight BMI: 42kg / (1.65m)^2 -> 15.43 (Underweight)', () => {
  const res = calculateBmi(1.65, 42.0);
  assert.strictEqual(res.isValid, true);
  assert.strictEqual(res.bmi, 15.43);
  assert.strictEqual(res.category, 'Underweight');
});

it('Calculates obese BMI: 95kg / (1.60m)^2 -> 37.11 (Obese)', () => {
  const res = calculateBmi(1.60, 95.0);
  assert.strictEqual(res.isValid, true);
  assert.strictEqual(res.bmi, 37.11);
  assert.strictEqual(res.category, 'Obese');
});

it('Does not calculate BMI for empty, zero, or out-of-range inputs', () => {
  assert.strictEqual(calculateBmi(null, 64).isValid, false);
  assert.strictEqual(calculateBmi(1.60, null).isValid, false);
  assert.strictEqual(calculateBmi(0, 64).isValid, false);
  assert.strictEqual(calculateBmi(1.60, 0).isValid, false);
  assert.strictEqual(calculateBmi(0.3, 60).isValid, false);
  assert.strictEqual(calculateBmi(1.60, 15).isValid, false);
});

// -------------------------------------------------------------
// 3. Walking Distance Conversions
// -------------------------------------------------------------
console.log('\n[Test Suite 3: Walking Distance & Step Estimation]');

it('1,000 steps with 0.75m step length equals 0.75 km', () => {
  const km = stepsToKm(1000, 0.75);
  assert.strictEqual(km, 0.75);
});

it('2,500 metres equals 2.50 km', () => {
  const km = metresToKm(2500);
  assert.strictEqual(km, 2.50);
});

it('3.0 km converts to approx 4,000 steps at 0.75m/step', () => {
  const steps = kmToSteps(3.0, 0.75);
  assert.strictEqual(steps, 4000);
});

it('computeDistanceEquivalents generates all 3 units accurately', () => {
  const eq = computeDistanceEquivalents(2.5, 0.75);
  assert.strictEqual(eq.km, 2.5);
  assert.strictEqual(eq.metres, 2500);
  assert.strictEqual(eq.steps, 3333);
  assert.strictEqual(eq.assumedStepLengthM, 0.75);
});

console.log(`\nVerification Summary: ${passed} / ${total} tests passed.`);
if (passed !== total) {
  process.exit(1);
}
