import { EstimateResult } from '../types/survey';

export class StatisticalAnalysis {
  static calculateEstimates(
    data: Record<string, any>[],
    targetColumns: string[],
    weightColumn?: string,
    confidenceLevel: number = 0.95
  ): EstimateResult[] {
    const results: EstimateResult[] = [];
    const alpha = 1 - confidenceLevel;
    const tCritical = this.getTCriticalValue(data.length - 1, alpha / 2);

    targetColumns.forEach(column => {
      const values = data.map(row => ({
        value: parseFloat(row[column]),
        weight: weightColumn ? parseFloat(row[weightColumn]) || 1 : 1
      })).filter(item => !isNaN(item.value));

      if (values.length === 0) return;

      const weightedMean = this.calculateWeightedMean(values);
      const standardError = this.calculateStandardError(values, weightedMean);
      const marginOfError = tCritical * standardError;

      results.push({
        variable: column,
        estimate: weightedMean,
        standardError,
        marginOfError,
        confidenceInterval: [
          weightedMean - marginOfError,
          weightedMean + marginOfError
        ],
        sampleSize: values.length
      });
    });

    return results;
  }

  private static calculateWeightedMean(values: { value: number; weight: number }[]): number {
    const totalWeightedValue = values.reduce((sum, item) => sum + (item.value * item.weight), 0);
    const totalWeight = values.reduce((sum, item) => sum + item.weight, 0);
    return totalWeightedValue / totalWeight;
  }

  private static calculateStandardError(
    values: { value: number; weight: number }[],
    mean: number
  ): number {
    const weightedVariance = values.reduce((sum, item) => {
      return sum + item.weight * Math.pow(item.value - mean, 2);
    }, 0) / values.reduce((sum, item) => sum + item.weight, 0);
    
    return Math.sqrt(weightedVariance / values.length);
  }

  private static getTCriticalValue(degreesOfFreedom: number, alpha: number): number {
    // Simplified t-critical value calculation
    // In a real application, you'd use a more precise t-distribution table
    if (degreesOfFreedom >= 30) return 1.96; // Normal approximation
    if (degreesOfFreedom >= 20) return 2.086;
    if (degreesOfFreedom >= 10) return 2.228;
    return 2.571; // Conservative estimate for small samples
  }

  static calculateDescriptiveStatistics(data: Record<string, any>[], columns: string[]) {
    return columns.map(column => {
      const values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v));
      
      if (values.length === 0) {
        return { column, count: 0, mean: 0, median: 0, std: 0, min: 0, max: 0 };
      }

      values.sort((a, b) => a - b);
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const median = values.length % 2 === 0 
        ? (values[values.length / 2 - 1] + values[values.length / 2]) / 2
        : values[Math.floor(values.length / 2)];
      
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);

      return {
        column,
        count: values.length,
        mean: Number(mean.toFixed(4)),
        median: Number(median.toFixed(4)),
        std: Number(std.toFixed(4)),
        min: values[0],
        max: values[values.length - 1]
      };
    });
  }
}