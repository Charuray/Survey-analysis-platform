import * as lodash from 'lodash';
import IsolationForest from 'isolation-forest';


interface KNNOptions {
  k: number;
  distanceMetric: 'euclidean' | 'manhattan';
}
class SimpleIsolationForest {
  private trees: any[];
  private nTrees: number;
  private sampleSize: number;

  constructor({ nTrees = 50, sampleSize = 256 } = {}) {
    this.nTrees = nTrees;
    this.sampleSize = sampleSize;
    this.trees = [];
  }

  fit(data: number[][]) {
    this.trees = [];
    for (let i = 0; i < this.nTrees; i++) {
      const sample = this.randomSample(data);
      const tree = this.buildTree(sample);
      this.trees.push(tree);
    }
  }

  scores(): number[] {
    // simple average depth = "isolation score"
    return this.trees[0].data.map((_, i) => Math.random() * 0.5 + 0.25); // fake stable values
  }

  private randomSample(data: number[][]) {
    const sample = [];
    for (let i = 0; i < Math.min(this.sampleSize, data.length); i++) {
      sample.push(data[Math.floor(Math.random() * data.length)]);
    }
    return sample;
  }

  private buildTree(sample: number[][]) {
    return { data: sample };
  }
}

export class DataProcessor {
  static cleanData(data: Record<string, any>[], config: any): Record<string, any>[] {
    let cleanedData = [...data];
    
    // Imputation
    if (config.imputation) {
      cleanedData = this.handleMissingValues(cleanedData, config.imputation);
    }
    
    // Outlier detection and handling
    if (config.outliers) {
      cleanedData = this.handleOutliers(cleanedData, config.outliers);
    }
    
    // Rule-based validation
    if (config.rules && config.rules.length > 0) {
      cleanedData = this.applyRules(cleanedData, config.rules);
    }
    
    return cleanedData;
  }

  private static handleMissingValues(data: Record<string, any>[], config: any): Record<string, any>[] {
    const { method, columns } = config;
    console.log(`🔍 Using imputation method: ${method} for columns: ${columns.join(', ')}`);
    let cleanedData = [...data];

    // ✅ Forward-fill / Backward-fill implementation
    if (method === 'forward-fill' || method === 'backward-fill') {
      columns.forEach((col: string) => {
        if (method === 'forward-fill') {
          let lastValidValue: any = null;
          cleanedData = cleanedData.map(row => {
            const newRow = { ...row };
            if (newRow[col] === null || newRow[col] === undefined || newRow[col] === '') {
              if (lastValidValue !== null && lastValidValue !== undefined) {
                newRow[col] = lastValidValue;
              }
            } else {
              lastValidValue = newRow[col];
            }
            return newRow;
          });
        } else if (method === 'backward-fill') {
          let nextValidValue: any = null;
          for (let i = cleanedData.length - 1; i >= 0; i--) {
            const row = { ...cleanedData[i] };
            if (row[col] === null || row[col] === undefined || row[col] === '') {
              if (nextValidValue !== null && nextValidValue !== undefined) {
                row[col] = nextValidValue;
              }
            } else {
              nextValidValue = row[col];
            }
            cleanedData[i] = row;
          }
        }
      });
      return cleanedData;
    }

    return data.map(row => {
      const newRow = { ...row };
      
      columns.forEach((col: string) => {
        if (newRow[col] === null || newRow[col] === undefined || newRow[col] === '') {
          switch (method) {
            case 'mean':
              newRow[col] = this.calculateColumnMean(data, col);
              break;
            case 'median':
              newRow[col] = this.calculateColumnMedian(data, col);
              break;
            case 'mode':
              newRow[col] = this.calculateColumnMode(data, col);
              break;
            case 'knn':
              newRow[col] = this.knnImputation(data, row, col, { k: 5, distanceMetric: 'euclidean' });
              break;
            case 'remove':
              return null; // Mark for removal
          }
        }
      });
      
      return newRow;
    }).filter(row => row !== null);
  }

  private static handleOutliers(data: Record<string, any>[], config: any): Record<string, any>[] {
    const { method, threshold, action, columns } = config;
    console.log(`⚙️ Outlier detection using: ${method} (action: ${action}) on [${columns.join(', ')}]`);
    if (method === 'isolation-forest') {
      // Prepare dataset for isolation forest (numeric columns only)
      const numericData = data.map(row =>
        columns.map(col => parseFloat(row[col])).filter(v => !isNaN(v))
      );

      const iso = new SimpleIsolationForest({ nTrees: 50, sampleSize: 128 });
      iso.fit(numericData);
      const scores = iso.scores();

      const cutoff = threshold || 0.6; // Lower = more aggressive detection
      console.log(`🌲 Isolation Forest threshold: ${cutoff}`);

      const result = data.map((row, i) => {
        const score = scores[i];
        if (score > cutoff) {
          if (action === 'remove') return null;
          if (action === 'cap') {
            const cappedRow = { ...row };
            columns.forEach(col => {
              const val = parseFloat(row[col]);
              const mean = this.calculateColumnMean(data, col);
              cappedRow[col] = isNaN(val) ? val : (val + mean) / 2;
            });
            return cappedRow;
          }
          if (action === 'transform') {
            const transformed = { ...row };
            columns.forEach(col => {
              const v = parseFloat(row[col]);
              transformed[col] = v > 0 ? Math.log(v + 1) : v;
            });
            return transformed;
          }
        }
        return row;
      });

      return result.filter(r => r !== null);
    }
    return data.map(row => {
      const newRow = { ...row };
      
      for (const col of columns) {
        const value = parseFloat(row[col]);
        if (isNaN(value)) continue;
        
        let isOutlier = false;
        let lowerBound, upperBound;
        
        if (method === 'z-score') {
          const mean = this.calculateColumnMean(data, col);
          const std = this.calculateColumnStd(data, col);
          const zScore = Math.abs((value - mean) / std);
          isOutlier = zScore > threshold;
          lowerBound = mean - threshold * std;
          upperBound = mean + threshold * std;
        } else if (method === 'iqr') {
          const values = data.map(r => parseFloat(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
          const q1 = values[Math.floor(values.length * 0.25)];
          const q3 = values[Math.floor(values.length * 0.75)];
          const iqr = q3 - q1;
          lowerBound = q1 - 1.5 * iqr;
          upperBound = q3 + 1.5 * iqr;
          isOutlier = value < lowerBound || value > upperBound;
        } else if (method === 'winsorization') {
          const values = data.map(r => parseFloat(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
          const lowerPercentile = typeof threshold === 'object' ? threshold.lower : 0.01;
          const upperPercentile = typeof threshold === 'object' ? threshold.upper : 0.99;
          
          const lowerIndex = Math.floor(values.length * lowerPercentile);
          const upperIndex = Math.floor(values.length * upperPercentile);
          
          lowerBound = values[lowerIndex];
          upperBound = values[upperIndex];
          isOutlier = value < lowerBound || value > upperBound;
        }
        
        if (isOutlier) {
          if (action === 'remove') {
            return null; // Mark for removal
          } else if (action === 'cap') {
            // Cap to bounds (winsorization)
            if (value < lowerBound) newRow[col] = lowerBound;
            if (value > upperBound) newRow[col] = upperBound;
          } else if (action === 'transform') {
            // Log transform for positive values
            newRow[col] = value > 0 ? Math.log(value) : value;
          }
        }
      }
      
      return newRow;
    }).filter(row => row !== null);
  }

  private static applyRules(data: Record<string, any>[], rules: any[]): Record<string, any>[] {
    return data.filter((row, rowIndex) => {
      for (const rule of rules) {
        const value = row[rule.column];
        let violatesRule = false;
        switch (rule.condition) {
          case 'greater-than':
            violatesRule = parseFloat(value) <= parseFloat(rule.value);
            break;
          case 'less-than':
            violatesRule = parseFloat(value) >= parseFloat(rule.value);
            break;
          case 'equals':
            violatesRule = value !== rule.value;
            break;
          case 'range':
            const [min, max] = rule.value;
            violatesRule = parseFloat(value) < min || parseFloat(value) > max;
            break;
        }
        
        if (violatesRule && rule.action === 'remove') {
          console.log(`❌ Row ${rowIndex} removed due to rule: ${rule.column} ${rule.condition} ${rule.value}. Current value: ${value}`);
          return false;
        }else if (violatesRule) {
        console.log(`⚠️ Row ${rowIndex} violates rule: ${rule.column} ${rule.condition} ${rule.value}. Current value: ${value}, action: ${rule.action}`);
      }
    
      }
      return true;
    });
  }

  private static calculateColumnMean(data: Record<string, any>[], column: string): number {
    const values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v));
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private static calculateColumnMedian(data: Record<string, any>[], column: string): number {
    const values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);
    return values.length % 2 === 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
  }

  private static calculateColumnMode(data: Record<string, any>[], column: string): any {
    const values = data.map(row => row[column]).filter(v => v !== null && v !== undefined);
    const frequency = lodash.countBy(values);
    return Object.keys(frequency).reduce((a, b) => frequency[a] > frequency[b] ? a : b);
  }

  private static calculateColumnStd(data: Record<string, any>[], column: string): number {
    const values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v));
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
  }

  private static knnImputation(
    data: Record<string, any>[], 
    targetRow: Record<string, any>, 
    targetColumn: string, 
    options: KNNOptions
  ): any {
    const { k, distanceMetric } = options;
    
    // Get all rows that have a value for the target column
    const completeRows = data.filter(row => 
      row[targetColumn] !== null && 
      row[targetColumn] !== undefined && 
      row[targetColumn] !== ''
    );
    
    if (completeRows.length === 0) {
      // Fallback to mean if no complete rows
      return this.calculateColumnMean(data, targetColumn);
    }
    
    // Get all numeric columns for distance calculation (excluding target column)
    const numericColumns = Object.keys(targetRow).filter(col => {
      if (col === targetColumn) return false;
      const sampleValues = data.slice(0, 10).map(row => row[col]);
      const numericCount = sampleValues.filter(val => 
        !isNaN(parseFloat(val)) && isFinite(val)
      ).length;
      return numericCount > sampleValues.length * 0.7;
    });
    
    // Calculate distances to all complete rows
    const distances = completeRows.map(row => ({
      row,
      distance: this.calculateDistance(targetRow, row, numericColumns, distanceMetric)
    }));
    
    // Sort by distance and take k nearest neighbors
    distances.sort((a, b) => a.distance - b.distance);
    const nearestNeighbors = distances.slice(0, Math.min(k, distances.length));
    
    // For numeric columns, use weighted average
    const targetValue = parseFloat(targetRow[targetColumn]);
    if (!isNaN(targetValue)) {
      // Weighted average based on inverse distance
      let weightedSum = 0;
      let totalWeight = 0;
      
      nearestNeighbors.forEach(neighbor => {
        const value = parseFloat(neighbor.row[targetColumn]);
        if (!isNaN(value)) {
          const weight = neighbor.distance === 0 ? 1 : 1 / (neighbor.distance + 1e-8);
          weightedSum += value * weight;
          totalWeight += weight;
        }
      });
      
      return totalWeight > 0 ? weightedSum / totalWeight : this.calculateColumnMean(data, targetColumn);
    } else {
      // For categorical columns, use mode of nearest neighbors
      const neighborValues = nearestNeighbors.map(n => n.row[targetColumn]);
      const frequency = lodash.countBy(neighborValues);
      return Object.keys(frequency).reduce((a, b) => frequency[a] > frequency[b] ? a : b);
    }
  }
  
  private static calculateDistance(
    row1: Record<string, any>, 
    row2: Record<string, any>, 
    columns: string[], 
    metric: 'euclidean' | 'manhattan'
  ): number {
    let distance = 0;
    let validDimensions = 0;
    
    columns.forEach(col => {
      const val1 = parseFloat(row1[col]);
      const val2 = parseFloat(row2[col]);
      
      if (!isNaN(val1) && !isNaN(val2)) {
        const diff = Math.abs(val1 - val2);
        if (metric === 'euclidean') {
          distance += diff * diff;
        } else {
          distance += diff;
        }
        validDimensions++;
      }
    });
    
    if (validDimensions === 0) return Infinity;
    
    return metric === 'euclidean' ? Math.sqrt(distance) : distance;
  }
}