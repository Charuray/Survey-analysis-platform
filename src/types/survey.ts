export interface SurveyData {
  headers: string[];
  rows: Record<string, any>[];
  filename: string;
  uploadTime: Date;
}

export interface CleaningConfig {
  imputation: {
    method: 'mean' | 'median' | 'mode' | 'forward-fill' | 'backward-fill' | 'remove';
    columns: string[];
  };
  outliers: {
    method: 'iqr' | 'z-score' | 'isolation-forest';
    threshold: number;
    action: 'remove' | 'cap' | 'transform';
    columns: string[];
  };
  rules: {
    id: string;
    column: string;
    condition: 'greater-than' | 'less-than' | 'equals' | 'not-equals' | 'range' | 'pattern';
    value: any;
    action: 'flag' | 'remove' | 'transform';
  }[];
}

export interface WeightConfig {
  weightColumn: string;
  stratificationColumns: string[];
  populationTotals?: Record<string, number>;
}

export interface AnalysisConfig {
  targetColumns: string[];
  confidenceLevel: number;
  groupByColumns: string[];
}

export interface EstimateResult {
  variable: string;
  estimate: number;
  standardError: number;
  marginOfError: number;
  confidenceInterval: [number, number];
  sampleSize: number;
}

export interface ProcessingStep {
  id: string;
  title: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
  message?: string;
}