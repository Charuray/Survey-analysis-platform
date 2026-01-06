import React, { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { DataPreview } from './components/DataPreview';
import { CleaningConfiguration } from './components/CleaningConfiguration';
import { WeightConfiguration } from './components/WeightConfiguration';
import { AnalysisConfiguration } from './components/AnalysisConfiguration';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultsDisplay } from './components/ResultsDisplay';
import { DataProcessor } from './utils/dataProcessor';
import { StatisticalAnalysis } from './utils/statisticalAnalysis';
import { ReportGenerator } from './utils/reportGenerator';
import { 
  SurveyData, 
  CleaningConfig, 
  WeightConfig, 
  AnalysisConfig, 
  EstimateResult,
  ProcessingStep 
} from './types/survey';
import { 
  ChevronRight, 
  Database, 
  Settings, 
  Scale, 
  TrendingUp, 
  Play, 
  BarChart3,
  AlertCircle,
  Sparkles
} from 'lucide-react';

type Step = 'upload' | 'preview' | 'cleaning' | 'weights' | 'analysis' | 'processing' | 'results';

function App() {
  const [currentStep, setCurrentStep] = useState<Step>('upload');
  const [surveyData, setSurveyData] = useState<SurveyData | null>(null);
  const [cleanedData, setCleanedData] = useState<Record<string, any>[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [cleaningConfig, setCleaningConfig] = useState<CleaningConfig>({
    imputation: {
      method: 'mean',
      columns: []
    },
    outliers: {
      method: 'iqr',
      threshold: 1.5,
      action: 'remove',
      columns: []
    },
    rules: []
  });

  const [weightConfig, setWeightConfig] = useState<WeightConfig>({
    weightColumn: '',
    stratificationColumns: [],
    populationTotals: {}
  });

  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>({
    targetColumns: [],
    confidenceLevel: 0.95,
    groupByColumns: []
  });

  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [estimates, setEstimates] = useState<EstimateResult[]>([]);
  const [descriptiveStats, setDescriptiveStats] = useState<any[]>([]);

  const handleDataLoaded = (data: SurveyData) => {
    setSurveyData(data);
    setError(null);
    setCurrentStep('preview');
    
    // Initialize configurations with intelligent defaults
    setCleaningConfig(prev => ({
      ...prev,
      imputation: { method: 'knn', columns: data.headers },
      outliers: { 
        method: 'winsorization',
        threshold: { lower: 0.05, upper: 0.95 },
        action: 'cap',
        columns: data.headers.filter(header => {
          // Only apply to numeric columns
          const sampleValues = data.rows.slice(0, 100).map(row => row[header]);
          const numericValues = sampleValues.filter(val => 
            !isNaN(parseFloat(val)) && isFinite(val)
          ).length;
          return numericValues > sampleValues.length * 0.7;
        })
      }
    }));
    
    // Auto-detect weight column (look for common weight column names)
    const weightColumnCandidates = data.headers.filter(header => 
      /weight|wt|w_|sampling|survey/i.test(header)
    );
    
    // Auto-detect stratification columns (look for demographic/categorical columns)
    const stratificationCandidates = data.headers.filter(header => {
      const sampleValues = data.rows.slice(0, 100).map(row => row[header]);
      const uniqueValues = new Set(sampleValues).size;
      const totalValues = sampleValues.length;
      // Likely categorical if unique values are less than 20% of total
      return uniqueValues < totalValues * 0.2 && uniqueValues > 1 && uniqueValues < 20;
    });
    
    setWeightConfig({
      weightColumn: weightColumnCandidates[0] || '',
      stratificationColumns: stratificationCandidates.slice(0, 3), // Max 3 strat variables
      populationTotals: {}
    });
    
    // Auto-select numeric columns for analysis (exclude weight and ID columns)
    const numericColumns = data.headers.filter(header => {
      // Skip if it's a weight column
      if (weightColumnCandidates.includes(header)) return false;
      
      // Skip if it looks like an ID column
      if (/id|identifier|key|index/i.test(header)) return false;
      
      // Check if column is numeric
      const sampleValues = data.rows.slice(0, 100).map(row => row[header]);
      const numericValues = sampleValues.filter(val => 
        !isNaN(parseFloat(val)) && isFinite(val)
      ).length;
      return numericValues > sampleValues.length * 0.7;
    });
    
    setAnalysisConfig(prev => ({
      ...prev,
      targetColumns: numericColumns.slice(0, 5), // Auto-select first 5 numeric columns
      groupByColumns: stratificationCandidates.slice(0, 2) // Auto-select up to 2 grouping variables
    }));
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const startProcessing = async () => {
    if (!surveyData) return;
    
    setCurrentStep('processing');
    setError(null);

    const steps: ProcessingStep[] = [
      { id: '1', title: 'Data Validation', status: 'pending' },
      { id: '2', title: 'Data Cleaning', status: 'pending' },
      { id: '3', title: 'Statistical Analysis', status: 'pending' },
      { id: '4', title: 'Report Generation', status: 'pending' }
    ];

    setProcessingSteps(steps);

    try {
      // Step 1: Data Validation
      setProcessingSteps(prev => prev.map(step => 
        step.id === '1' ? { ...step, status: 'processing', message: 'Validating data structure and types...' } : step
      ));
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setProcessingSteps(prev => prev.map(step => 
        step.id === '1' ? { ...step, status: 'complete', message: `Validated ${surveyData.rows.length} records` } : step
      ));

      // Step 2: Data Cleaning
      setProcessingSteps(prev => prev.map(step => 
        step.id === '2' ? { ...step, status: 'processing', message: 'Applying cleaning rules...' } : step
      ));

      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const cleaned = DataProcessor.cleanData(surveyData.rows, cleaningConfig);
      setCleanedData(cleaned);
      
      setProcessingSteps(prev => prev.map(step => 
        step.id === '2' ? { 
          ...step, 
          status: 'complete', 
          message: `Cleaned data: ${cleaned.length} records retained (${((cleaned.length / surveyData.rows.length) * 100).toFixed(1)}%)` 
        } : step
      ));

      // Step 3: Statistical Analysis
      setProcessingSteps(prev => prev.map(step => 
        step.id === '3' ? { ...step, status: 'processing', message: 'Computing estimates and confidence intervals...' } : step
      ));

      await new Promise(resolve => setTimeout(resolve, 2000));

      const estimates = StatisticalAnalysis.calculateEstimates(
        cleaned,
        analysisConfig.targetColumns,
        weightConfig.weightColumn,
        analysisConfig.confidenceLevel
      );

      const descriptiveStats = StatisticalAnalysis.calculateDescriptiveStatistics(
        cleaned,
        analysisConfig.targetColumns
      );

      setEstimates(estimates);
      setDescriptiveStats(descriptiveStats);

      setProcessingSteps(prev => prev.map(step => 
        step.id === '3' ? { 
          ...step, 
          status: 'complete', 
          message: `Analyzed ${analysisConfig.targetColumns.length} variables` 
        } : step
      ));

      // Step 4: Report Generation
      setProcessingSteps(prev => prev.map(step => 
        step.id === '4' ? { ...step, status: 'processing', message: 'Generating reports...' } : step
      ));

      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingSteps(prev => prev.map(step => 
        step.id === '4' ? { ...step, status: 'complete', message: 'Reports ready for download' } : step
      ));

      setCurrentStep('results');

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Processing failed';
      setError(errorMessage);
      setProcessingSteps(prev => prev.map(step => 
        step.status === 'processing' ? { ...step, status: 'error', message: errorMessage } : step
      ));
    }
  };

  const downloadPDF = async () => {
    if (!surveyData || !cleanedData) return;

    try {
      const reportData = {
        filename: surveyData.filename,
        originalRows: surveyData.rows.length,
        cleanedRows: cleanedData.length,
        estimates,
        descriptiveStats,
        processingSteps: processingSteps.filter(step => step.status === 'complete')
      };

      const pdfBlob = await ReportGenerator.generatePDF(reportData);
      
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `survey-analysis-${surveyData.filename.replace('.csv', '')}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to generate PDF report');
    }
  };

  const downloadHTML = () => {
    if (!surveyData || !cleanedData) return;

    try {
      const reportData = {
        filename: surveyData.filename,
        originalRows: surveyData.rows.length,
        cleanedRows: cleanedData.length,
        estimates,
        descriptiveStats,
        processingSteps: processingSteps.filter(step => step.status === 'complete')
      };

      const htmlContent = ReportGenerator.generateHTML(reportData);
      
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `survey-analysis-${surveyData.filename.replace('.csv', '')}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to generate HTML report');
    }
  };

  // Auto-configure weights based on data patterns
  const autoConfigureWeights = (data: SurveyData): WeightConfig => {
    // Look for weight columns
    const weightColumnCandidates = data.headers.filter(header => 
      /weight|wt|w_|sampling|survey|prob/i.test(header)
    );
    
    // Look for stratification variables (categorical with reasonable number of categories)
    const stratificationCandidates = data.headers.filter(header => {
      const sampleValues = data.rows.slice(0, 200).map(row => row[header]);
      const uniqueValues = new Set(sampleValues.filter(v => v !== null && v !== undefined && v !== '')).size;
      const totalValues = sampleValues.filter(v => v !== null && v !== undefined && v !== '').length;
      
      // Good stratification variable: 2-15 unique values, represents at least 5% of data
      return uniqueValues >= 2 && uniqueValues <= 15 && totalValues > data.rows.length * 0.05;
    });
    
    return {
      weightColumn: weightColumnCandidates[0] || '',
      stratificationColumns: stratificationCandidates.slice(0, 3),
      populationTotals: {}
    };
  };
  
  // Auto-configure analysis based on data patterns
  const autoConfigureAnalysis = (data: SurveyData, weights: WeightConfig): AnalysisConfig => {
    // Find numeric columns (excluding weight and ID columns)
    const numericColumns = data.headers.filter(header => {
      // Skip weight column
      if (header === weights.weightColumn) return false;
      
      // Skip ID-like columns
      if (/id|identifier|key|index|record/i.test(header)) return false;
      
      // Check if numeric
      const sampleValues = data.rows.slice(0, 200).map(row => row[header]);
      const numericValues = sampleValues.filter(val => 
        val !== null && val !== undefined && val !== '' && 
        !isNaN(parseFloat(val)) && isFinite(parseFloat(val))
      ).length;
      
      return numericValues > sampleValues.length * 0.7;
    });
    
    // Find good grouping variables (categorical with meaningful categories)
    const groupingCandidates = data.headers.filter(header => {
      if (header === weights.weightColumn) return false;
      
      const sampleValues = data.rows.slice(0, 200).map(row => row[header]);
      const uniqueValues = new Set(sampleValues.filter(v => v !== null && v !== undefined && v !== '')).size;
      
      // Good for grouping: 2-10 categories
      return uniqueValues >= 2 && uniqueValues <= 10;
    });
    
    return {
      targetColumns: numericColumns.slice(0, 6), // Select up to 6 numeric variables
      confidenceLevel: 0.95,
      groupByColumns: groupingCandidates.slice(0, 2) // Select up to 2 grouping variables
    };
  };

  const stepConfig = {
    upload: { title: 'Upload Data', icon: Database, color: 'blue' },
    preview: { title: 'Data Preview', icon: BarChart3, color: 'indigo' },
    cleaning: { title: 'Data Cleaning', icon: Settings, color: 'emerald' },
    weights: { title: 'Survey Weights', icon: Scale, color: 'purple' },
    analysis: { title: 'Analysis Setup', icon: TrendingUp, color: 'orange' },
    processing: { title: 'Processing', icon: Play, color: 'blue' },
    results: { title: 'Results', icon: BarChart3, color: 'green' }
  };

  const steps = Object.keys(stepConfig) as Step[];
  const currentStepIndex = steps.indexOf(currentStep);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Survey Analysis Platform</h1>
                <p className="text-sm text-gray-600">Advanced statistical analysis with automated reporting</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Indicator */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            <div className="flex items-center justify-between">
              {steps.map((step, index) => {
                const config = stepConfig[step];
                const isActive = step === currentStep;
                const isCompleted = index < currentStepIndex;
                const Icon = config.icon;

                return (
                  <React.Fragment key={step}>
                    <div className="flex items-center">
                      <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors ${
                        isActive 
                          ? 'border-blue-600 bg-blue-600 text-white'
                          : isCompleted
                          ? 'border-green-600 bg-green-600 text-white'
                          : 'border-gray-300 bg-white text-gray-400'
                      }`}>
                        <Icon className="h-5 w-5" />
                      </div>
                      <div className="ml-3">
                        <p className={`text-sm font-medium ${
                          isActive ? 'text-blue-600' : isCompleted ? 'text-green-600' : 'text-gray-500'
                        }`}>
                          {config.title}
                        </p>
                      </div>
                    </div>
                    {index < steps.length - 1 && (
                      <ChevronRight className="h-5 w-5 text-gray-400" />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <div className="flex items-start space-x-3">
              <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
              <div>
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-600 hover:text-red-800"
              >
                ×
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentStep === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Upload Survey Data</h2>
              <p className="text-lg text-gray-600">
                Start by uploading your survey data in CSV format. Our platform will analyze it with advanced cleaning and statistical methods.
              </p>
            </div>
            <FileUpload onDataLoaded={handleDataLoaded} onError={handleError} />
          </div>
        )}

        {currentStep === 'preview' && surveyData && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Data Preview</h2>
              <button
                onClick={() => setCurrentStep('cleaning')}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center space-x-2"
              >
                <span>Continue to Cleaning</span>
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
            <DataPreview data={surveyData} cleanedData={cleanedData} showCleaned={!!cleanedData} />
          </div>
        )}

        {currentStep === 'cleaning' && surveyData && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Configure Data Cleaning</h2>
              <div className="flex space-x-3">
                <button
                  onClick={() => setCurrentStep('preview')}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={async () => {
                    // Preview cleaning results
                    const cleaned = DataProcessor.cleanData(surveyData.rows, cleaningConfig);
                    setCleanedData(cleaned);
                    setCurrentStep('preview');
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Preview Cleaned Data</span>
                </button>
                <button
                  onClick={() => setCurrentStep('weights')}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <span>Continue to Weights</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
            <CleaningConfiguration
              headers={surveyData.headers}
              config={cleaningConfig}
              onConfigChange={setCleaningConfig}
            />
          </div>
        )}

        {currentStep === 'weights' && surveyData && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Configure Survey Weights</h2>
              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    // Auto-configure weights based on data patterns
                    const autoWeightConfig = autoConfigureWeights(surveyData);
                    setWeightConfig(autoWeightConfig);
                  }}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Auto-Configure</span>
                </button>
                <button
                  onClick={() => setCurrentStep('cleaning')}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={() => setCurrentStep('analysis')}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <span>Continue to Analysis</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
            <WeightConfiguration
              headers={surveyData.headers}
              config={weightConfig}
              onConfigChange={setWeightConfig}
            />
          </div>
        )}

        {currentStep === 'analysis' && surveyData && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Configure Analysis</h2>
              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    // Auto-configure analysis based on data patterns
                    const autoAnalysisConfig = autoConfigureAnalysis(surveyData, weightConfig);
                    setAnalysisConfig(autoAnalysisConfig);
                  }}
                  className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Auto-Configure</span>
                </button>
                <button
                  onClick={() => setCurrentStep('weights')}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    // Skip to processing with current config
                    if (analysisConfig.targetColumns.length > 0) {
                      startProcessing();
                    }
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium flex items-center space-x-2"
                >
                  <Play className="h-4 w-4" />
                  <span>Quick Start</span>
                </button>
                <button
                  onClick={startProcessing}
                  disabled={analysisConfig.targetColumns.length === 0}
                  className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play className="h-4 w-4" />
                  <span>Start Analysis</span>
                </button>
              </div>
            </div>
            <AnalysisConfiguration
              headers={surveyData.headers}
              config={analysisConfig}
              onConfigChange={setAnalysisConfig}
            />
          </div>
        )}

        {currentStep === 'processing' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Processing Your Data</h2>
              <p className="text-lg text-gray-600">
                Please wait while we clean your data and perform statistical analysis
              </p>
            </div>
            <ProcessingStatus steps={processingSteps} />
          </div>
        )}

        {currentStep === 'results' && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Complete</h2>
              <p className="text-lg text-gray-600">
                Your survey data has been successfully analyzed. Review the results and download reports below.
              </p>
            </div>
            <ResultsDisplay
              estimates={estimates}
              descriptiveStats={descriptiveStats}
              onDownloadPDF={downloadPDF}
              onDownloadHTML={downloadHTML}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;