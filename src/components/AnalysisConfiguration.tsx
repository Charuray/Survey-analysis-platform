import React from 'react';
import { TrendingUp, Target, Sparkles } from 'lucide-react';
import { AnalysisConfig } from '../types/survey';

interface AnalysisConfigurationProps {
  headers: string[];
  config: AnalysisConfig;
  onConfigChange: (config: AnalysisConfig) => void;
}

export const AnalysisConfiguration: React.FC<AnalysisConfigurationProps> = ({
  headers,
  config,
  onConfigChange
}) => {
  const updateConfig = (updates: Partial<AnalysisConfig>) => {
      console.log("🔍 Updating config:", updates);

  if (updates.confidenceLevel !== undefined) {
    console.log(`🎯 Confidence Level set to: ${updates.confidenceLevel * 100}%`);
    console.log(
      updates.confidenceLevel === 0.99
        ? "Higher confidence = wider intervals, more conservative estimates"
        : updates.confidenceLevel === 0.9
        ? "Lower confidence = narrower intervals, less certainty"
        : "Balanced 95% confidence level selected"
    );
  }
    onConfigChange({
      ...config,
      ...updates
    });
  };

  const toggleColumn = (column: string, field: 'targetColumns' | 'groupByColumns') => {
    const columns = config[field].includes(column)
      ? config[field].filter(col => col !== column)
      : [...config[field], column];
    
    updateConfig({ [field]: columns });
  };

  // Get smart recommendations
  const getNumericColumnRecommendations = () => {
    return headers.filter(header => {
      // Skip ID-like columns
      if (/id|identifier|key|index|record/i.test(header)) return false;
      // Skip weight-like columns  
      if (/weight|wt|w_|sampling/i.test(header)) return false;
      // This is simplified - in real implementation you'd check actual data types
      return true;
    });
  };
  
  const getGroupingRecommendations = () => {
    return headers.filter(header => 
      /region|state|city|gender|age|group|category|type|class|level|status/i.test(header)
    );
  };
  
  const numericRecommendations = getNumericColumnRecommendations();
  const groupingRecommendations = getGroupingRecommendations();

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="bg-gradient-to-r from-orange-50 to-red-50 p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-orange-600" />
          Analysis Configuration
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Select variables for analysis and configure statistical parameters
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Smart recommendations */}
        {(numericRecommendations.length > 0 || groupingRecommendations.length > 0) && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <Sparkles className="h-5 w-5 text-green-600 mt-0.5" />
              <div className="text-sm text-green-800">
                <p className="font-medium">Analysis Recommendations</p>
                {numericRecommendations.length > 0 && (
                  <p className="mt-1">
                    <strong>Suggested target variables:</strong> {numericRecommendations.slice(0, 5).join(', ')}
                  </p>
                )}
                {groupingRecommendations.length > 0 && (
                  <p className="mt-1">
                    <strong>Potential grouping variables:</strong> {groupingRecommendations.slice(0, 3).join(', ')}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
            <Target className="h-4 w-4 mr-1 text-orange-600" />
            Target Variables for Analysis
            <span className="ml-2 text-xs text-green-600">
              ({config.targetColumns.length} selected)
            </span>
          </label>
          <p className="text-xs text-gray-500 mb-3">
            Select the variables you want to calculate estimates for
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
            {headers.map(header => (
              <label key={header} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.targetColumns.includes(header)}
                  onChange={() => toggleColumn(header, 'targetColumns')}
                  className="rounded border-gray-300 text-orange-600 focus:ring-orange-500"
                />
                <span className={`text-sm ${
                  numericRecommendations.includes(header) 
                    ? 'text-green-700 font-medium' 
                    : 'text-gray-700'
                }`}>
                  {header}
                  {numericRecommendations.includes(header) && ' ⭐'}
                </span>
              </label>
            ))}
          </div>
          
          {config.targetColumns.length === 0 && (
            <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-800">
              ⚠️ Please select at least one target variable for analysis
            </div>
          )}
          
          {config.targetColumns.length > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              Selected: {config.targetColumns.join(', ')}
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Level
          </label>
          <div className="flex space-x-4">
            {[0.90, 0.95, 0.99].map(level => (
              <label key={level} className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="confidenceLevel"
                  value={level}
                  checked={config.confidenceLevel === level}
                  onChange={() => updateConfig({ confidenceLevel: level })}
                  className="border-gray-300 text-orange-600 focus:ring-orange-500"
                />
                <span className="text-sm text-gray-700">{(level * 100)}%</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Higher confidence levels result in wider confidence intervals
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Group By Variables (Optional)
            <span className="ml-2 text-xs text-orange-600">
              ({config.groupByColumns.length} selected)
            </span>
          </label>
          <p className="text-xs text-gray-500 mb-3">
            Select variables to calculate separate estimates for different groups
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
            {headers.map(header => (
              <label key={header} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.groupByColumns.includes(header)}
                  onChange={() => toggleColumn(header, 'groupByColumns')}
                  className="rounded border-gray-300 text-orange-600 focus:ring-orange-500"
                />
                <span className={`text-sm ${
                  groupingRecommendations.includes(header) 
                    ? 'text-blue-700 font-medium' 
                    : 'text-gray-700'
                }`}>
                  {header}
                  {groupingRecommendations.includes(header) && ' ⭐'}
                </span>
              </label>
            ))}
          </div>
          
          {config.groupByColumns.length > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              Selected: {config.groupByColumns.join(', ')}
            </p>
          )}
        </div>

        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Analysis Summary</h4>
          <div className="space-y-1 text-sm text-gray-600">
            <p>Target variables: {config.targetColumns.length} selected</p>
            <p>Confidence level: {(config.confidenceLevel * 100)}%</p>
            <p>
              Grouping: {config.groupByColumns.length === 0 
                ? 'Overall estimates only' 
                : `${config.groupByColumns.length} grouping variables`}
            </p>
            {config.targetColumns.length > 0 && (
              <p className="text-green-600 font-medium">✓ Ready for analysis</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};