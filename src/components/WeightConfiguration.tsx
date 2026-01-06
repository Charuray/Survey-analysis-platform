import React from 'react';
import { Scale, Info, Sparkles } from 'lucide-react';
import { WeightConfig } from '../types/survey';

interface WeightConfigurationProps {
  headers: string[];
  config: WeightConfig;
  onConfigChange: (config: WeightConfig) => void;
}

export const WeightConfiguration: React.FC<WeightConfigurationProps> = ({
  headers,
  config,
  onConfigChange
}) => {
  const updateConfig = (updates: Partial<WeightConfig>) => {
    onConfigChange({
      ...config,
      ...updates
    });
  };

  const toggleStratificationColumn = (column: string) => {
    const columns = config.stratificationColumns.includes(column)
      ? config.stratificationColumns.filter(col => col !== column)
      : [...config.stratificationColumns, column];
    
    updateConfig({ stratificationColumns: columns });
  };

  // Get column recommendations
  const getWeightColumnRecommendations = () => {
    return headers.filter(header => 
      /weight|wt|w_|sampling|survey|prob/i.test(header)
    );
  };
  
  const getStratificationRecommendations = () => {
    return headers.filter(header => {
      // This is a simplified check - in real implementation you'd analyze the data
      return /region|state|city|gender|age|group|category|type|class/i.test(header);
    });
  };
  
  const weightRecommendations = getWeightColumnRecommendations();
  const stratRecommendations = getStratificationRecommendations();

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <Scale className="h-5 w-5 mr-2 text-purple-600" />
          Survey Weight Configuration
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Configure survey weights for accurate population estimates
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Auto-configuration suggestions */}
        {(weightRecommendations.length > 0 || stratRecommendations.length > 0) && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <Sparkles className="h-5 w-5 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">Smart Recommendations</p>
                {weightRecommendations.length > 0 && (
                  <p className="mt-1">
                    <strong>Potential weight columns:</strong> {weightRecommendations.join(', ')}
                  </p>
                )}
                {stratRecommendations.length > 0 && (
                  <p className="mt-1">
                    <strong>Suggested stratification variables:</strong> {stratRecommendations.slice(0, 3).join(', ')}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <Info className="h-5 w-5 text-blue-600 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-medium">About Survey Weights</p>
              <p className="mt-1">
                Survey weights adjust for sampling design and non-response to ensure your estimates 
                represent the target population. If your data doesn't include weights, the analysis 
                will assume equal probability sampling.
              </p>
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Weight Column
            {weightRecommendations.length > 0 && (
              <span className="ml-2 text-xs text-blue-600">
                (Recommended: {weightRecommendations[0]})
              </span>
            )}
          </label>
          <select
            value={config.weightColumn}
            onChange={(e) => updateConfig({ weightColumn: e.target.value })}
            className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 ${
              weightRecommendations.includes(config.weightColumn) 
                ? 'border-blue-300 bg-blue-50' 
                : 'border-gray-300'
            }`}
          >
            <option value="">No weights (equal probability sampling)</option>
            {headers.map(header => (
              <option key={header} value={header}>
                {header}
                {weightRecommendations.includes(header) && ' ⭐'}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Select the column containing survey weights, or leave blank for unweighted analysis
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Stratification Variables
          </label>
          <p className="text-xs text-gray-500 mb-3">
            Select columns used for stratification in the survey design (optional)
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
            {headers.map(header => (
              <label key={header} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.stratificationColumns.includes(header)}
                  onChange={() => toggleStratificationColumn(header)}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                <span className="text-sm text-gray-700">{header}</span>
              </label>
            ))}
          </div>
          
          {config.stratificationColumns.length > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              Selected: {config.stratificationColumns.join(', ')}
            </p>
          )}
        </div>

        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Weight Summary</h4>
          <div className="space-y-1 text-sm text-gray-600">
            <p>Weight variable: {config.weightColumn || 'None (unweighted)'}</p>
            <p>Stratification: {config.stratificationColumns.length} variables selected</p>
            <p>
              Analysis type: {config.weightColumn ? 'Complex survey design' : 'Simple random sampling'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};