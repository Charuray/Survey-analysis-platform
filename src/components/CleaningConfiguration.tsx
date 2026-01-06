import React, { useState } from 'react';
import { Plus, Trash2, Settings, AlertTriangle } from 'lucide-react';
import { CleaningConfig } from '../types/survey';
interface CleaningConfigurationProps {
  headers: string[];
  config: CleaningConfig;
  onConfigChange: (config: CleaningConfig) => void;
}
// Utility for threshold defaults
const getDefaultThreshold = (method: string) => {
  // These defaults are based on widely used statistical conventions
  if (method === 'iqr') return 1.5;
  if (method === 'z-score') return 3.0;
  if (method === 'winsorization') return { lower: 0.05, upper: 0.95 };
  if (method === 'isolation-forest') return 'auto';
  return '';
};
export const CleaningConfiguration: React.FC<CleaningConfigurationProps> = ({
  headers,
  config,
  onConfigChange
}) => {
  const [activeTab, setActiveTab] = useState<'imputation' | 'outliers' | 'rules'>('imputation');
  const numericHeaders = headers; // In a real app, filter numeric columns only
  // Update config utility
  const updateConfig = (section: keyof CleaningConfig, updates: any) => {
    onConfigChange({
      ...config,
      [section]: {
        ...config[section],
        ...updates
      }
    });
  };
  // Add new validation rule
  const addRule = () => {
    const newRule = {
      id: Date.now().toString(),
      column: headers[0] || '',
      condition: 'greater-than' as const,
      value: '',
      action: 'flag' as const
    };
    onConfigChange({
      ...config,
      rules: [...config.rules, newRule]
    });
  };
  // Update a specific rule
  const updateRule = (ruleId: string, updates: any) => {
    onConfigChange({
      ...config,
      rules: config.rules.map(rule =>
        rule.id === ruleId ? { ...rule, ...updates } : rule
      )
    });
  };
  // Remove a specific rule
  const removeRule = (ruleId: string) => {
    onConfigChange({
      ...config,
      rules: config.rules.filter(rule => rule.id !== ruleId)
    });
  };
  // Tabs config
  const tabs = [
    { id: 'imputation', label: 'Missing Values', icon: Settings },
    { id: 'outliers', label: 'Outliers', icon: AlertTriangle },
    { id: 'rules', label: 'Validation Rules', icon: Plus }
  ];
  // Outlier method change handler for auto threshold
  const handleOutlierMethodChange = (method: string) => {
    // Automatically update threshold when method changes
    if (method === 'winsorization') {
      updateConfig('outliers', {
        method,
        threshold: getDefaultThreshold(method)
      });
    } else {
      updateConfig('outliers', {
        method,
        threshold: getDefaultThreshold(method)
      });
    }
  };
  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <Settings className="h-5 w-5 mr-2 text-emerald-600" />
          Data Cleaning Configuration
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Configure data cleaning methods and validation rules
        </p>
      </div>
      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex space-x-8 px-6">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-4 border-b-2 font-medium text-sm transition-colors flex items-center space-x-2 ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="p-6">
        {activeTab === 'imputation' && (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Imputation Method
              </label>
              <select
                value={config.imputation.method}
                onChange={(e) => updateConfig('imputation', { method: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="mean">Mean Imputation</option>
                <option value="median">Median Imputation</option>
                <option value="mode">Mode Imputation</option>
                <option value="knn">KNN Imputation (K=5)</option>
                <option value="forward-fill">Forward Fill</option>
                <option value="backward-fill">Backward Fill</option>
                <option value="remove">Remove Missing Records</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                KNN uses 5 nearest neighbors with Euclidean distance for intelligent imputation
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Apply to Columns
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
                {headers.map(header => (
                  <label key={header} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={config.imputation.columns.includes(header)}
                      onChange={(e) => {
                        const columns = e.target.checked
                          ? [...config.imputation.columns, header]
                          : config.imputation.columns.filter(col => col !== header);
                        updateConfig('imputation', { columns });
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{header}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
        {activeTab === 'outliers' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Detection Method
                </label>
                <select
                  value={config.outliers.method}
                  onChange={(e) => handleOutlierMethodChange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="iqr">Interquartile Range (IQR)</option>
                  <option value="z-score">Z-Score</option>
                  <option value="winsorization">Winsorization</option>
                  <option value="isolation-forest">Isolation Forest</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Threshold
                </label>
                {config.outliers.method === 'winsorization' ? (
                  <div className="flex items-center space-x-2">
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.5"
                      value={
                        typeof config.outliers.threshold === 'object'
                          ? config.outliers.threshold.lower
                          : 0.05
                      }
                      onChange={(e) => {
                        const newThreshold = {
                          lower: parseFloat(e.target.value),
                          upper: typeof config.outliers.threshold === 'object' 
                            ? config.outliers.threshold.upper 
                            : 0.95
                        };
                        updateConfig('outliers', { threshold: newThreshold });
                      }}
                      className="w-20 px-2 py-1 border border-gray-300 rounded-lg focus:ring-1 focus:ring-blue-500"
                    />
                    <span className="text-gray-500 text-sm">Lower %</span>
                    <input
                      type="number"
                      step="0.01"
                      min="0.5"
                      max="1"
                      value={
                        typeof config.outliers.threshold === 'object'
                          ? config.outliers.threshold.upper
                          : 0.95
                      }
                      onChange={(e) => {
                        const newThreshold = {
                          lower: typeof config.outliers.threshold === 'object' 
                            ? config.outliers.threshold.lower 
                            : 0.05,
                          upper: parseFloat(e.target.value)
                        };
                        updateConfig('outliers', { threshold: newThreshold });
                      }}
                      className="w-20 px-2 py-1 border border-gray-300 rounded-lg focus:ring-1 focus:ring-blue-500"
                    />
                    <span className="text-gray-500 text-sm">Upper %</span>
                  </div>
                ) : (
                  <input
                    type="number"
                    step="0.1"
                    value={
                      typeof config.outliers.threshold === 'number'
                        ? config.outliers.threshold
                        : ''
                    }
                    disabled // Automatic threshold so editing is disabled
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-100"
                    placeholder={
                      config.outliers.method === 'iqr'
                        ? 'e.g., 1.5'
                        : config.outliers.method === 'z-score'
                        ? 'e.g., 3.0'
                        : config.outliers.method === 'isolation-forest'
                        ? 'auto'
                        : ''
                    }
                  />
                )}
                <p className="text-xs text-gray-500 mt-1">
                  {config.outliers.method === 'winsorization' 
                    ? 'Adjust percentiles for winsorization (e.g., 0.05 = 5th percentile)'
                    : 'Threshold value is set automatically for classic methods'
                  }
                </p>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Action for Outliers
              </label>
              <select
                value={config.outliers.action}
                onChange={(e) => updateConfig('outliers', { action: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="cap">Cap to Threshold</option>
                <option value="remove">Remove Outliers</option>
                <option value="transform">Log Transform</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Cap to Threshold (Winsorization) is recommended for preserving data
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Apply to Columns
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-3">
                {numericHeaders.map(header => (
                  <label key={header} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={config.outliers.columns.includes(header)}
                      onChange={(e) => {
                        const columns = e.target.checked
                          ? [...config.outliers.columns, header]
                          : config.outliers.columns.filter(col => col !== header);
                        updateConfig('outliers', { columns });
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{header}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}
        {activeTab === 'rules' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-gray-900">Validation Rules</h4>
              <button
                onClick={addRule}
                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              >
                <Plus className="h-4 w-4" />
                <span>Add Rule</span>
              </button>
            </div>
            {config.rules.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No validation rules configured</p>
                <p className="text-sm">Add rules to validate data quality</p>
              </div>
            ) : (
              <div className="space-y-4">
                {config.rules.map(rule => (
                  <div key={rule.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          Column
                        </label>
                        <select
                          value={rule.column}
                          onChange={(e) => updateRule(rule.id, { column: e.target.value })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                        >
                          {headers.map(header => (
                            <option key={header} value={header}>{header}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          Condition
                        </label>
                        <select
                          value={rule.condition}
                          onChange={(e) => updateRule(rule.id, { condition: e.target.value })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="greater-than">Greater than</option>
                          <option value="less-than">Less than</option>
                          <option value="equals">Equals</option>
                          <option value="not-equals">Not equals</option>
                          <option value="range">In range</option>
                          <option value="pattern">Matches pattern</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          Value
                        </label>
                        <input
                          type="text"
                          value={rule.value}
                          onChange={(e) => updateRule(rule.id, { value: e.target.value })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                          placeholder="Enter value..."
                        />
                      </div>
                      <div className="flex items-end space-x-2">
                        <div className="flex-1">
                          <label className="block text-xs font-medium text-gray-600 mb-1">
                            Action
                          </label>
                          <select
                            value={rule.action}
                            onChange={(e) => updateRule(rule.id, { action: e.target.value })}
                            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                          >
                            <option value="flag">Flag</option>
                            <option value="remove">Remove</option>
                            <option value="transform">Transform</option>
                          </select>
                        </div>
                        <button
                          onClick={() => removeRule(rule.id)}
                          className="p-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};