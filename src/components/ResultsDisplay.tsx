import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, Download, FileText, Info } from 'lucide-react';
import { EstimateResult } from '../types/survey';

interface ResultsDisplayProps {
  estimates: EstimateResult[];
  descriptiveStats: any[];
  finalData: Array<Record<string, any>>; // Cleaned dataset rows
  onDownloadPDF: () => void;
  onDownloadHTML: () => void;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  estimates,
  descriptiveStats,
  finalData,
  onDownloadPDF,
  onDownloadHTML
}) => {
  const chartData = estimates.map(est => ({
    variable: est.variable,
    estimate: est.estimate,
    lower: est.confidenceInterval[0],
    upper: est.confidenceInterval[1],
    marginOfError: est.marginOfError
  }));

  return (
    <div className="space-y-6">
      {/* Results Header */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
                Analysis Results
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                Statistical estimates with confidence intervals and margins of error
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={onDownloadHTML}
                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              >
                <FileText className="h-4 w-4" />
                <span>HTML Report</span>
              </button>
              <button
                onClick={onDownloadPDF}
                className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>PDF Report</span>
              </button>
            </div>
          </div>
        </div>
        {/* Summary Statistics */}
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">{estimates.length}</div>
              <div className="text-sm text-gray-600">Variables Analyzed</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">
                {estimates.reduce((sum, est) => sum + est.sampleSize, 0)}
              </div>
              <div className="text-sm text-gray-600">Total Sample Size</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-600">95%</div>
              <div className="text-sm text-gray-600">Confidence Level</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-orange-600">
                {(estimates.reduce((sum, est) => sum + est.marginOfError, 0) / estimates.length).toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Avg Margin of Error</div>
            </div>
          </div>
        </div>
      </div>

      {/* Estimates Chart */}
      {estimates.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h4 className="text-lg font-semibold text-gray-900">Estimates with Confidence Intervals</h4>
          </div>
          <div className="p-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="variable" />
                  <YAxis />
                  <Tooltip
                    formatter={(value, name) => [
                      typeof value === 'number' ? value.toFixed(4) : value,
                      name
                    ]}
                  />
                  <Bar dataKey="estimate" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Results Table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h4 className="text-lg font-semibold text-gray-900">Statistical Estimates</h4>
          <p className="text-sm text-gray-600 mt-1">
            Detailed estimates with standard errors and confidence intervals
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Variable
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Estimate
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Standard Error
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Margin of Error
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  95% Confidence Interval
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Sample Size
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {estimates.map(estimate => (
                <tr key={estimate.variable} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {estimate.variable}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {estimate.estimate.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {estimate.standardError.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ±{estimate.marginOfError.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    [{estimate.confidenceInterval[0].toFixed(4)}, {estimate.confidenceInterval[1].toFixed(4)}]
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {estimate.sampleSize.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Descriptive Statistics */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h4 className="text-lg font-semibold text-gray-900">Descriptive Statistics</h4>
          <p className="text-sm text-gray-600 mt-1">
            Summary statistics for all analyzed variables
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Variable
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Count
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Mean
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Median
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Std Dev
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Min
                </th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Max
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {descriptiveStats.map(stat => (
                <tr key={stat.column} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {stat.column}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.count.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.mean}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.median}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.std}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.min}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stat.max}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Final Cleaned Data Preview */}
      {finalData && finalData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden mt-6">
          <div className="p-6 border-b border-gray-200">
            <h4 className="text-lg font-semibold text-gray-900">Final Cleaned Data (Preview)</h4>
            <p className="text-sm text-gray-600 mt-1">
              Showing the first {Math.min(finalData.length, 20)} rows after cleaning
            </p>
          </div>
          <div className="overflow-x-auto max-h-96">
            <table className="w-full">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  {Object.keys(finalData[0]).map(column => (
                    <th
                      key={column}
                      className="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {finalData.slice(0, 20).map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    {Object.keys(row).map(col => (
                      <td key={col} className="px-4 py-2 whitespace-nowrap text-sm text-gray-900">
                        {String(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Analysis Notes */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <div className="flex items-start space-x-3">
          <Info className="h-5 w-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-800">
            <p className="font-medium">Analysis Notes</p>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>Confidence intervals are calculated using the t-distribution</li>
              <li>Standard errors account for survey weights when specified</li>
              <li>Margin of error represents half the width of the confidence interval</li>
              <li>Results assume the survey design specified in the weight configuration</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
