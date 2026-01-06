import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, BarChart3, Database, Sparkles, TrendingDown } from 'lucide-react';
import { SurveyData } from '../types/survey';

interface DataPreviewProps {
  data: SurveyData;
  cleanedData?: Record<string, any>[] | null;
  showCleaned?: boolean;
}

export const DataPreview: React.FC<DataPreviewProps> = ({ data, cleanedData, showCleaned = false }) => {
  const [currentPage, setCurrentPage] = useState(0);
  const [viewMode, setViewMode] = useState<'original' | 'cleaned'>('original');
  const rowsPerPage = 10;
  console.log("data:", data);
  console.log("cleanedData:", cleanedData);

  const displayData = Array.isArray(viewMode === 'cleaned' && cleanedData ? cleanedData : data.rows)
  ? (viewMode === 'cleaned' && cleanedData ? cleanedData : data.rows)
  : Object.values(viewMode === 'cleaned' && cleanedData ? cleanedData : data.rows || []);

  const totalPages = Math.ceil(displayData.length / rowsPerPage);

  const currentRows = displayData.slice(
    currentPage * rowsPerPage,
    (currentPage + 1) * rowsPerPage
  );

  const getColumnType = (header: string): string => {
    const sampleValues = data.rows.slice(0, 100).map(row => row[header]);
    const numericValues = sampleValues.filter(val => 
      !isNaN(parseFloat(val)) && isFinite(val)
    ).length;
    
    return numericValues > sampleValues.length * 0.7 ? 'numeric' : 'text';
  };

  const getColumnStats = (header: string) => {
    const sourceData = viewMode === 'cleaned' && cleanedData ? cleanedData : data.rows;
    const values = sourceData.map(row => row[header]).filter(val => 
      val !== null && val !== undefined && val !== ''
    );
    
    const missingCount = sourceData.length - values.length;
    const missingPercentage = (missingCount / sourceData.length) * 100;
    
    if (getColumnType(header) === 'numeric') {
      const numericValues = values.map(val => parseFloat(val)).filter(val => !isNaN(val));
      if (numericValues.length > 0) {
        const min = Math.min(...numericValues);
        const max = Math.max(...numericValues);
        const mean = numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
        
        return {
          type: 'numeric',
          missing: missingCount,
          missingPercentage,
          min: min.toFixed(2),
          max: max.toFixed(2),
          mean: mean.toFixed(2),
          unique: new Set(numericValues).size
        };
      }
    }
    
    return {
      type: 'text',
      missing: missingCount,
      missingPercentage,
      unique: new Set(values).size,
      mostCommon: Object.entries(
        values.reduce((acc: Record<string, number>, val) => {
          acc[val] = (acc[val] || 0) + 1;
          return acc;
        }, {})
      ).sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A'
    };
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Database className="h-6 w-6 text-blue-600" />
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{data.filename}</h3>
              <p className="text-sm text-gray-600">
                {viewMode === 'original' 
                  ? `${data.rows.length.toLocaleString()} rows × ${data.headers.length} columns`
                  : `${displayData.length.toLocaleString()} rows × ${data.headers.length} columns (cleaned)`
                }
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {showCleaned && cleanedData && (
              <div className="flex bg-white rounded-lg border border-gray-200 p-1">
                <button
                  onClick={() => {
                    setViewMode('original');
                    setCurrentPage(0);
                  }}
                  className={`px-3 py-1 text-sm rounded-md transition-colors flex items-center space-x-1 ${
                    viewMode === 'original'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Database className="h-3 w-3" />
                  <span>Original</span>
                </button>
                <button
                  onClick={() => {
                    setViewMode('cleaned');
                    setCurrentPage(0);
                  }}
                  className={`px-3 py-1 text-sm rounded-md transition-colors flex items-center space-x-1 ${
                    viewMode === 'cleaned'
                      ? 'bg-green-600 text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <Sparkles className="h-3 w-3" />
                  <span>Cleaned</span>
                </button>
              </div>
            )}
            <div className="text-right">
              <p className="text-sm text-gray-500">
                {viewMode === 'cleaned' ? 'Cleaned Data' : 'Uploaded'}
              </p>
              <p className="text-sm font-medium text-gray-900">
                {viewMode === 'cleaned' && cleanedData
                  ? `${((cleanedData.length / data.rows.length) * 100).toFixed(1)}% retained`
                  : data.uploadTime.toLocaleString()
                }
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Column Statistics */}
      <div className="p-6 border-b border-gray-200 bg-gray-50">
        <h4 className="text-sm font-semibold text-gray-900 mb-4 flex items-center">
          {viewMode === 'cleaned' ? (
            <>
              <Sparkles className="h-4 w-4 mr-2 text-green-600" />
              Column Statistics (After Cleaning)
            </>
          ) : (
            <>
              <BarChart3 className="h-4 w-4 mr-2" />
              Column Statistics (Original)
            </>
          )}
        </h4>
        {viewMode === 'cleaned' && cleanedData && (
          <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2 text-sm text-green-800">
              <TrendingDown className="h-4 w-4" />
              <span className="font-medium">Data Cleaning Applied:</span>
              <span>KNN Imputation + Winsorization (5th-95th percentiles)</span>
            </div>
            <div className="mt-1 text-xs text-green-700">
              Missing values filled using 5 nearest neighbors • Outliers capped to percentile bounds
            </div>
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.headers.slice(0, 6).map(header => {
            const stats = getColumnStats(header);
            return (
              <div key={header} className="bg-white p-4 rounded-lg border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="font-medium text-gray-900 text-sm truncate" title={header}>
                    {header}
                  </h5>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    stats.type === 'numeric' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {stats.type}
                  </span>
                </div>
                <div className="space-y-1 text-xs text-gray-600">
                  <div className="flex justify-between">
                    <span>Missing:</span>
                    <span className={stats.missingPercentage > 20 ? 'text-red-600' : ''}>
                      {stats.missing} ({stats.missingPercentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Unique:</span>
                    <span>{stats.unique}</span>
                  </div>
                  {stats.type === 'numeric' ? (
                    <>
                      <div className="flex justify-between">
                        <span>Range:</span>
                        <span>{stats.min} - {stats.max}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Mean:</span>
                        <span>{stats.mean}</span>
                      </div>
                    </>
                  ) : (
                    <div className="flex justify-between">
                      <span>Most common:</span>
                      <span className="truncate max-w-16" title={stats.mostCommon}>
                        {stats.mostCommon}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        {data.headers.length > 6 && (
          <p className="text-sm text-gray-500 mt-4">
            Showing statistics for first 6 columns. {data.headers.length - 6} more columns available.
          </p>
        )}
      </div>

      {/* Data Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {data.headers.map(header => (
                <th key={header} className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider whitespace-nowrap">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {currentRows.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50 transition-colors">
                {data.headers.map(header => (
                  <td key={header} className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                    {row[header] === null || row[header] === undefined || row[header] === '' ? (
                      <span className="text-gray-400 italic">—</span>
                    ) : (
                      String(row[header])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-gray-50 px-6 py-3 border-t border-gray-200 flex items-center justify-between">
          <div className="flex items-center text-sm text-gray-600">
            <span>
              Showing {currentPage * rowsPerPage + 1} to {Math.min((currentPage + 1) * rowsPerPage, displayData.length)} of {displayData.length} rows
              {viewMode === 'cleaned' && cleanedData && (
                <span className="ml-2 text-green-600 font-medium">
                  ({((cleanedData.length / data.rows.length) * 100).toFixed(1)}% of original)
                </span>
              )}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0}
              className="px-3 py-1 rounded-md border border-gray-300 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </button>
            <span className="text-sm text-gray-600">
              Page {currentPage + 1} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
              disabled={currentPage === totalPages - 1}
              className="px-3 py-1 rounded-md border border-gray-300 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};