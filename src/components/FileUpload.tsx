import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import Papa from 'papaparse';
import { SurveyData } from '../types/survey';

interface FileUploadProps {
  onDataLoaded: (data: SurveyData) => void;
  onError: (error: string) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onDataLoaded, onError }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      onError('Please upload a CSV file');
      return;
    }

    Papa.parse(file, {
      complete: (result) => {
        try {
          if (result.errors.length > 0) {
            onError(`CSV parsing error: ${result.errors[0].message}`);
            return;
          }

          const [headers, ...dataRows] = result.data as string[][];
          
          if (!headers || headers.length === 0) {
            onError('No headers found in CSV file');
            return;
          }

          const rows = dataRows
            .filter(row => row.length > 1 && row.some(cell => cell.trim() !== ''))
            .map(row => {
              const rowData: Record<string, any> = {};
              headers.forEach((header, index) => {
                let value = row[index] || '';
                
                // Try to parse as number
                const numValue = parseFloat(value);
                if (!isNaN(numValue) && value.trim() !== '') {
                  value = numValue;
                }
                
                rowData[header.trim()] = value;
              });
              return rowData;
            });

          if (rows.length === 0) {
            onError('No data rows found in CSV file');
            return;
          }

          const surveyData: SurveyData = {
            headers: headers.map(h => h.trim()),
            rows,
            filename: file.name,
            uploadTime: new Date()
          };

          onDataLoaded(surveyData);
        } catch (error) {
          onError(`Error processing file: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      },
      header: false,
      skipEmptyLines: true,
      error: (error) => {
        onError(`File parsing error: ${error.message}`);
      }
    });
  }, [onDataLoaded, onError]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB limit
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive && !isDragReject 
            ? 'border-blue-400 bg-blue-50' 
            : isDragReject 
            ? 'border-red-400 bg-red-50' 
            : 'border-gray-300 bg-gray-50 hover:border-blue-300 hover:bg-blue-50'
          }
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {isDragReject ? (
            <AlertCircle className="h-12 w-12 text-red-400" />
          ) : (
            <div className="relative">
              <Upload className="h-12 w-12 text-gray-400" />
              <FileText className="h-6 w-6 text-blue-500 absolute -bottom-1 -right-1" />
            </div>
          )}
          
          <div>
            <p className="text-lg font-medium text-gray-700">
              {isDragActive 
                ? isDragReject 
                  ? 'File type not supported'
                  : 'Drop your CSV file here'
                : 'Upload Survey Data'
              }
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {isDragReject 
                ? 'Only CSV files are supported'
                : 'Drag and drop your CSV file here, or click to browse'
              }
            </p>
          </div>
          
          {!isDragActive && (
            <button
              type="button"
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium"
            >
              Choose File
            </button>
          )}
        </div>
        
        <div className="mt-6 text-xs text-gray-400 border-t pt-4">
          <p>Supported format: CSV files up to 10MB</p>
          <p>Ensure your CSV has headers in the first row</p>
        </div>
      </div>
    </div>
  );
};